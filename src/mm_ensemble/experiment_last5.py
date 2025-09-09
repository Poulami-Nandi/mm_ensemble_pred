#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiment_last5.py

Train on the last 1 year (excluding the last 5 trading days) and predict the last 5 days:
- Config A: all features (prices + trends + news + fundamentals)
- Config B: OHLCV-only features

Per ticker, stores under --outdir/<TICKER>/:
  - predictions CSVs (last 5)
  - metrics JSON
  - model artifacts (pkl + feature schema/medians)
  - plots:
      * actual_vs_pred_all.png
      * actual_vs_pred_ohlcv.png
      * compare_all_vs_ohlcv.png
"""

import argparse, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR

# -------------------- helpers --------------------

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _all_numeric(df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    cand = [c for c in df.columns if c not in drop_cols]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]

def _impute_matrix(X: pd.DataFrame, med: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if med is None:
        med = X.median(numeric_only=True)
    X = X.copy()
    X = X.fillna(med)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    med = med.fillna(0.0)
    return X, med

def _select_ohlcv_features(df: pd.DataFrame, target: str) -> List[str]:
    """
    Keep only OHLCV-derived columns. We include:
      - raw OHLCV: open, high, low, close, adj_close, volume
      - price features created in build_dataset.py: ret_*, logret_*, ma_*, vol_*, ma_5_over_20, ma_20_over_50
    Drop everything that looks like trends/news/fundamentals.
    """
    keep_prefixes = ("open","high","low","close","adj_close","volume","ret_","logret_","ma_","vol_")
    keep_exact = {"ma_5_over_20", "ma_20_over_50"}
    drop_cols = {"ds", target}
    drop_cols |= {c for c in df.columns if c.lower().startswith("target_") and c != target}

    feats = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c in keep_exact or c.startswith(keep_prefixes):
            feats.append(c)
    # Ensure uniqueness/order
    feats = list(dict.fromkeys(feats))
    return feats

def _select_all_features(df: pd.DataFrame, target: str) -> List[str]:
    drop_cols = {"ds", target}
    drop_cols |= {c for c in df.columns if c.lower().startswith("target_") and c != target}
    return _all_numeric(df, list(drop_cols))

def _time_split_last_year(df: pd.DataFrame, n_holdout: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter to the last 365 days (inclusive) and split:
      - train: all but last n_holdout rows
      - test: last n_holdout rows
    Assumes df is daily, sorted by ds.
    """
    if df.empty:
        raise RuntimeError("Empty dataframe for split")
    end = df["ds"].max()
    start = end - pd.Timedelta(days=365)
    sub = df.loc[df["ds"].between(start, end)].copy().sort_values("ds").reset_index(drop=True)
    if len(sub) <= n_holdout:
        raise RuntimeError(f"Not enough rows ({len(sub)}) in last year to hold out {n_holdout} days")
    train = sub.iloc[: -n_holdout].copy()
    test  = sub.iloc[-n_holdout: ].copy()
    return train, test

def _build_val_from_train(train: pd.DataFrame, min_val: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve out a small validation tail from train for early stopping.
    If training is too short, use 10% as val with minimum=min_val (capped).
    """
    n = len(train)
    n_val = max(min_val, int(0.1 * n))
    n_val = min(n_val, max(1, n // 4))  # cap at 25% to keep most data for fit
    if n <= n_val + 5:
        # if extremely short, make tiny val
        n_val = max(1, n // 5)
    tr = train.iloc[: n - n_val].copy()
    va = train.iloc[n - n_val :].copy()
    return tr, va

# -------------------- models --------------------

def _fit_model_xgb(Xtr, ytr, Xva, yva):
    """
    Cross-version xgboost fitting:
      - try callbacks.EarlyStopping
      - try early_stopping_rounds
      - fallback: plain fit
    """
    from xgboost import XGBRegressor  # type: ignore
    model = XGBRegressor(
        n_estimators=1200, learning_rate=0.03, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, n_jobs=0, eval_metric="rmse"
    )
    ok = False
    try:
        from xgboost.callback import EarlyStopping  # type: ignore
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                  callbacks=[EarlyStopping(rounds=80, save_best=True)])
        ok = True
    except Exception:
        pass
    if not ok:
        try:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                      early_stopping_rounds=80)
            ok = True
        except Exception:
            pass
    if not ok:
        model.fit(Xtr, ytr)
    return model, "xgb"

def _fit_model_fallback(Xtr, ytr, Xva, yva):
    """
    scikit-learn fallback (handles environments without xgboost).
    We use HistGradientBoostingRegressor (accepts NaNs), fallback to GradientBoostingRegressor if needed.
    """
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=None, learning_rate=0.05, max_iter=600, random_state=42
        )
        model.fit(Xtr, ytr)
        return model, "hgb"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        # GBM does NOT accept NaNs — but our imputation eliminates them
        model = GradientBoostingRegressor(n_estimators=800, learning_rate=0.05, max_depth=3, random_state=42)
        Xfull = pd.concat([Xtr, Xva], axis=0)
        yfull = pd.concat([ytr, yva], axis=0)
        model.fit(Xfull, yfull)
        return model, "gbm"

def _fit_model_safe(Xtr, ytr, Xva, yva):
    try:
        return _fit_model_xgb(Xtr, ytr, Xva, yva)
    except Exception:
        return _fit_model_fallback(Xtr, ytr, Xva, yva)

# -------------------- training wrapper --------------------

@dataclass
class FitArtifacts:
    model_kind: str
    model_path: Path
    feats_path: Path
    med_path: Path
    metrics_path: Path
    preds_path: Path
    plot_path: Path

def _prepare_xy(df: pd.DataFrame, feats: List[str], target: str, train_medians: Optional[pd.Series] = None):
    X = df[feats].copy()
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    X, med = _impute_matrix(X, train_medians)
    return X, y, med

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not m.any(): return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

def _fit_and_predict(
    ticker: str,
    df_full: pd.DataFrame,
    feats: List[str],
    target: str,
    out_subdir: Path,
    config_name: str,
    holdout_days: int = 5
) -> FitArtifacts:
    """
    Train on last year excluding last 5, predict last 5, save artifacts & plot.
    """
    _ensure_dir(out_subdir)

    # Split
    train_all, test5 = _time_split_last_year(df_full, n_holdout=holdout_days)
    train_tr, train_va = _build_val_from_train(train_all)

    # Prepare XY
    Xtr, ytr, med = _prepare_xy(train_tr, feats, target, None)
    Xva, yva, _   = _prepare_xy(train_va, feats, target, med)
    Xte, yte, _   = _prepare_xy(test5,    feats, target, med)

    # Fit
    model, model_kind = _fit_model_safe(Xtr, ytr, Xva, yva)

    # Predict last 5
    yhat_last5 = model.predict(Xte)

    # Metrics
    rmse = _rmse(yte.values, yhat_last5)
    mae  = float(np.nanmean(np.abs(yte.values - yhat_last5))) if len(yte) else float("nan")

    # Save artifacts
    from joblib import dump
    model_path = out_subdir / f"model_{config_name}.pkl"
    feats_path = out_subdir / f"feature_names_{config_name}.json"
    med_path   = out_subdir / f"feature_medians_{config_name}.json"
    preds_path = out_subdir / f"pred_last5_{config_name}.csv"
    metrics_path = out_subdir / f"metrics_{config_name}.json"
    plot_path = out_subdir / f"actual_vs_pred_{config_name}.png"

    dump(model, model_path)
    Path(feats_path).write_text(json.dumps(feats, indent=2), encoding="utf-8")
    med_dict = {k: (0.0 if pd.isna(v) else float(v)) for k, v in med.items()}
    Path(med_path).write_text(json.dumps(med_dict, indent=2), encoding="utf-8")

    preds_df = test5[["ds"]].copy()
    preds_df["y_true"] = yte.values
    preds_df["y_pred"] = yhat_last5
    preds_df.to_csv(preds_path, index=False)

    metrics = {
        "ticker": ticker,
        "config": config_name,
        "model_kind": model_kind,
        "target": target,
        "rmse_last5": rmse,
        "mae_last5": mae,
        "n_train": int(len(train_all)),
        "n_val": int(len(train_va)),
        "n_test5": int(len(test5)),
        "train_start": str(train_all["ds"].min().date()),
        "train_end": str(train_all["ds"].max().date()),
        "test5_start": str(test5["ds"].min().date()),
        "test5_end": str(test5["ds"].max().date()),
    }
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plot actual vs predicted (last 5)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(preds_df["ds"], preds_df["y_true"], label="Actual")
    ax.plot(preds_df["ds"], preds_df["y_pred"], label="Predicted")
    ax.axhline(0.0, linestyle="--")
    ax.set_title(f"{ticker} — {config_name} — Last 5 days\nRMSE={rmse:.6f}")
    ax.set_xlabel("Date"); ax.set_ylabel(target)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return FitArtifacts(
        model_kind=model_kind,
        model_path=model_path,
        feats_path=feats_path,
        med_path=med_path,
        metrics_path=metrics_path,
        preds_path=preds_path,
        plot_path=plot_path,
    )

# -------------------- main runner --------------------

@dataclass
class RunCfg:
    root: Path
    outdir: Path
    tickers: List[str]
    target: str
    holdout_days: int

def _load_dataset(root: Path, ticker: str) -> pd.DataFrame:
    p = root / ticker / "dataset.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset for {ticker}: {p}")
    df = pd.read_parquet(p)
    df["ds"] = _norm_ds(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df

def _combined_compare_plot(ticker: str, preds_all_csv: Path, preds_ohlcv_csv: Path, out_path: Path, target: str):
    df_all = pd.read_csv(preds_all_csv)
    df_ohl = pd.read_csv(preds_ohlcv_csv)
    m = pd.merge(df_all, df_ohl, on="ds", suffixes=("_all", "_ohlcv"))
    # ensure y_true agrees (take from _all)
    m["y_true"] = pd.to_numeric(m["y_true_all"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(pd.to_datetime(m["ds"]), m["y_true"], label="Actual")
    ax.plot(pd.to_datetime(m["ds"]), m["y_pred_ohlcv"], label="Pred (OHLCV)")
    ax.plot(pd.to_datetime(m["ds"]), m["y_pred_all"], label="Pred (All inputs)")
    ax.axhline(0.0, linestyle="--")
    ax.set_title(f"{ticker} — Compare last 5 days")
    ax.set_xlabel("Date"); ax.set_ylabel(target)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run(cfg: RunCfg):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_out = cfg.outdir
    _ensure_dir(base_out)

    summary = {"target": cfg.target, "holdout_days": cfg.holdout_days, "tickers": {}}

    for t in cfg.tickers:
        tdir = base_out / t
        _ensure_dir(tdir)

        # Load dataset
        df = _load_dataset(cfg.root, t)

        # Feature sets
        feats_all   = _select_all_features(df, cfg.target)
        feats_ohlcv = _select_ohlcv_features(df, cfg.target)

        # Fit & predict: all inputs
        art_all = _fit_and_predict(t, df, feats_all, cfg.target, tdir, config_name="all_inputs", holdout_days=cfg.holdout_days)
        # Fit & predict: ohlcv only
        art_ohl = _fit_and_predict(t, df, feats_ohlcv, cfg.target, tdir, config_name="ohlcv_only", holdout_days=cfg.holdout_days)

        # Combined compare plot
        combined_png = tdir / "compare_all_vs_ohlcv.png"
        _combined_compare_plot(t, art_all.preds_path, art_ohl.preds_path, combined_png, cfg.target)

        # Record per-ticker in summary
        summary["tickers"][t] = {
            "all_inputs": {
                "model_kind": art_all.model_kind,
                "metrics": json.loads(Path(art_all.metrics_path).read_text(encoding="utf-8")),
                "paths": {
                    "model": str(art_all.model_path),
                    "feats": str(art_all.feats_path),
                    "medians": str(art_all.med_path),
                    "preds_csv": str(art_all.preds_path),
                    "plot_png": str(art_all.plot_path),
                }
            },
            "ohlcv_only": {
                "model_kind": art_ohl.model_kind,
                "metrics": json.loads(Path(art_ohl.metrics_path).read_text(encoding="utf-8")),
                "paths": {
                    "model": str(art_ohl.model_path),
                    "feats": str(art_ohl.feats_path),
                    "medians": str(art_ohl.med_path),
                    "preds_csv": str(art_ohl.preds_path),
                    "plot_png": str(art_ohl.plot_path),
                }
            },
            "compare_plot": str(combined_png),
        }

    # Write run summary
    summary_path = base_out / "summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    for t in cfg.tickers:
        print(f"- {t}:")
        print(f"    All inputs plot   -> {summary['tickers'][t]['all_inputs']['paths']['plot_png']}")
        print(f"    OHLCV-only plot   -> {summary['tickers'][t]['ohlcv_only']['paths']['plot_png']}")
        print(f"    Compare plot      -> {summary['tickers'][t]['compare_plot']}")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root where data/<TICKER>/dataset.parquet lives")
    ap.add_argument("--outdir", default="data/runs/last5", help="Output directory (NOT Colab root)")
    ap.add_argument("--tickers", nargs="+", default=["SBUX","PFE"])
    ap.add_argument("--target", default="target_return_1d")
    ap.add_argument("--holdout-days", type=int, default=5)
    args = ap.parse_args()

    cfg = RunCfg(
        root=Path(args.root),
        outdir=Path(args.outdir),
        tickers=[t.upper() for t in args.tickers],
        target=args.target,
        holdout_days=args.holdout_days,
    )
    run(cfg)

if __name__ == "__main__":
    main()