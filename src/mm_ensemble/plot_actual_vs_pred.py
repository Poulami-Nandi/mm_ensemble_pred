#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------- helpers ---------------------

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _load_parquet(p: Path):
    if not p.exists():
        return None
    return pd.read_parquet(p)

def _all_numeric(df: pd.DataFrame, drop_cols):
    cols = [c for c in df.columns if c not in drop_cols]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def _impute_matrix(X: pd.DataFrame, med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if med is None:
        med = X.median(numeric_only=True)
    X = X.copy()
    X = X.fillna(med)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, med

def _prepare_xy_auto(df: pd.DataFrame, target: str, medians: pd.Series | None = None):
    drop_cols = ["ds", target] + [c for c in df.columns if c.lower().startswith("target_") and c != target]
    feats = _all_numeric(df, drop_cols)
    X = df[feats].copy()
    y = df[target].astype(float) if target in df.columns else None
    X, med_used = _impute_matrix(X, medians)
    if y is not None:
        y = y.fillna(0.0)
    return X, y, feats, med_used

def _prepare_xy_with_feats(df: pd.DataFrame, target: str, feats: list[str], med: pd.Series | None):
    X = df.copy()
    for f in feats:
        if f not in X.columns:
            X[f] = 0.0
    X = X[feats].copy()
    if med is not None:
        med = med.reindex(feats)
    X, med_used = _impute_matrix(X, med)
    y = df[target].astype(float).fillna(0.0) if target in df.columns else None
    return X, y, feats, med_used

def _try_load_artifacts(tdir: Path):
    model = None; feats_saved = None; med_saved = None
    try:
        from joblib import load  # type: ignore
        if (tdir / "model_xgb.pkl").exists():
            model = load(tdir / "model_xgb.pkl")
        elif (tdir / "model_gbm.pkl").exists():
            model = load(tdir / "model_gbm.pkl")
    except Exception:
        model = None
    feats_path = tdir / "feature_names.json"
    if feats_path.exists():
        try:
            feats_saved = json.loads(feats_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    med_path = tdir / "feature_medians.json"
    if med_path.exists():
        try:
            med_saved = pd.Series(json.loads(med_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return model, feats_saved, med_saved

def _fit_quick_model(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str,
                     feats: list[str] | None, med: pd.Series | None):
    tr = train_df.sort_values("ds").copy()
    va = val_df.sort_values("ds").copy()
    if feats is not None:
        Xtr, ytr, feats_used, med_used = _prepare_xy_with_feats(tr, target, feats, med)
        Xva, yva, _, _ = _prepare_xy_with_feats(va, target, feats_used, med_used)
    else:
        Xtr, ytr, feats_used, med_used = _prepare_xy_auto(tr, target, med)
        Xva, yva, _, _ = _prepare_xy_auto(va, target, med_used)
    # Try XGB
    try:
        from xgboost import XGBRegressor  # type: ignore
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=42, n_jobs=0, eval_metric="rmse"
        )
        ok = False
        try:
            from xgboost.callback import EarlyStopping  # type: ignore
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                      callbacks=[EarlyStopping(rounds=50, save_best=True)])
            ok = True
        except Exception:
            pass
        if not ok:
            try:
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=50)
                ok = True
            except Exception:
                pass
        if not ok:
            model.fit(Xtr, ytr)
        return model, feats_used, med_used, "xgb_refit"
    except Exception:
        pass
    # Fallback: GBM
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR
    model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=3, random_state=42)
    Xfull = pd.concat([Xtr, Xva], axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yfull = pd.concat([ytr, yva], axis=0).fillna(0.0)
    model.fit(Xfull, yfull)
    return model, feats_used, med_used, "gbm_refit"

# --------------------- plotting sources ---------------------

def _from_predictions(tdir: Path, model_name: str, split: str, n: int, target: str) -> pd.DataFrame:
    """Load predictions.parquet, pick model col, merge with y_true, return last n rows."""
    preds = pd.read_parquet(tdir / "predictions.parquet")
    preds["ds"] = _norm_ds(preds["ds"])
    preds = preds.loc[preds["split"] == split].copy().sort_values("ds")
    col_map = {"ensemble": "y_ens", "xgb": "y_xgb", "arima": "y_arima"}
    if model_name not in col_map:
        raise ValueError("model must be ensemble|xgb|arima")
    preds = preds.rename(columns={col_map[model_name]: "yhat"})
    # Already has y_true
    out = preds[["ds", "y_true", "yhat"]].tail(n).reset_index(drop=True)
    return out

def _from_inference(tdir: Path, target: str, n: int) -> pd.DataFrame:
    """Recompute yhat on full dataset using saved artifacts (or refit), return last n rows with y_true."""
    ds_path = tdir / "dataset.parquet"
    tr_path = tdir / "train.parquet"
    va_path = tdir / "val.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)
    df = pd.read_parquet(ds_path)
    df["ds"] = _norm_ds(df["ds"]); df = df.sort_values("ds")
    train = _load_parquet(tr_path); val = _load_parquet(va_path)
    model_saved, feats_saved, med_saved = _try_load_artifacts(tdir)

    # Decide feature schema & medians
    if feats_saved:
        feats = [f for f in feats_saved if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    else:
        if train is not None:
            train["ds"] = _norm_ds(train["ds"])
            _, _, feats, _ = _prepare_xy_auto(train.sort_values("ds"), target, None)
        else:
            _, _, feats, _ = _prepare_xy_auto(df, target, None)

    if isinstance(med_saved, pd.Series):
        med = med_saved.reindex(feats).fillna(0.0)
    else:
        if train is not None:
            train["ds"] = _norm_ds(train["ds"])
            Xtr_tmp, _, _, _ = _prepare_xy_auto(train.sort_values("ds"), target, None)
            med = Xtr_tmp[feats].median(numeric_only=True).fillna(0.0)
        else:
            med = df[feats].median(numeric_only=True).fillna(0.0)

    # Model
    if model_saved is not None:
        model = model_saved
    else:
        tr_df = train if train is not None else df.iloc[:-10].copy()
        va_df = val if val is not None else df.iloc[-10:].copy()
        model, feats, med, _ = _fit_quick_model(tr_df, va_df, target, feats, med)

    # Score all rows, keep last n with actual
    X_all, _, _, _ = _prepare_xy_with_feats(df, target, feats, med)
    df["yhat"] = model.predict(X_all)
    if target in df.columns:
        df["y_true"] = pd.to_numeric(df[target], errors="coerce")
    else:
        df["y_true"] = np.nan
    out = df[["ds", "y_true", "yhat"]].tail(n).reset_index(drop=True)
    return out

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="target_return_1d")
    ap.add_argument("--source", choices=["predictions","inference"], default="inference",
                    help="Use trainer predictions.parquet or recompute with saved model")
    ap.add_argument("--model", choices=["ensemble","xgb","arima"], default="ensemble",
                    help="Which prediction column to use when --source=predictions")
    ap.add_argument("--split", choices=["val","test"], default="test",
                    help="Which split to plot when --source=predictions")
    ap.add_argument("--n", type=int, default=60, help="Last N days to plot")
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    tdir = Path(args.root) / args.ticker.upper()
    if args.source == "predictions":
        dfp = _from_predictions(tdir, args.model, args.split, args.n, args.target)
        subtitle = f"{args.ticker.upper()} — {args.model.upper()} ({args.split.upper()})"
    else:
        dfp = _from_inference(tdir, args.target, args.n)
        subtitle = f"{args.ticker.upper()} — Inference (full dataset)"

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfp["ds"], dfp["y_true"], label="Actual")
    ax.plot(dfp["ds"], dfp["yhat"], label="Predicted")
    ax.axhline(0.0, linestyle="--")
    ax.set_title(f"Actual vs Predicted — last {len(dfp)} days\n{subtitle}")
    ax.set_xlabel("Date")
    ax.set_ylabel(args.target)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    if args.out_png:
        out = Path(args.out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    else:
        # In notebooks, this will display inline
        plt.show()

if __name__ == "__main__":
    main()