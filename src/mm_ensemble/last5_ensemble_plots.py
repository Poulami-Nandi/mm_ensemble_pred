#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# Module: mm_ensemble/last5_ensemble_plots.py
# Overview:
#   Trains on last ~1y, predicts last 5 trading days, produces comparison plots (Actual vs OHLCV vs All inputs).
# Notes:
#   - This file has been annotated with verbose comments for clarity.
#   - Logic is unchanged; only comments were added.
##############################################################################


# Imports: stdlib, scientific stack, and optional ML/TS libs
import os, json, math, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= config =========

# Configuration / constants / paths
ROOT = DATA_DIR              # where data/<TICKER>/dataset.parquet lives
OUT_BASE = OUTPUTS_DIR / "last5" # outputs go here (NOT under data/)
TICKERS = ["SBUX", "PFE"]
TARGET = "target_return_1d"
HOLDOUT_DAYS = 5
RANDOM_STATE = 42

# ========= helpers =========

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _load_dataset(root: Path, ticker: str) -> pd.DataFrame:
    p = root / ticker / "dataset.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset for {ticker}: {p}")
    df = pd.read_parquet(p)
    df["ds"] = _norm_ds(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df

def _all_numeric(df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    cand = [c for c in df.columns if c not in drop_cols]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]

def _select_all_features(df: pd.DataFrame, target: str) -> List[str]:
    drop_cols = {"ds", target}
    drop_cols |= {c for c in df.columns if c.lower().startswith("target_") and c != target}
    return _all_numeric(df, list(drop_cols))

def _select_ohlcv_features(df: pd.DataFrame, target: str) -> List[str]:
    keep_prefixes = ("open","high","low","close","adj_close","volume","ret_","logret_","ma_","vol_")
    keep_exact = {"ma_5_over_20", "ma_20_over_50"}
    drop_cols = {"ds", target}
    drop_cols |= {c for c in df.columns if c.lower().startswith("target_") and c != target}
    feats = []
    for c in df.columns:
        if c in drop_cols: continue
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if c in keep_exact or c.startswith(keep_prefixes):
            feats.append(c)
    return list(dict.fromkeys(feats))

def _time_split_last_year(df: pd.DataFrame, n_holdout: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    n = len(train)
    n_val = max(min_val, int(0.1 * n))
    n_val = min(n_val, max(1, n // 4))
    if n <= n_val + 5: n_val = max(1, n // 5)
    tr = train.iloc[: n - n_val].copy()
    va = train.iloc[n - n_val :].copy()
    return tr, va

def _impute_matrix(X: pd.DataFrame, med: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if med is None:
        med = X.median(numeric_only=True)
    X = X.copy()
    X = X.fillna(med)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, med.fillna(0.0)

def _prepare_xy(df: pd.DataFrame, feats: List[str], target: str, med: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[feats].copy()
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    X, med_used = _impute_matrix(X, med)
    return X, y, med_used

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any(): return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2)))

# ========= models =========


# --- Function `_fit_xgb(Xtr, ytr, Xva, yva)` ---
# Purpose: Describe inputs/outputs and key steps.
# (Auto-generated comment: refine as needed.)
def _fit_xgb(Xtr, ytr, Xva, yva):
    """XGBoost version-agnostic fit, fallback to scikit GBM."""
    try:
        from xgboost import XGBRegressor  # type: ignore
        model = XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=0, eval_metric="rmse"
        )
        ok = False
        try:
            from xgboost.callback import EarlyStopping  # type: ignore
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                      callbacks=[EarlyStopping(rounds=80, save_best=True)])
            ok = True
        except Exception:
            pass
        if not ok:
            try:
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=80)
                ok = True
            except Exception:
                pass
        if not ok:
            model.fit(Xtr, ytr)
        return model, "xgb"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        model = GradientBoostingRegressor(n_estimators=800, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
        model.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        return model, "gbm"


# --- Function `_fit_arima(series: pd.Series)` ---
# Purpose: Describe inputs/outputs and key steps.
# (Auto-generated comment: refine as needed.)
def _fit_arima(series: pd.Series):
    """Try pmdarima, fallback to statsmodels ARIMA(1,0,0), else zeros."""
    y = pd.to_numeric(series, errors="coerce").values
    y = y[np.isfinite(y)]
    if len(y) < 20: return None, "naive"
    try:
        import pmdarima as pm
        model = pm.auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True,
                              error_action="ignore", max_p=3, max_q=3, max_d=1)
        return model, "pmdarima"
    except Exception:
        pass
    try:
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(1,0,0)).fit()
        return model, "statsmodels"
    except Exception:
        return None, "naive"

def _forecast_arima(model, n_steps: int) -> np.ndarray:
    if model is None or n_steps <= 0: return np.zeros(n_steps)
    try:
        import pmdarima as pm  # noqa: F401
        return model.predict(n_periods=n_steps)
    except Exception:
        pass
    try:
        return model.forecast(steps=n_steps)
    except Exception:
        return np.zeros(n_steps)

def _opt_weight_closed_form(y_true: np.ndarray, y_xgb: np.ndarray, y_ar: np.ndarray) -> float:
    y = y_true.astype(float); a = y_xgb.astype(float); b = y_ar.astype(float)
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if not m.any(): return 1.0
    y = y[m]; a = a[m]; b = b[m]
    d = a - b
    den = float(np.dot(d, d))
    if den <= 0: return 1.0
    num = float(np.dot((y - b), d))
    return float(np.clip(num / den, 0.0, 1.0))

# ========= core routine per feature set =========

def train_and_predict_last5(df: pd.DataFrame, feats: List[str], target: str) -> Dict[str, object]:
    train_all, test5 = _time_split_last_year(df, n_holdout=HOLDOUT_DAYS)
    tr, va = _build_val_from_train(train_all)

    Xtr, ytr, med = _prepare_xy(tr, feats, target, None)
    Xva, yva, _   = _prepare_xy(va, feats, target, med)
    Xte, yte, _   = _prepare_xy(test5, feats, target, med)

    # supervised
    sup_model, sup_kind = _fit_xgb(Xtr, ytr, Xva, yva)
    yhat_va_sup = sup_model.predict(Xva)
    yhat_te_sup = sup_model.predict(Xte)

    # arima
    arima_model, arima_kind = _fit_arima(train_all[target])
    yhat_va_ar = _forecast_arima(arima_model, len(va))
    yhat_te_ar = _forecast_arima(arima_model, len(test5))

    # auto weights from val
    w = _opt_weight_closed_form(yva.values, yhat_va_sup, yhat_va_ar)
    yhat_te_ens = w * yhat_te_sup + (1.0 - w) * yhat_te_ar

    rmse_all = _rmse(yte.values, yhat_te_ens)
    return {
        "train_len": len(train_all),
        "val_len": len(va),
        "test_len": len(test5),
        "dates_last5": test5["ds"].tolist(),
        "y_true_last5": yte.values,
        "y_sup_last5": yhat_te_sup,
        "y_ar_last5": yhat_te_ar,
        "y_ens_last5": yhat_te_ens,
        "rmse_last5": rmse_all,
        "w_xgb": float(w),
        "w_arima": float(1.0 - w),
        "sup_kind": sup_kind,
        "arima_kind": arima_kind,
        "feats_count": len(feats),
    }

# ========= plotting =========


# --- Function `plot_actual_vs_pred(dates, y_true, y_pred, title, rmse, out_png=None, weights: Optional[Tuple[float,float]] = None)` ---
# Purpose: Describe inputs/outputs and key steps.
# (Auto-generated comment: refine as needed.)
def plot_actual_vs_pred(dates, y_true, y_pred, title, rmse, out_png=None, weights: Optional[Tuple[float,float]] = None):
    # If weights provided, append into title
    if weights is not None:
        w_x, w_a = weights
        title = f"{title} â€” w_xgb={w_x:.2f}, w_arima={w_a:.2f}"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(dates, y_true, label="Actual price")
    ax.plot(dates, y_pred, label=f"Predicted price (RMSE=${rmse:.2f})")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price: Actual vs Prediction")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    plt.show()
    plt.close(fig)

def plot_compare(dates, y_true, y_ohlcv, rmse_ohlcv, y_all, rmse_all, title,
                 out_png=None, weights_ohlcv: Optional[Tuple[float,float]] = None,
                 weights_all: Optional[Tuple[float,float]] = None):
    # Build legend labels that carry weights for each line
    if weights_ohlcv is not None:
        wxo, wao = weights_ohlcv
        lbl_ohl = f"Pred (OHLCV, RMSE=${rmse_ohlcv:.2f}, w_xgb={wxo:.2f}, w_arima={wao:.2f})"
    else:
        lbl_ohl = f"Pred (OHLCV, RMSE=${rmse_ohlcv:.2f})"
    if weights_all is not None:
        wxa, waa = weights_all
        lbl_all = f"Pred (All inputs, RMSE=${rmse_all:.2f}, w_xgb={wxa:.2f}, w_arima={waa:.2f})"
    else:
        lbl_all = f"Pred (All inputs, RMSE=${rmse_all:.2f})"

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(dates, y_true,  label="Actual price")
    ax.plot(dates, y_ohlcv, label=lbl_ohl)
    ax.plot(dates, y_all,   label=lbl_all)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price: Actual vs Prediction")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    plt.show()
    plt.close(fig)

# ========= price conversion helpers =========

def _returns_to_prices(ticker: str, dates, ret_pred, target_col_pred_name: str) -> pd.DataFrame:
    """
    Map next-day return predictions to next-day prices using previous day's close.
    Returns a DataFrame with columns: ds, actual_price, <target_col_pred_name>.
    """
    prices_path = ROOT / ticker / "prices.parquet"
    px = pd.read_parquet(prices_path)[["ds","close"]].copy()
    px["ds"] = _norm_ds(px["ds"])
    px = px.sort_values("ds")

    dfp = pd.DataFrame({"ds": pd.to_datetime(dates)})
    # actual price_t = close on that date
    dfp = dfp.merge(px, on="ds", how="left")
    # previous day close
    prev = px.rename(columns={"ds":"ds_prev","close":"close_prev"})
    dfp = dfp.merge(prev, left_on="ds", right_on="ds_prev", how="left")
    dfp["close_prev"] = dfp["close_prev"].ffill()

    dfp[target_col_pred_name] = dfp["close_prev"].values * (1.0 + np.asarray(ret_pred, float))
    dfp["actual_price"] = dfp["close"].astype(float)
    return dfp[["ds","actual_price", target_col_pred_name]]

def _rmse_price(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")

# ========= main =========


# --- Function `main()` ---
# Purpose: Describe inputs/outputs and key steps.
# (Auto-generated comment: refine as needed.)
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary = {"tickers": {}}

    for t in TICKERS:
        tdir = OUT_BASE / t
        tdir.mkdir(parents=True, exist_ok=True)

        df = _load_dataset(ROOT, t)
        feats_all   = _select_all_features(df, TARGET)
        feats_ohlcv = _select_ohlcv_features(df, TARGET)

        # --- All inputs ensemble (returns) ---
        res_all = train_and_predict_last5(df, feats_all, TARGET)
        # save return CSV/JSON (keep for completeness)
        pd.DataFrame({
            "ds": res_all["dates_last5"],
            "y_true": res_all["y_true_last5"],
            "y_pred_all": res_all["y_ens_last5"],
        }).to_csv(tdir / "pred_last5_all_inputs.csv", index=False)
        Path(tdir / "metrics_all_inputs.json").write_text(json.dumps({
            "rmse_last5_return": res_all["rmse_last5"],
            "weights": {"w_xgb": res_all["w_xgb"], "w_arima": res_all["w_arima"]},
            "model": {"supervised": res_all["sup_kind"], "arima": res_all["arima_kind"]},
            "feats_count": res_all["feats_count"],
            "train_len": res_all["train_len"], "val_len": res_all["val_len"], "test_len": res_all["test_len"]
        }, indent=2), encoding="utf-8")

        # --- OHLCV-only ensemble (returns) ---
        res_ohl = train_and_predict_last5(df, feats_ohlcv, TARGET)
        pd.DataFrame({
            "ds": res_ohl["dates_last5"],
            "y_true": res_ohl["y_true_last5"],
            "y_pred_ohlcv": res_ohl["y_ens_last5"],
        }).to_csv(tdir / "pred_last5_ohlcv_only.csv", index=False)
        Path(tdir / "metrics_ohlcv_only.json").write_text(json.dumps({
            "rmse_last5_return": res_ohl["rmse_last5"],
            "weights": {"w_xgb": res_ohl["w_xgb"], "w_arima": res_ohl["w_arima"]},
            "model": {"supervised": res_ohl["sup_kind"], "arima": res_ohl["arima_kind"]},
            "feats_count": res_ohl["feats_count"],
            "train_len": res_ohl["train_len"], "val_len": res_ohl["val_len"], "test_len": res_ohl["test_len"]
        }, indent=2), encoding="utf-8")

        # ======== PRICE-VIEW: convert returns -> prices and plot ========
        # All inputs -> price
        df_all_price = _returns_to_prices(t, res_all["dates_last5"], res_all["y_ens_last5"], "pred_price_all")
        rmse_all_price = _rmse_price(df_all_price["actual_price"].values, df_all_price["pred_price_all"].values)
        df_all_price.to_csv(tdir / "pred_last5_all_inputs_PRICE.csv", index=False)
        plot_actual_vs_pred(
            df_all_price["ds"], df_all_price["actual_price"], df_all_price["pred_price_all"],
            title=f"{t} â€” All inputs â€” Last 5 days (price)", rmse=rmse_all_price,
            out_png=tdir / "actual_vs_pred_all_inputs_PRICE.png",
            weights=(res_all["w_xgb"], res_all["w_arima"])
        )

        # OHLCV-only -> price
        df_ohl_price = _returns_to_prices(t, res_ohl["dates_last5"], res_ohl["y_ens_last5"], "pred_price_ohlcv")
        rmse_ohl_price = _rmse_price(df_ohl_price["actual_price"].values, df_ohl_price["pred_price_ohlcv"].values)
        df_ohl_price.to_csv(tdir / "pred_last5_ohlcv_only_PRICE.csv", index=False)
        plot_actual_vs_pred(
            df_ohl_price["ds"], df_ohl_price["actual_price"], df_ohl_price["pred_price_ohlcv"],
            title=f"{t} â€” OHLCV only â€” Last 5 days (price)", rmse=rmse_ohl_price,
            out_png=tdir / "actual_vs_pred_ohlcv_only_PRICE.png",
            weights=(res_ohl["w_xgb"], res_ohl["w_arima"])
        )

        # Comparison (price) â€” include weights in legend lines
        plot_compare(
            df_all_price["ds"],
            df_all_price["actual_price"],
            df_ohl_price["pred_price_ohlcv"], rmse_ohl_price,
            df_all_price["pred_price_all"],  rmse_all_price,
            title=f"{t} â€” Compare (Actual vs OHLCV vs All) â€” Price",
            out_png=tdir / "compare_all_vs_ohlcv_PRICE.png",
            weights_ohlcv=(res_ohl["w_xgb"], res_ohl["w_arima"]),
            weights_all=(res_all["w_xgb"], res_all["w_arima"])
        )

        # Update summary with price plots as the primaries
        summary["tickers"][t] = {
            "all_inputs": {
                "rmse_last5_price": rmse_all_price,
                "weights": {"w_xgb": res_all["w_xgb"], "w_arima": res_all["w_arima"]},
                "plot_price": str(tdir / "actual_vs_pred_all_inputs_PRICE.png"),
                "csv_price":  str(tdir / "pred_last5_all_inputs_PRICE.csv"),
                "csv_return": str(tdir / "pred_last5_all_inputs.csv"),
            },
            "ohlcv_only": {
                "rmse_last5_price": rmse_ohl_price,
                "weights": {"w_xgb": res_ohl["w_xgb"], "w_arima": res_ohl["w_arima"]},
                "plot_price": str(tdir / "actual_vs_pred_ohlcv_only_PRICE.png"),
                "csv_price":  str(tdir / "pred_last5_ohlcv_only_PRICE.csv"),
                "csv_return": str(tdir / "pred_last5_ohlcv_only.csv"),
            },
            "compare_plot_price": str(tdir / "compare_all_vs_ohlcv_PRICE.png"),
        }

    Path(OUT_BASE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {OUT_BASE / 'summary.json'}")
    for t in TICKERS:
        print(f"- {t} price plots saved under {OUT_BASE / t}")


# Entrypoint: parse CLI args and run main routine.
if __name__ == "__main__":
    main()
