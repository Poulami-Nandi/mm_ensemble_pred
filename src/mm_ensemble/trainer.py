#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer with AUTO ensemble weights (XGB + ARIMA) + price-level outputs

What this script does:
- Trains an XGB regressor (or GBM fallback) on prebuilt train/val/test parquet splits.
- Fits a simple ARIMA time-series baseline on the target series.
- Learns an ensemble blend weight w on the validation split (min RMSE), clamps to [0,1].
- Converts predicted daily returns -> price predictions using previous day's close.
- Saves artifacts under data/<TICKER>/:
    - predictions.parquet   (includes returns + price columns: p_true, p_xgb, p_arima, p_ens)
    - metrics.json          (RMSE/MAE/R2/DirectionAcc + price RMSE for val/test)
    - feature_importance.csv (if model exposes importances)
    - model_xgb.pkl or model_gbm.pkl
    - feature_names.json, feature_medians.json (for robust inference)
"""

import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------- utilities --------------------------

def _norm_ds(s: pd.Series) -> pd.Series:
    """Normalize any datetime-like series to naive daily timestamps."""
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _load_split(tdir: Path, name: str) -> pd.DataFrame:
    """Load a time split parquet (train/val/test), normalize dates, sort."""
    df = pd.read_parquet(tdir / f"{name}.parquet")
    df["ds"] = _norm_ds(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)

# Basic metrics computed in return space
def _metric_mae(y, yhat): return float(np.nanmean(np.abs(y - yhat)))
def _metric_rmse(y, yhat): return float(np.sqrt(np.nanmean((y - yhat) ** 2)))
def _metric_r2(y, yhat):
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
def _metric_diracc(y, yhat):
    """Directional accuracy: fraction of days where predicted sign matches true sign."""
    a = np.sign(y).astype(float); b = np.sign(yhat).astype(float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.mean((a[m] > 0) == (b[m] > 0))) if m.any() else float("nan")

def _collect_metrics(y_true: np.ndarray, y_pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute a standard metric dict for multiple prediction series."""
    out = {}
    for k, v in y_pred.items():
        out[k] = {
            "MAE": _metric_mae(y_true, v),
            "RMSE": _metric_rmse(y_true, v),
            "R2": _metric_r2(y_true, v),
            "DirectionAcc": _metric_diracc(y_true, v),
        }
    return out

def _all_numeric(df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    """Return all numeric feature columns, excluding drop_cols."""
    cand = [c for c in df.columns if c not in drop_cols]
    return [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]

def _prepare_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Select numeric features (excluding date/target/other targets),
    then median-impute NaNs, replace inf with NaN -> fill 0;
    return X, y, and the feature list.
    """
    drop_cols = ["ds", target]
    drop_cols += [c for c in df.columns if c.lower().startswith("target_") and c != target]
    feats = _all_numeric(df, drop_cols)
    X = df[feats].copy()
    y = df[target].astype(float).copy()
    # robust impute: medians -> replace inf -> fill 0
    med = X.median(numeric_only=True)
    X = X.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.fillna(0.0)
    return X, y, feats


# --------------------- optional dependencies ---------------------

def _get_xgb_model(random_state: int = 42, n_estimators: int = 1000, lr: float = 0.03, max_depth: int = 6):
    """
    Try to return an XGBRegressor with a cross-version-safe config.
    If xgboost is unavailable, fallback to sklearn's GradientBoostingRegressor.
    Returns tuple: (model, is_xgb: bool).
    """
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=0,
            # Important: set eval_metric on the model itself to be compatible
            # with both XGBoost 1.x and 2.x (2.x removed eval_metric from fit()).
            eval_metric="rmse",
        )
        return model, True
    except Exception:
        # Fallback: GradientBoosting (no native early stopping; we fit on train+val)
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=3, random_state=random_state
        )
        return model, False


def _fit_arima(series: pd.Series, seasonal: bool = False):
    """
    Fit an ARIMA-like baseline on the target series (returns).
    - Prefer pmdarima.auto_arima (stepwise), fallback to statsmodels ARIMA(1,0,0).
    - If too short or both unavailable, return None ("naive").
    Returns (model, engine_name).
    """
    y = series.astype(float).values
    y = y[~np.isnan(y)]
    if len(y) < 20:
        return None, "naive"

    # Try pmdarima first (convenient auto-order search)
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            y, seasonal=seasonal, stepwise=True, suppress_warnings=True,
            error_action="ignore", max_p=3, max_q=3, max_d=1, max_P=1, max_Q=1, max_D=1
        )
        return model, "pmdarima"
    except Exception:
        pass

    # Fallback: statsmodels ARIMA
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(1,0,0)).fit()
        return model, "statsmodels"
    except Exception:
        return None, "naive"


def _forecast_arima(model, n_steps: int) -> np.ndarray:
    """Forecast n_steps ahead from a fitted ARIMA-like model (robust to engine)."""
    if model is None or n_steps <= 0:
        return np.zeros(n_steps)
    try:
        # pmdarima engine
        import pmdarima as pm  # noqa: F401
        return model.predict(n_periods=n_steps)
    except Exception:
        pass
    try:
        # statsmodels engine
        return model.forecast(steps=n_steps)
    except Exception:
        return np.zeros(n_steps)


def _fit_xgb_cross_version(model, Xtr, ytr, Xva, yva):
    """
    Handle both xgboost 1.x and 2.x training flows.
    Try (in order):
      - callbacks.EarlyStopping (2.x / late 1.x)
      - early_stopping_rounds (older 1.x)
      - plain fit (no early stopping)
    """
    try:
        from xgboost.callback import EarlyStopping
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
            callbacks=[EarlyStopping(rounds=50, save_best=True)]
        )
        return
    except Exception:
        pass
    try:
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
            early_stopping_rounds=50
        )
        return
    except Exception:
        pass
    model.fit(Xtr, ytr)


# --------------------------- price helpers ---------------------------

def _pick_price_column(tdir: Path, pref: str = "auto") -> Optional[str]:
    """
    Decide which price column to use when mapping returns -> prices.
    Preference: 'adj_close' (if present) else 'close', unless overridden.
    """
    px_path = tdir / "prices.parquet"
    if not px_path.exists():
        return None
    cols = set(pd.read_parquet(px_path, columns=None).columns.str.lower())
    if pref == "adj_close" and "adj_close" in cols:
        return "adj_close"
    if pref == "close" and "close" in cols:
        return "close"
    # auto mode: prefer adj_close if available
    if "adj_close" in cols:
        return "adj_close"
    if "close" in cols:
        return "close"
    return None

def _map_returns_to_prices(tdir: Path, ds_series: pd.Series, y_pred: np.ndarray, price_col: str) -> Tuple[pd.Series, pd.Series]:
    """
    Convert predicted returns for the dates in ds_series to price predictions using
    previous day's close from prices.parquet. Returns (actual_price_series, predicted_price_series).
    """
    px = pd.read_parquet(tdir / "prices.parquet")[["ds", price_col]].copy()
    px["ds"] = _norm_ds(px["ds"])
    px = px.sort_values("ds")

    # Use previous day's close as the base for 1-day forward return compounding
    px["prev_close"] = px[price_col].shift(1)

    # Align to requested dates
    df_dates = pd.DataFrame({"ds": _norm_ds(ds_series)})
    df_aligned = df_dates.merge(px.rename(columns={price_col: "actual_price"}), on="ds", how="left")

    # Edge-case handling: if prev_close is NaN at the start, ffill from earliest available
    df_aligned["prev_close"] = df_aligned["prev_close"].ffill()

    pred_price = df_aligned["prev_close"].values * (1.0 + np.asarray(y_pred, float))
    return df_aligned["actual_price"].astype(float), pd.Series(pred_price, index=df_aligned.index)


# --------------------------- trainer core ---------------------------

@dataclass
class TrainCfg:
    """Configuration used during a single run across tickers."""
    root: Path
    tickers: List[str]
    target: str
    w_xgb: float
    w_arima: float
    auto_weights: bool
    random_state: int
    n_estimators: int
    lr: float
    max_depth: int
    price_pref: str

def _opt_weight_closed_form(y_true: np.ndarray, y_xgb: np.ndarray, y_ar: np.ndarray) -> float:
    """
    Solve for w* in [0,1] minimizing || y - (w y_xgb + (1-w) y_ar) ||^2 over points
    where all values are finite. Closed-form solution with clipping to [0,1].
    """
    y = y_true.astype(float)
    a = y_xgb.astype(float)
    b = y_ar.astype(float)
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 1.0  # fallback to pure XGB if validation has no usable points
    y = y[m]; a = a[m]; b = b[m]
    d = (a - b)
    den = float(np.dot(d, d))
    if den <= 0:
        return 1.0
    num = float(np.dot((y - b), d))
    w = num / den
    return float(np.clip(w, 0.0, 1.0))

def _train_one_ticker(cfg: TrainCfg, ticker: str) -> Tuple[Path, Path]:
    """Fit models, compute ensemble, write predictions/metrics for a single ticker."""
    tdir = cfg.root / ticker
    train = _load_split(tdir, "train")
    val   = _load_split(tdir, "val")
    test  = _load_split(tdir, "test")

    # ----------------- XGB (or GBM fallback) -----------------
    Xtr, ytr, feats = _prepare_xy(train, cfg.target)
    Xva, yva, _     = _prepare_xy(val, cfg.target)
    Xte, yte, _     = _prepare_xy(test, cfg.target)

    model, is_xgb = _get_xgb_model(cfg.random_state, cfg.n_estimators, cfg.lr, cfg.max_depth)

    if is_xgb:
        _fit_xgb_cross_version(model, Xtr, ytr, Xva, yva)
    else:
        # GBM fallback: no early stopping, fit on train+val merged
        from sklearn.ensemble import GradientBoostingRegressor
        model: GradientBoostingRegressor
        model.fit(pd.concat([Xtr, Xva], axis=0), pd.concat([ytr, yva], axis=0))

    # Supervised predictions in return space
    yhat_tr_xgb = model.predict(Xtr)
    yhat_va_xgb = model.predict(Xva)
    yhat_te_xgb = model.predict(Xte)

    # Feature importance (if model exposes it)
    fi_path = tdir / "feature_importance.csv"
    try:
        if hasattr(model, "feature_importances_"):
            pd.DataFrame({"feature": feats, "importance": model.feature_importances_}) \
              .sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    except Exception:
        pass

    # ----------------- ARIMA baseline -----------------
    arima_model, arima_kind = _fit_arima(train[cfg.target])
    n_va = len(val); n_te = len(test)
    yhat_va_ar = _forecast_arima(arima_model, n_va)
    yhat_te_ar = _forecast_arima(arima_model, n_te)

    # ----------------- Weights (AUTO or fixed) -----------------
    if cfg.auto_weights:
        # Learn w on validation split using closed-form least squares
        w1 = _opt_weight_closed_form(yva.values, yhat_va_xgb, yhat_va_ar)
        w2 = 1.0 - w1
    else:
        # Use user-specified fixed weights (normalized to sum to 1)
        w1, w2 = cfg.w_xgb, cfg.w_arima
        s = (w1 + w2) if (w1 + w2) != 0 else 1.0
        w1, w2 = w1 / s, w2 / s

    # ----------------- Ensemble in return space -----------------
    yhat_va_ens = w1 * yhat_va_xgb + w2 * yhat_va_ar
    yhat_te_ens = w1 * yhat_te_xgb + w2 * yhat_te_ar

    # ----------------- Metrics (return space) -----------------
    metrics = {
        "model_info": {
            "xgb_fallback": (not is_xgb),
            "arima_engine": arima_kind,
            "weights": {"auto": bool(cfg.auto_weights), "w_xgb": float(w1), "w_arima": float(w2)},
            "target": cfg.target,
            "price_pref": cfg.price_pref,
        },
        "val": _collect_metrics(yva.values, {
            "xgb": yhat_va_xgb, "arima": yhat_va_ar, "ensemble": yhat_va_ens
        }),
        "test": _collect_metrics(yte.values, {
            "xgb": yhat_te_xgb, "arima": yhat_te_ar, "ensemble": yhat_te_ens
        }),
    }

    # ----------------- Save predictions (return space) -----------------
    pred_va = pd.DataFrame({
        "ds": val["ds"], "y_true": yva.values,
        "y_xgb": yhat_va_xgb, "y_arima": yhat_va_ar, "y_ens": yhat_va_ens,
        "split": "val"
    })
    pred_te = pd.DataFrame({
        "ds": test["ds"], "y_true": yte.values,
        "y_xgb": yhat_te_xgb, "y_arima": yhat_te_ar, "y_ens": yhat_te_ens,
        "split": "test"
    })

    # ----------------- Add price-space columns -----------------
    price_col = _pick_price_column(tdir, cfg.price_pref)
    if price_col is not None:
        # Validation price mapping (true + each model + ensemble)
        p_va_true, p_va_ens = _map_returns_to_prices(tdir, pred_va["ds"], pred_va["y_ens"].values, price_col)
        _,        p_va_xgb = _map_returns_to_prices(tdir, pred_va["ds"], pred_va["y_xgb"].values, price_col)
        _,        p_va_ari = _map_returns_to_prices(tdir, pred_va["ds"], pred_va["y_arima"].values, price_col)

        # Test price mapping (true + each model + ensemble)
        p_te_true, p_te_ens = _map_returns_to_prices(tdir, pred_te["ds"], pred_te["y_ens"].values, price_col)
        _,        p_te_xgb = _map_returns_to_prices(tdir, pred_te["ds"], pred_te["y_xgb"].values, price_col)
        _,        p_te_ari = _map_returns_to_prices(tdir, pred_te["ds"], pred_te["y_arima"].values, price_col)

        # Attach price columns
        pred_va["p_true"]  = p_va_true.values
        pred_va["p_xgb"]   = p_va_xgb.values
        pred_va["p_arima"] = p_va_ari.values
        pred_va["p_ens"]   = p_va_ens.values

        pred_te["p_true"]  = p_te_true.values
        pred_te["p_xgb"]   = p_te_xgb.values
        pred_te["p_arima"] = p_te_ari.values
        pred_te["p_ens"]   = p_te_ens.values

        # Price RMSE (on val/test)
        def _safe_rmse(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            m = np.isfinite(a) & np.isfinite(b)
            return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else float("nan")

        metrics["val_price"]  = {"RMSE": _safe_rmse(pred_va["p_true"], pred_va["p_ens"])}
        metrics["test_price"] = {"RMSE": _safe_rmse(pred_te["p_true"], pred_te["p_ens"])}

    # ----------------- Persist -----------------
    preds = pd.concat([pred_va, pred_te], axis=0).sort_values(["split","ds"])
    preds_path = tdir / "predictions.parquet"
    metrics_path = tdir / "metrics.json"
    preds.to_parquet(preds_path, index=False)
    (tdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ----------------- Save model + artifacts -----------------
    try:
        from joblib import dump
        artifact_path = tdir / ("model_xgb.pkl" if is_xgb else "model_gbm.pkl")
        dump(model, artifact_path)
        # Also save feature names and training medians for robust inference
        feat_path = tdir / "feature_names.json"
        med_ser = Xtr.median(numeric_only=True).fillna(0.0)
        med_path  = tdir / "feature_medians.json"
        feat_path.write_text(json.dumps(list(Xtr.columns), indent=2), encoding="utf-8")
        med_path.write_text(json.dumps({k: float(v) for k, v in med_ser.items()}, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[{ticker}] Warning: failed to save model or artifacts: {e}")

    # ----------------- Console summary -----------------
    print(f"[{ticker}] Weights (auto={cfg.auto_weights}) -> XGB: {w1:.3f}, ARIMA: {w2:.3f}")
    print(f"[{ticker}] Val RMSE (returns: xgb/arima/ens): "
          f"{metrics['val']['xgb']['RMSE']:.6f} / {metrics['val']['arima']['RMSE']:.6f} / {metrics['val']['ensemble']['RMSE']:.6f}")
    print(f"[{ticker}] Test RMSE (returns: xgb/arima/ens): "
          f"{metrics['test']['xgb']['RMSE']:.6f} / {metrics['test']['arima']['RMSE']:.6f} / {metrics['test']['ensemble']['RMSE']:.6f}")
    if "val_price" in metrics and "test_price" in metrics:
        print(f"[{ticker}] Val RMSE (price):  ${metrics['val_price']['RMSE']:.4f}")
        print(f"[{ticker}] Test RMSE (price): ${metrics['test_price']['RMSE']:.4f}")

    return preds_path, metrics_path


# ------------------------------ CLI ------------------------------

@dataclass
class TrainArgs:
    """CLI-parseable arguments packed for _train_one_ticker."""
    root: Path
    tickers: List[str]
    target: str
    w_xgb: float
    w_arima: float
    auto_weights: bool
    random_state: int
    n_estimators: int
    lr: float
    max_depth: int
    price_pref: str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root data folder")
    ap.add_argument("--tickers", nargs="+", default=["SBUX","PFE"])
    ap.add_argument("--target", default="target_return_1d",
                    help="Target column to fit (default: target_return_1d)")
    # weights: keep flags for manual override, but default to auto
    ap.add_argument("--w-xgb", type=float, default=0.7, help="(Only if --no-auto-weights) Weight for XGB")
    ap.add_argument("--w-arima", type=float, default=0.3, help="(Only if --no-auto-weights) Weight for ARIMA")
    ap.add_argument("--auto-weights", dest="auto_weights", action="store_true", help="Learn ensemble weights on validation")
    ap.add_argument("--no-auto-weights", dest="auto_weights", action="store_false", help="Disable auto weighting (use fixed)")
    ap.set_defaults(auto_weights=True)  # default to AUTO weighting

    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--price-col-pref", choices=["auto","adj_close","close"], default="auto",
                    help="Which price column to use when mapping returns->prices (default: auto=prefer adj_close)")
    args = ap.parse_args()

    cfg = TrainArgs(
        root=Path(args.root),
        tickers=[t.upper() for t in args.tickers],
        target=args.target,
        w_xgb=args.w_xgb,
        w_arima=args.w_arima,
        auto_weights=args.auto_weights,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        lr=args.lr,
        max_depth=args.max_depth,
        price_pref=args.price_col_pref,
    )

    written = []
    for t in cfg.tickers:
        preds_path, metrics_path = _train_one_ticker(cfg, t)
        written += [str(preds_path), str(metrics_path)]
    print("Wrote:")
    for w in written:
        print(" ", w)

if __name__ == "__main__":
    main()
