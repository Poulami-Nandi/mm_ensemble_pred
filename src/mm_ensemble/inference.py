#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------- helpers ---------------------

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _all_numeric(df: pd.DataFrame, drop_cols):
    cols = [c for c in df.columns if c not in drop_cols]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def _load_parquet(p: Path):
    if not p.exists():
        return None
    return pd.read_parquet(p)

def _impute_matrix(X: pd.DataFrame, med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Median impute -> replace inf -> final 0.0 fill to remove any NaNs."""
    if med is None:
        med = X.median(numeric_only=True)
    X = X.copy()
    X = X.fillna(med)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, med

def _prepare_xy_auto(df: pd.DataFrame, target: str, medians: pd.Series | None = None):
    """
    Auto-select features (all numeric except ds + target + other target_*).
    Return (X, y, feats, medians_used).
    """
    drop_cols = ["ds", target] + [c for c in df.columns if c.lower().startswith("target_") and c != target]
    feats = _all_numeric(df, drop_cols)
    X = df[feats].copy()
    y = df[target].astype(float) if target in df.columns else None
    X, med_used = _impute_matrix(X, medians)
    if y is not None:
        y = y.fillna(0.0)
    return X, y, feats, med_used

def _prepare_xy_with_feats(df: pd.DataFrame, target: str, feats: list[str], med: pd.Series | None):
    """
    Build X with a fixed feature list:
      - add any missing columns with 0.0
      - order columns to 'feats'
      - impute using provided medians aligned to feats
    """
    X = df.copy()
    # add missing expected features with 0
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
    """
    Returns (model, feats_saved, med_saved)
    - model is loaded if model_xgb.pkl or model_gbm.pkl exists
    - feats_saved: list or None
    - med_saved: pandas Series or None
    """
    model = None
    feats_saved = None
    med_saved = None

    # Load model if present
    try:
        from joblib import load  # type: ignore
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR
        if (tdir / "model_xgb.pkl").exists():
            model = load(tdir / "model_xgb.pkl")
        elif (tdir / "model_gbm.pkl").exists():
            model = load(tdir / "model_gbm.pkl")
    except Exception:
        model = None

    # Load feature names if present
    feats_path = tdir / "feature_names.json"
    if feats_path.exists():
        try:
            feats_saved = json.loads(feats_path.read_text(encoding="utf-8"))
        except Exception:
            feats_saved = None

    # Load feature medians if present
    med_path = tdir / "feature_medians.json"
    if med_path.exists():
        try:
            med_saved = pd.Series(json.loads(med_path.read_text(encoding="utf-8")))
        except Exception:
            med_saved = None

    return model, feats_saved, med_saved

# --------------------- (re)fit fallback ---------------------

def _fit_quick_model(train_df: pd.DataFrame, val_df: pd.DataFrame, target: str,
                     feats: list[str] | None, med: pd.Series | None):
    """
    Quick training used only when no saved model is found.
    - If feats provided, use strict feature alignment; else auto-select.
    - XGBoost if available (API-version agnostic); else scikit-learn GBM.
    Returns (model, train_feats, train_medians, train_mode_str).
    """
    tr = train_df.sort_values("ds").copy()
    va = val_df.sort_values("ds").copy()

    if feats is not None:
        Xtr, ytr, feats_used, med_used = _prepare_xy_with_feats(tr, target, feats, med)
        Xva, yva, _, _ = _prepare_xy_with_feats(va, target, feats_used, med_used)
    else:
        Xtr, ytr, feats_used, med_used = _prepare_xy_auto(tr, target, med)
        Xva, yva, _, _ = _prepare_xy_auto(va, target, med_used)

    # Try XGBoost first
    try:
        from xgboost import XGBRegressor  # type: ignore
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", random_state=42, n_jobs=0, eval_metric="rmse"
        )
        # Version-agnostic fit: try callbacks -> early_stopping_rounds -> plain fit
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

    # Fallback: scikit-learn GBM (ensure no NaNs)
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=3, random_state=42)
    Xfull = pd.concat([Xtr, Xva], axis=0)
    yfull = pd.concat([ytr, yva], axis=0)
    Xfull = Xfull.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yfull = yfull.fillna(0.0)
    model.fit(Xfull, yfull)
    return model, feats_used, med_used, "gbm_refit"

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="target_return_1d")
    ap.add_argument("--n-recent", type=int, default=5, help="Emit predictions for the last N rows")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    tdir = Path(args.root) / args.ticker.upper()
    ds_path = tdir / "dataset.parquet"
    tr_path = tdir / "train.parquet"
    va_path = tdir / "val.parquet"
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)

    # Load dataset
    df = pd.read_parquet(ds_path)
    df["ds"] = _norm_ds(df["ds"])
    df = df.sort_values("ds")

    # Try to load artifacts (model + feature names + medians)
    model_saved, feats_saved, med_saved = _try_load_artifacts(tdir)

    # Decide features + medians
    train = _load_parquet(tr_path)
    val   = _load_parquet(va_path)

    # 1) Feature list: prefer saved list, else derive from train (fallback: whole df)
    if feats_saved:
        # keep only numeric columns that still exist
        feats = [f for f in feats_saved if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    else:
        if train is not None:
            train["ds"] = _norm_ds(train["ds"])
            _, _, feats, _ = _prepare_xy_auto(train.sort_values("ds"), args.target, None)
        else:
            _, _, feats, _ = _prepare_xy_auto(df, args.target, None)

    # 2) Medians: prefer saved medians aligned to feats; else compute from train/df
    if isinstance(med_saved, pd.Series):
        med = med_saved.reindex(feats).fillna(0.0)
    else:
        if train is not None:
            train["ds"] = _norm_ds(train["ds"])
            Xtr_tmp, _, _, _ = _prepare_xy_auto(train.sort_values("ds"), args.target, None)
            med = Xtr_tmp[feats].median(numeric_only=True).fillna(0.0)
        else:
            med = df[feats].median(numeric_only=True).fillna(0.0)

    # Build or load model
    if model_saved is not None:
        model = model_saved
        mode  = "loaded"
    else:
        # If splits missing, use last 10 rows as "val" fallback
        tr_df = train if train is not None else df.iloc[:-10].copy()
        va_df = val if val is not None else df.iloc[-10:].copy()
        model, feats, med, mode = _fit_quick_model(tr_df, va_df, args.target, feats, med)

    # Inference on last N rows, aligned to feature list and imputed via training medians
    # Align features: add missing, order, impute
    X_all, _y_unused, _feats_used, _med_used = _prepare_xy_with_feats(df, args.target, feats, med)
    yhat_all = model.predict(X_all)
    last_pred = float(yhat_all[-1])

    out = {
        "ticker": args.ticker.upper(),
        "target": args.target,
        "predict_mode": mode,
        "last_date": str(df["ds"].iloc[-1].date()),
        "next_day_return_prediction": last_pred,
        "n_rows_scored": int(len(df)),
        "n_features": int(len(feats))
    }

    # Optionally persist outputs
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    n = max(1, args.n_recent)
    tail = df[["ds"]].copy()
    tail["yhat"] = yhat_all
    tail = tail.tail(n)
    if args.out_csv:
        tail.to_csv(args.out_csv, index=False)

    # Print a compact JSON for quick view
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()