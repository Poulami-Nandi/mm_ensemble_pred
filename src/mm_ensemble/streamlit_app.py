#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, json
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Bootstrap: ensure 'src' is importable both locally and on Cloud
here = Path(__file__).resolve()
repo_root = here.parents[2]  # .../repo_root/src/mm_ensemble/streamlit_app.py
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer shared paths; fallback to repo root /data and /outputs
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = repo_root / "data"
    OUTPUTS_DIR = repo_root / "outputs"

st.set_page_config(page_title="mm_ensemble — Last-5 price demo", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo")

TICKERS = ["SBUX", "PFE"]

# ---------------- helpers ----------------

def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _is_returns_like(x: pd.Series) -> bool:
    """Heuristic: returns are small; prices are usually >> 1 for equities."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return False
    frac_small = np.mean(np.abs(x) <= 0.2)
    return bool(frac_small > 0.8 and np.abs(np.nanmedian(x)) < 0.05)

def _pick_price_column(dataset_df: pd.DataFrame) -> pd.Series:
    cols = [c.lower() for c in dataset_df.columns]
    if "adj_close" in cols:
        return pd.to_numeric(dataset_df[dataset_df.columns[cols.index("adj_close")]], errors="coerce")
    if "close" in cols:
        return pd.to_numeric(dataset_df[dataset_df.columns[cols.index("close")]], errors="coerce")
    for c in dataset_df.columns:
        if str(c).lower().endswith("_close"):
            return pd.to_numeric(dataset_df[c], errors="coerce")
    raise KeyError("No price column (adj_close/close) found in dataset.parquet")

def _reconstruct_price_path(
    ds: pd.Series, pred_series: pd.Series, full_prices: pd.DataFrame
) -> pd.Series:
    """
    From 1-day returns, reconstruct a price path over the last-5 window.
    Anchors on the last price BEFORE the first ds (or first-day price if needed).
    """
    pred_series = pd.to_numeric(pred_series, errors="coerce").astype(float)
    ds = pd.to_datetime(ds, errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")

    full_prices = full_prices.copy()
    full_prices["ds"] = pd.to_datetime(full_prices["ds"], errors="coerce", utc=True)\
                               .dt.tz_convert(None).dt.floor("D")

    merged = ds.to_frame(name="ds").merge(full_prices[["ds", "price"]], on="ds", how="left")

    first_day = ds.min()
    prior = full_prices.loc[full_prices["ds"] < first_day, "price"]
    if len(prior) > 0 and np.isfinite(prior.values[-1]):
        p0 = float(prior.values[-1])
    else:
        p0 = float(merged["price"].iloc[0]) if np.isfinite(merged["price"].iloc[0]) else float("nan")

    if not np.isfinite(p0):
        return pred_series  # fall back: no reliable anchor

    path = p0 * np.cumprod(1.0 + pred_series.values)
    return pd.Series(path, index=pred_series.index)

def _rmse(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

def _plot_two(dates, actual, pred, pred_label, rmse, title):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(dates, actual, label="Actual")
    lab = f"{pred_label} (RMSE={rmse:.6f})" if isinstance(rmse, (int, float, np.floating)) and np.isfinite(rmse) else pred_label
    ax.plot(dates, pred, label=lab)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def _plot_three(dates, actual, pred_ohlcv, rmse_ohlcv, pred_all, rmse_all, title):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(dates, actual, label="Actual")
    lab_o = "Prediction_OHLCV" + (f" (RMSE={rmse_ohlcv:.6f})" if isinstance(rmse_ohlcv, (int,float,np.floating)) and np.isfinite(rmse_ohlcv) else "")
    lab_a = "Prediction_all"   + (f" (RMSE={rmse_all:.6f})"    if isinstance(rmse_all, (int,float,np.floating)) and np.isfinite(rmse_all) else "")
    ax.plot(dates, pred_ohlcv, label=lab_o)
    ax.plot(dates, pred_all,   label=lab_a)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def _find_artifact_dir(ticker: str) -> Path:
    """Prefer new layout outputs/last5/<TICKER>, else fallback to data/runs/last5/<TICKER>."""
    p1 = OUTPUTS_DIR / "last5" / ticker
    if p1.exists():
        return p1
    p2 = DATA_DIR / "runs" / "last5" / ticker
    if p2.exists():
        return p2
    return p1  # default

def _load_preds_and_metrics(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict, Dict, Path]:
    run_dir = _find_artifact_dir(ticker)
    all_csv   = run_dir / "pred_last5_all_inputs.csv"
    ohlcv_csv = run_dir / "pred_last5_ohlcv_only.csv"
    all_met   = run_dir / "metrics_all_inputs.json"
    ohlcv_met = run_dir / "metrics_ohlcv_only.json"

    df_all = pd.read_csv(all_csv, parse_dates=["ds"]) if all_csv.exists() else None
    df_ohl = pd.read_csv(ohlcv_csv, parse_dates=["ds"]) if ohlcv_csv.exists() else None
    met_all = _load_json(all_met) or {}
    met_ohl = _load_json(ohlcv_met) or {}
    return df_all, df_ohl, met_all, met_ohl, run_dir

def _load_dataset_prices(ticker: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / ticker / "dataset.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
    try:
        price = _pick_price_column(df)
        return pd.DataFrame({"ds": df["ds"], "price": price})
    except Exception:
        return None

def _ensure_price_df(df_pred: Optional[pd.DataFrame], pred_col: str, prices_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Ensure the prediction frame is in price space and aligned with actual prices:
    - Normalize ds to naive daily.
    - If predictions look like returns, rebuild a price path.
    - Attach 'Actual' price from dataset.parquet (ffill/bfill to avoid tiny gaps).
    """
    if df_pred is None:
        return None
    out = df_pred.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")

    if prices_df is not None:
        base = prices_df.copy()
        base["ds"] = pd.to_datetime(base["ds"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("D")
        out = out.merge(base, on="ds", how="left")  # adds 'price'
        out["Actual"] = pd.to_numeric(out["price"], errors="coerce")
        out.drop(columns=["price"], inplace=True)
        out["Actual"] = out["Actual"].ffill().bfill()

        if _is_returns_like(out[pred_col]):
            fullp = prices_df.sort_values("ds")
            out[pred_col] = _reconstruct_price_path(out["ds"], out[pred_col], fullp)
    else:
        # No price frame available -> leave as-is (may already be prices)
        if "Actual" not in out.columns:
            out["Actual"] = np.nan

    return out

# --------------- UI per ticker ---------------

def _section_for_ticker(ticker: str):
    st.header(ticker)

    df_all, df_ohl, met_all, met_ohl, run_dir = _load_preds_and_metrics(ticker)
    prices_df = _load_dataset_prices(ticker)

    st.caption(f"Artifact dir: {run_dir}")

    rmse_all_m = met_all.get("rmse_last5")
    rmse_ohl_m = met_ohl.get("rmse_last5")
    w_all = (met_all.get("weights") or {})
    w_ohl = (met_ohl.get("weights") or {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (All inputs, from metrics)", f"{rmse_all_m:.6f}" if isinstance(rmse_all_m, (int,float,np.floating)) else "n/a")
    c2.metric("w_xgb (All)", f"{w_all.get('w_xgb', 'n/a')}")
    c3.metric("RMSE (OHLCV only, from metrics)", f"{rmse_ohl_m:.6f}" if isinstance(rmse_ohl_m, (int,float,np.floating)) else "n/a")
    c4.metric("w_xgb (OHLCV)", f"{w_ohl.get('w_xgb', 'n/a')}")

    # Rename for display
    if df_all is not None:
        df_all = df_all.rename(columns={"y_true":"Actual","y_pred_all":"Prediction_all"})
    if df_ohl is not None:
        df_ohl = df_ohl.rename(columns={"y_true":"Actual","y_pred_ohlcv":"Prediction_OHLCV"})

    # Ensure price space & align with actuals
    df_ohl_p = _ensure_price_df(df_ohl, "Prediction_OHLCV", prices_df) if df_ohl is not None else None
    df_all_p = _ensure_price_df(df_all, "Prediction_all",   prices_df) if df_all is not None else None

    # Recompute RMSE in price space (for legend)
    rmse_ohl = _rmse(df_ohl_p["Actual"], df_ohl_p["Prediction_OHLCV"]) if df_ohl_p is not None else float("nan")
    rmse_all = _rmse(df_all_p["Actual"], df_all_p["Prediction_all"])   if df_all_p is not None else float("nan")

    # a) OHLCV vs Actual
    st.subheader("a) OHLCV vs Actual")
    if df_ohl_p is not None and {"ds","Actual","Prediction_OHLCV"}.issubset(df_ohl_p.columns):
        _plot_two(
            df_ohl_p["ds"], df_ohl_p["Actual"], df_ohl_p["Prediction_OHLCV"],
            pred_label="Prediction_OHLCV",
            rmse=rmse_ohl,
            title=f"{ticker} — OHLCV vs Actual (Last 5 trading days)"
        )
    else:
        st.warning("OHLCV prediction artifacts not found. Expected pred_last5_ohlcv_only.csv under outputs/last5 or data/runs/last5.")

    # b) All inputs vs Actual
    st.subheader("b) All inputs vs Actual")
    if df_all_p is not None and {"ds","Actual","Prediction_all"}.issubset(df_all_p.columns):
        _plot_two(
            df_all_p["ds"], df_all_p["Actual"], df_all_p["Prediction_all"],
            pred_label="Prediction_all",
            rmse=rmse_all,
            title=f"{ticker} — All inputs vs Actual (Last 5 trading days)"
        )
    else:
        st.warning("All-inputs prediction artifacts not found. Expected pred_last5_all_inputs.csv under outputs/last5 or data/runs/last5.")

    # c) OHLCV vs All inputs vs Actual
    st.subheader("c) OHLCV vs All inputs vs Actual")
    if df_all_p is not None and df_ohl_p is not None:
        m = df_all_p[["ds","Actual","Prediction_all"]].merge(
            df_ohl_p[["ds","Prediction_OHLCV"]],
            on="ds", how="inner"
        )
        _plot_three(
            m["ds"], m["Actual"],
            m["Prediction_OHLCV"], rmse_ohl,
            m["Prediction_all"],   rmse_all,
            title=f"{ticker} — Compare (Actual vs OHLCV vs All)"
        )
    else:
        st.info("Comparison requires both CSVs; one or both are missing.")

# Tiny debug (kept minimal)
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")

for t in TICKERS:
    _section_for_ticker(t)
