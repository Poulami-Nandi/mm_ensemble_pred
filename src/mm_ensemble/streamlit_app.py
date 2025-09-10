#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, json
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

def _load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _plot_two(dates, actual, pred, pred_label, rmse, title):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(dates, actual, label="Actual")
    lab = f"{pred_label} (RMSE={rmse:.6f})" if isinstance(rmse, (int, float, float)) else pred_label
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
    lab_o = "Prediction_OHLCV" + (f" (RMSE={rmse_ohlcv:.6f})" if isinstance(rmse_ohlcv, (int, float)) else "")
    lab_a = "Prediction_all"   + (f" (RMSE={rmse_all:.6f})"    if isinstance(rmse_all, (int, float)) else "")
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

def _section_for_ticker(ticker: str):
    st.header(ticker)

    run_dir = OUTPUTS_DIR / "last5" / ticker

    # Files
    all_csv   = run_dir / "pred_last5_all_inputs.csv"
    ohlcv_csv = run_dir / "pred_last5_ohlcv_only.csv"
    all_met   = run_dir / "metrics_all_inputs.json"
    ohlcv_met = run_dir / "metrics_ohlcv_only.json"

    # Load metrics (weights + RMSE)
    met_all = _load_json(all_met) or {}
    met_oh  = _load_json(ohlcv_met) or {}
    rmse_all = met_all.get("rmse_last5")
    rmse_ohl = met_oh.get("rmse_last5")
    w_all = (met_all.get("weights") or {})
    w_ohl = (met_oh.get("weights") or {})

    # Show weights & RMSE per stock
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (All inputs)", f"{rmse_all:.6f}" if isinstance(rmse_all, (int,float)) else "n/a")
    c2.metric("w_xgb (All)", f"{w_all.get('w_xgb', 'n/a')}")
    c3.metric("RMSE (OHLCV only)", f"{rmse_ohl:.6f}" if isinstance(rmse_ohl, (int,float)) else "n/a")
    c4.metric("w_xgb (OHLCV)", f"{w_ohl.get('w_xgb', 'n/a')}")

    # Load CSVs
    df_all = pd.read_csv(all_csv, parse_dates=["ds"]) if all_csv.exists() else None
    df_ohl = pd.read_csv(ohlcv_csv, parse_dates=["ds"]) if ohlcv_csv.exists() else None

    # Rename series for plotting/legend (display names only)
    if df_all is not None:
        df_all = df_all.rename(columns={
            "y_true": "Actual",
            "y_pred_all": "Prediction_all",
        })
    if df_ohl is not None:
        df_ohl = df_ohl.rename(columns={
            "y_true": "Actual",
            "y_pred_ohlcv": "Prediction_OHLCV",
        })

    # 3 plots per stock:
    # a) OHLCV vs Actual
    st.subheader("a) OHLCV vs Actual")
    if df_ohl is not None and {"ds","Actual","Prediction_OHLCV"}.issubset(df_ohl.columns):
        _plot_two(
            df_ohl["ds"], df_ohl["Actual"], df_ohl["Prediction_OHLCV"],
            pred_label="Prediction_OHLCV",
            rmse=rmse_ohl,
            title=f"{ticker} — OHLCV vs Actual (Last 5 trading days)"
        )
    else:
        st.warning("OHLCV prediction CSV not found or malformed.")

    # b) All inputs vs Actual
    st.subheader("b) All inputs vs Actual")
    if df_all is not None and {"ds","Actual","Prediction_all"}.issubset(df_all.columns):
        _plot_two(
            df_all["ds"], df_all["Actual"], df_all["Prediction_all"],
            pred_label="Prediction_all",
            rmse=rmse_all,
            title=f"{ticker} — All inputs vs Actual (Last 5 trading days)"
        )
    else:
        st.warning("All-inputs prediction CSV not found or malformed.")

    # c) OHLCV vs All inputs vs Actual
    st.subheader("c) OHLCV vs All inputs vs Actual")
    if df_all is not None and df_ohl is not None:
        # inner-join on ds; prefer 'Actual' from df_all
        m = df_all[["ds","Actual","Prediction_all"]].merge(
            df_ohl[["ds","Prediction_OHLCV"]],
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

# Render sections for each ticker (SBUX, PFE)
for t in TICKERS:
    _section_for_ticker(t)
