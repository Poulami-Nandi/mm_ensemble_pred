#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import json
import pandas as pd
import streamlit as st

# ── Bootstrap: ensure 'src' is importable when running on Cloud or locally
here = Path(__file__).resolve()
repo_root = here.parents[2]  # .../repo_root/src/mm_ensemble/streamlit_app.py -> up two = repo root
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer your shared path utility; fall back to repo_root/data|outputs if missing
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = repo_root / "data"
    OUTPUTS_DIR = repo_root / "outputs"

st.set_page_config(page_title="mm_ensemble — Last-5 demo", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Demo")

# Sidebar controls
tickers = ["SBUX", "PFE"]
ticker = st.sidebar.selectbox("Ticker", tickers, index=0)
model_choice = st.sidebar.radio("Model", ["All inputs", "OHLCV only"], index=0)

run_dir = OUTPUTS_DIR / "last5" / ticker
if model_choice == "All inputs":
    pred_csv = run_dir / "pred_last5_all_inputs.csv"
    metrics_json = run_dir / "metrics_all_inputs.json"
    plot_png = run_dir / "actual_vs_pred_all_inputs.png"
else:
    pred_csv = run_dir / "pred_last5_ohlcv_only.csv"
    metrics_json = run_dir / "metrics_ohlcv_only.json"
    plot_png = run_dir / "actual_vs_pred_ohlcv_only.png"

# Tiny debug (useful on Cloud)
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")

# Metrics
metrics = None
if metrics_json.exists():
    try:
        metrics = json.loads(metrics_json.read_text())
    except Exception:
        metrics = None

if metrics:
    rmse = metrics.get("rmse_last5")
    weights = metrics.get("weights", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (last-5)", f"{rmse:.6f}" if isinstance(rmse, (int, float)) else "n/a")
    c2.metric("w_xgb", f"{weights.get('w_xgb', 'n/a')}")
    c3.metric("w_arima", f"{weights.get('w_arima', 'n/a')}")

# Main plot: prefer saved PNG, else plot from CSV
st.subheader(f"{ticker} — {model_choice}")
if plot_png.exists():
    st.image(str(plot_png), caption=plot_png.name, use_column_width=True)
else:
    if pred_csv.exists():
        try:
            df = pd.read_csv(pred_csv, parse_dates=["ds"])
            st.line_chart(df.set_index("ds"))
        except Exception as e:
            st.error(f"Failed to render from CSV: {e}")
    else:
        st.warning("No artifacts found for this selection.\n"
                   f"Expected:\n• {plot_png}\n• {pred_csv}\n• {metrics_json}")

# Comparison tab
tab1, tab2 = st.tabs(["Single model", "Compare (All vs OHLCV)"])
with tab2:
    compare_png = run_dir / "compare_all_vs_ohlcv.png"
    if compare_png.exists():
        st.image(str(compare_png), caption=compare_png.name, use_column_width=True)
    else:
        all_csv = run_dir / "pred_last5_all_inputs.csv"
        ohlcv_csv = run_dir / "pred_last5_ohlcv_only.csv"
        if all_csv.exists() and ohlcv_csv.exists():
            try:
                da = pd.read_csv(all_csv, parse_dates=["ds"]).rename(columns={"y_pred_all": "pred_all"})
                do = pd.read_csv(ohlcv_csv, parse_dates=["ds"]).rename(columns={"y_pred_ohlcv": "pred_ohlcv"})
                m = da.merge(do, on=["ds", "y_true"], how="inner")
                st.line_chart(m.set_index("ds")[["y_true", "pred_ohlcv", "pred_all"]])
            except Exception as e:
                st.error(f"Failed to render comparison: {e}")
        else:
            st.info("Comparison image/CSVs not found.")
