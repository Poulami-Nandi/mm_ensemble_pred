#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import json
import streamlit as st

# ── Bootstrap: make sure we can import from src/
here = Path(__file__).resolve()
repo_root = here.parents[2]  # repo_root/src/mm_ensemble/streamlit_app.py -> up 2
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer shared paths; fallback to repo root /data and /outputs if utils not present
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = repo_root / "data"
    OUTPUTS_DIR = repo_root / "outputs"

st.set_page_config(page_title="mm_ensemble — Last-5 price demo (static)", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")

st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")

# -------------- Helpers (no ML, no plotting libs) --------------

def _artifact_dir_for(ticker: str) -> Path:
    p1 = OUTPUTS_DIR / "last5" / ticker
    if p1.exists():
        return p1
    p2 = DATA_DIR / "runs" / "last5" / ticker  # legacy location
    if p2.exists():
        return p2
    return p1  # default to new location

def _read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _discover_tickers() -> list[str]:
    # Prefer new layout
    p1 = OUTPUTS_DIR / "last5"
    if p1.exists():
        ts = [d.name for d in p1.iterdir() if d.is_dir()]
        if ts:
            return sorted(ts)
    # Fallback to legacy layout
    p2 = DATA_DIR / "runs" / "last5"
    if p2.exists():
        ts = [d.name for d in p2.iterdir() if d.is_dir()]
        if ts:
            return sorted(ts)
    # Final fallback (your demo tickers)
    return ["SBUX", "PFE"]

def _metric_tile_row(title_rmse: str, rmse, title_wx: str, wx, title_wa: str, wa):
    c1, c2, c3 = st.columns(3)
    c1.metric(title_rmse, "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)))
    c2.metric(title_wx, "n/a" if wx is None else str(wx))
    c3.metric(title_wa, "n/a" if wa is None else str(wa))

def _show_img_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Missing: {path}")

# -------------- UI (static artifacts only) --------------

tickers = _discover_tickers()
for t in tickers:
    st.header(t)
    run_dir = _artifact_dir_for(t)
    st.caption(f"Artifact dir: {run_dir}")

    # Metrics files
    met_all = _read_json(run_dir / "metrics_all_inputs.json")
    met_ohl = _read_json(run_dir / "metrics_ohlcv_only.json")

    rmse_all = met_all.get("rmse_last5", None)
    w_all = (met_all.get("weights") or {})
    rmse_ohl = met_ohl.get("rmse_last5", None)
    w_ohl = (met_ohl.get("weights") or {})

    # Row 1: All-inputs metrics
    _metric_tile_row(
        "RMSE (All inputs)", rmse_all,
        "w_xgb (All)", w_all.get("w_xgb"),
        "w_arima (All)", w_all.get("w_arima"),
    )
    # Row 2: OHLCV-only metrics
    _metric_tile_row(
        "RMSE (OHLCV only)", rmse_ohl,
        "w_xgb (OHLCV)", w_ohl.get("w_xgb"),
        "w_arima (OHLCV)", w_ohl.get("w_arima"),
    )

    st.subheader("a) OHLCV vs Actual")
    _show_img_if_exists(run_dir / "actual_vs_pred_ohlcv_only.png", "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_img_if_exists(run_dir / "actual_vs_pred_all_inputs.png", "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_img_if_exists(run_dir / "compare_all_vs_ohlcv.png", "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

st.success("Done. This app only displays precomputed artifacts—no training or inference is run here.")
