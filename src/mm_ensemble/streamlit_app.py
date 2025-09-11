#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, json
import streamlit as st

# ── Make sure we can import from src/ when running via Streamlit Cloud
here = Path(__file__).resolve()
repo_root = here.parents[2]  # repo_root/src/mm_ensemble/streamlit_app.py -> up 2
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer shared paths; fallback to repo root /data and /outputs
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = repo_root / "data"
    OUTPUTS_DIR = repo_root / "outputs"

st.set_page_config(page_title="mm_ensemble — Last-5 price demo (static)", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")

st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")

# ---------- helpers: search BOTH new and legacy locations per-file ----------

def _bases_for(ticker: str) -> list[Path]:
    """Candidate base dirs in priority order."""
    return [
        OUTPUTS_DIR / "last5" / ticker,      # new
        DATA_DIR    / "runs"   / "last5" / ticker,  # legacy
    ]

def _first_existing(ticker: str, rel_name: str) -> Path | None:
    """Return first existing full path for a given relative filename."""
    for base in _bases_for(ticker):
        p = base / rel_name
        if p.exists():
            return p
    return None

def _read_json_first(ticker: str, rel_name: str) -> dict:
    p = _first_existing(ticker, rel_name)
    if not p:
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _metric_tile_row(title_rmse: str, rmse, title_wx: str, wx, title_wa: str, wa):
    c1, c2, c3 = st.columns(3)
    c1.metric(title_rmse, "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)))
    c2.metric(title_wx, "n/a" if wx is None else str(wx))
    c3.metric(title_wa, "n/a" if wa is None else str(wa))

def _show_img_for(ticker: str, rel_name: str, caption: str):
    p = _first_existing(ticker, rel_name)
    if p:
        st.image(str(p), caption=caption, use_column_width=True)
    else:
        # Show both searched locations to make it obvious what’s missing
        bases = "  • " + "\n  • ".join(str(b / rel_name) for b in _bases_for(ticker))
        st.warning(f"Missing:\n{bases}")

def _discover_tickers() -> list[str]:
    """Find tickers from either new or legacy tree; fall back to demo defaults."""
    new_root = OUTPUTS_DIR / "last5"
    if new_root.exists():
        ts = sorted([d.name for d in new_root.iterdir() if d.is_dir()])
        if ts: return ts
    old_root = DATA_DIR / "runs" / "last5"
    if old_root.exists():
        ts = sorted([d.name for d in old_root.iterdir() if d.is_dir()])
        if ts: return ts
    return ["SBUX", "PFE"]

# ---------- UI: static only (no training, no inference) ----------

tickers = _discover_tickers()
for t in tickers:
    st.header(t)

    # Metrics (read whichever file exists)
    met_all = _read_json_first(t, "metrics_all_inputs.json")
    met_ohl = _read_json_first(t, "metrics_ohlcv_only.json")

    rmse_all = met_all.get("rmse_last5", None)
    w_all    = (met_all.get("weights") or {})
    rmse_ohl = met_ohl.get("rmse_last5", None)
    w_ohl    = (met_ohl.get("weights") or {})

    _metric_tile_row(
        "RMSE (All inputs)", rmse_all,
        "w_xgb (All)", w_all.get("w_xgb"),
        "w_arima (All)", w_all.get("w_arima"),
    )
    _metric_tile_row(
        "RMSE (OHLCV only)", rmse_ohl,
        "w_xgb (OHLCV)", w_ohl.get("w_xgb"),
        "w_arima (OHLCV)", w_ohl.get("w_arima"),
    )

    st.subheader("a) OHLCV vs Actual")
    _show_img_for(t, "actual_vs_pred_ohlcv_only.png", "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_img_for(t, "actual_vs_pred_all_inputs.png", "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_img_for(t, "compare_all_vs_ohlcv.png", "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

st.success("Done. This app only displays precomputed artifacts—no training or inference is run here.")
