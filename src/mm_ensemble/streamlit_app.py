#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, os, json
from typing import Optional, List
import streamlit as st

# ── Find repo root and make 'src' importable when running on Streamlit Cloud
HERE = Path(__file__).resolve()
# .../repo_root/src/mm_ensemble/streamlit_app.py  -> parents[2] == repo_root
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Prefer shared paths; fall back to repo-root-relative folders
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = REPO_ROOT / "data"
    OUTPUTS_DIR = REPO_ROOT / "outputs"

# ── GitHub raw fallback (no probing; let Streamlit fetch directly)
GITHUB_REPO_SLUG = os.getenv("GITHUB_REPO_SLUG", "Poulami-Nandi/mm_ensemble_pred")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
RAW_PREFIX = f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/{GITHUB_BRANCH}"

st.set_page_config(page_title="mm_ensemble — Last-5 price demo (static)", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")
st.caption(f"GitHub fallback: {GITHUB_REPO_SLUG}@{GITHUB_BRANCH}")

# ---------- helpers: search local (multiple roots) or display from GitHub raw ----------

def _candidate_local_paths(ticker: str, rel_name: str) -> List[Path]:
    """Search in new layout, legacy layout, and repo-root-relative mirrors."""
    return [
        OUTPUTS_DIR / "last5" / ticker / rel_name,              # new local
        DATA_DIR / "runs" / "last5" / ticker / rel_name,        # legacy local
        REPO_ROOT / "outputs" / "last5" / ticker / rel_name,    # repo-root-relative (when OUTPUTS_DIR not set)
        REPO_ROOT / "data" / "runs" / "last5" / ticker / rel_name,
    ]

def _first_existing_local(ticker: str, rel_name: str) -> Optional[Path]:
    for p in _candidate_local_paths(ticker, rel_name):
        if p.exists():
            return p
    return None

def _json_local_or_raw(ticker: str, rel_name: str) -> dict:
    """Try local JSON; if missing, fetch from GitHub raw; otherwise return {}."""
    p = _first_existing_local(ticker, rel_name)
    if p:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    # Remote JSON fallback (best-effort)
    try:
        import requests
        for base in [f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"]:
            url = f"{RAW_PREFIX}/{base}/{rel_name}"
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and r.text.strip():
                return r.json()
    except Exception:
        pass
    return {}

def _show_image(ticker: str, rel_name: str, caption: str):
    """Show image from first local hit; if none, render raw GitHub URL directly (no HEAD)."""
    p = _first_existing_local(ticker, rel_name)
    if p:
        st.image(str(p), caption=caption, use_column_width=True)
        return
    # Remote (Streamlit will fetch the PNG; works with public repos)
    # Try new layout first, then legacy
    url_new = f"{RAW_PREFIX}/outputs/last5/{ticker}/{rel_name}"
    url_old = f"{RAW_PREFIX}/data/runs/last5/{ticker}/{rel_name}"
    # We won’t probe — just try to render the new URL first; if it truly 404s, the image widget shows broken,
    # so we add a tiny toggle to try legacy immediately.
    tabs = st.tabs(["new path", "legacy path"])
    with tabs[0]:
        st.image(url_new, caption=f"{caption} (from GitHub raw, new path)", use_column_width=True)
    with tabs[1]:
        st.image(url_old, caption=f"{caption} (from GitHub raw, legacy path)", use_column_width=True)

def _discover_tickers() -> List[str]:
    # Prefer local discovery (when running locally)
    roots = [OUTPUTS_DIR / "last5", DATA_DIR / "runs" / "last5", REPO_ROOT / "outputs" / "last5", REPO_ROOT / "data" / "runs" / "last5"]
    for r in roots:
        if r.exists():
            tickers = sorted([d.name for d in r.iterdir() if d.is_dir()])
            if tickers:
                return tickers
    # Fallback to expected demo tickers
    return ["SBUX", "PFE"]

def _metric_tiles(met: dict, title_prefix: str):
    rmse = met.get("rmse_last5")
    w = met.get("weights") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(f"RMSE ({title_prefix})", "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)))
    c2.metric(f"w_xgb ({title_prefix})", "n/a" if w.get("w_xgb") is None else str(w.get("w_xgb")))
    c3.metric(f"w_arima ({title_prefix})", "n/a" if w.get("w_arima") is None else str(w.get("w_arima")))

# ---------- UI: static only (no training/inference) ----------

for ticker in _discover_tickers():
    st.header(ticker)

    met_all = _json_local_or_raw(ticker, "metrics_all_inputs.json")
    met_ohl = _json_local_or_raw(ticker, "metrics_ohlcv_only.json")
    _metric_tiles(met_all, "All inputs")
    _metric_tiles(met_ohl, "OHLCV only")

    st.subheader("a) OHLCV vs Actual")
    _show_image(ticker, "actual_vs_pred_ohlcv_only.png", "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_image(ticker, "actual_vs_pred_all_inputs.png", "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_image(ticker, "compare_all_vs_ohlcv.png", "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

st.success("Done. Displaying precomputed artifacts — no training or inference is run here.")
