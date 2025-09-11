#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, json, os
from typing import Optional
import streamlit as st

# ── Make 'src' importable if app runs via Streamlit Cloud from repo root
here = Path(__file__).resolve()
repo_root_guess = here.parents[2]  # .../repo_root/src/mm_ensemble/streamlit_app.py -> up 2 = repo_root
src_dir = repo_root_guess / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer shared paths module; otherwise fall back to repo_root/data & repo_root/outputs
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = repo_root_guess / "data"
    OUTPUTS_DIR = repo_root_guess / "outputs"

# ── GitHub fallback config
GITHUB_REPO_SLUG = os.getenv("GITHUB_REPO_SLUG", "Poulami-Nandi/mm_ensemble_pred")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

def _raw_url(rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/{GITHUB_BRANCH}/{rel_path}"

st.set_page_config(page_title="mm_ensemble — Last-5 price demo (static)", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")
st.caption(f"GitHub fallback: {GITHUB_REPO_SLUG}@{GITHUB_BRANCH}")

# ---------- helpers: search BOTH local and GitHub raw per-file ----------

def _bases_for(ticker: str) -> list[Path]:
    """Local candidate base dirs in priority order."""
    return [
        OUTPUTS_DIR / "last5" / ticker,            # new
        DATA_DIR    / "runs" / "last5" / ticker,   # legacy
    ]

def _first_existing_local(ticker: str, rel_name: str) -> Optional[Path]:
    """Return first existing local path for a given relative filename."""
    for base in _bases_for(ticker):
        p = base / rel_name
        if p.exists():
            return p
    return None

def _json_from_anywhere(ticker: str, rel_name: str) -> dict:
    """Load JSON from local path if present, else from GitHub raw."""
    # Try local
    p = _first_existing_local(ticker, rel_name)
    if p:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    # Try GitHub raw
    import requests
    rel_repo_path = None
    # Prefer new layout path on GitHub first
    for base in [f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"]:
        candidate = f"{base}/{rel_name}"
        url = _raw_url(candidate)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                return r.json()
        except Exception:
            pass
    return {}

def _show_image_from_anywhere(ticker: str, rel_name: str, caption: str):
    """Display image from local if available, else from GitHub raw; warn if neither exists."""
    # Local
    p = _first_existing_local(ticker, rel_name)
    if p:
        st.image(str(p), caption=caption, use_column_width=True)
        return
    # GitHub raw fallback
    import requests
    tried = []
    for base in [f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"]:
        candidate = f"{base}/{rel_name}"
        url = _raw_url(candidate)
        tried.append(url)
        try:
            # HEAD first to avoid pulling big images unnecessarily
            h = requests.head(url, timeout=10)
            if h.status_code == 200:
                st.image(url, caption=f"{caption} (served from GitHub)", use_column_width=True)
                return
        except Exception:
            pass
    # Not found anywhere
    details = "\n  • " + "\n  • ".join(
        [str(b / rel_name) for b in _bases_for(ticker)]
        + tried
    )
    st.warning(f"Missing artifact:\n{details}")

def _discover_tickers() -> list[str]:
    """Discover tickers locally; if none, fall back to SBUX/PFE (your demo set)."""
    new_root = OUTPUTS_DIR / "last5"
    if new_root.exists():
        ts = sorted([d.name for d in new_root.iterdir() if d.is_dir()])
        if ts:
            return ts
    old_root = DATA_DIR / "runs" / "last5"
    if old_root.exists():
        ts = sorted([d.name for d in old_root.iterdir() if d.is_dir()])
        if ts:
            return ts
    # Fallback to your known tickers present on GitHub
    return ["SBUX", "PFE"]

def _metric_tile_row(title_rmse: str, rmse, title_wx: str, wx, title_wa: str, wa):
    c1, c2, c3 = st.columns(3)
    c1.metric(title_rmse, "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)))
    c2.metric(title_wx, "n/a" if wx is None else str(wx))
    c3.metric(title_wa, "n/a" if wa is None else str(wa))

# ---------- UI: static only (no training, no inference) ----------

tickers = _discover_tickers()
for t in tickers:
    st.header(t)

    # Metrics (read whichever exists: local → GitHub fallback)
    met_all = _json_from_anywhere(t, "metrics_all_inputs.json")
    met_ohl = _json_from_anywhere(t, "metrics_ohlcv_only.json")

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
    _show_image_from_anywhere(t, "actual_vs_pred_ohlcv_only.png", "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_image_from_anywhere(t, "actual_vs_pred_all_inputs.png", "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_image_from_anywhere(t, "compare_all_vs_ohlcv.png", "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

st.success("Done. Displaying precomputed artifacts — no training or inference is run here.")
