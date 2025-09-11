#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys, os, json
from typing import Optional, List
import streamlit as st

# ── Repo root & src on sys.path (works on Streamlit Cloud and local)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # repo_root/src/mm_ensemble/streamlit_app.py
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Prefer shared paths module; otherwise repo-root defaults
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = REPO_ROOT / "data"
    OUTPUTS_DIR = REPO_ROOT / "outputs"

# ── GitHub (public) raw fallback (you can override via env)
GITHUB_REPO_SLUG = os.getenv("GITHUB_REPO_SLUG", "Poulami-Nandi/mm_ensemble_pred")
GITHUB_BRANCH    = os.getenv("GITHUB_BRANCH", "main")
RAW_PREFIX = f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/{GITHUB_BRANCH}"

# Correct filenames (PRICE plots)
IMG_FILES = {
    "ohlcv_only": "actual_vs_pred_ohlcv_only_PRICE.png",
    "all_inputs": "actual_vs_pred_all_inputs_PRICE.png",
    "compare":    "compare_all_vs_ohlcv_PRICE.png",
}
JSON_FILES = {
    "all_inputs": "metrics_all_inputs.json",
    "ohlcv_only": "metrics_ohlcv_only.json",
}

# --------- Streamlit page ----------
st.set_page_config(page_title="mm_ensemble — Last-5 price demo", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")
st.caption(f"GitHub fallback: {GITHUB_REPO_SLUG}@{GITHUB_BRANCH}")

# --------- Helpers ----------
def _candidate_local_paths(ticker: str, rel_name: str) -> List[Path]:
    """Search new + legacy + repo-relative mirrors."""
    return [
        OUTPUTS_DIR / "last5" / ticker / rel_name,                  # new local
        DATA_DIR    / "runs"  / "last5" / ticker / rel_name,        # legacy local
        REPO_ROOT   / "outputs" / "last5" / ticker / rel_name,      # repo mirror
        REPO_ROOT   / "data"    / "runs"  / "last5" / ticker / rel_name,
    ]

def _load_local_bytes(ticker: str, rel_name: str) -> Optional[bytes]:
    for p in _candidate_local_paths(ticker, rel_name):
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                pass
    return None

def _fetch_raw_bytes(url: str) -> Optional[bytes]:
    try:
        import requests
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None

def _json_local_or_raw(ticker: str, rel_name: str) -> dict:
    # local first
    b = _load_local_bytes(ticker, rel_name)
    if b:
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return {}
    # raw fallback (new path → legacy)
    for base in (f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"):
        url = f"{RAW_PREFIX}/{base}/{rel_name}"
        try:
            jb = _fetch_raw_bytes(url)
            if jb:
                return json.loads(jb.decode("utf-8"))
        except Exception:
            continue
    return {}

def _show_image_anywhere(ticker: str, rel_name: str, caption: str):
    # local bytes
    b = _load_local_bytes(ticker, rel_name)
    if b:
        st.image(BytesIO(b), caption=caption, use_container_width=True)
        return
    # raw bytes (new path first, then legacy)
    tried = []
    for base in (f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"):
        url = f"{RAW_PREFIX}/{base}/{rel_name}"
        tried.append(url)
        b = _fetch_raw_bytes(url)
        if b:
            st.image(BytesIO(b), caption=f"{caption} (from GitHub)", use_container_width=True)
            return
    # Missing
    st.warning("Missing image:\n" + "\n".join(
        [f"• {p}" for p in _candidate_local_paths(ticker, rel_name)]
        + [f"• {u}" for u in tried]
    ))

def _discover_tickers() -> List[str]:
    # prefer local dir discovery
    for root in (OUTPUTS_DIR / "last5", DATA_DIR / "runs" / "last5", REPO_ROOT / "outputs" / "last5"):
        if root.exists():
            ts = sorted([d.name for d in root.iterdir() if d.is_dir()])
            if ts:
                return ts
    return ["SBUX", "PFE"]

def _metric_tiles(met: dict, title: str):
    rmse = met.get("rmse_last5")
    w    = met.get("weights") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(f"RMSE ({title})", "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)))
    c2.metric(f"w_xgb ({title})", "n/a" if w.get("w_xgb") is None else str(w.get("w_xgb")))
    c3.metric(f"w_arima ({title})", "n/a" if w.get("w_arima") is None else str(w.get("w_arima")))

# --------- UI: SBUX + PFE, three plots each ----------
tickers = _discover_tickers()
for t in tickers:
    st.header(t)

    met_all = _json_local_or_raw(t, JSON_FILES["all_inputs"])
    met_ohl = _json_local_or_raw(t, JSON_FILES["ohlcv_only"])

    _metric_tiles(met_all, "All inputs")
    _metric_tiles(met_ohl, "OHLCV only")

    st.subheader("a) OHLCV vs Actual")
    _show_image_anywhere(t, IMG_FILES["ohlcv_only"], "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_image_anywhere(t, IMG_FILES["all_inputs"], "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_image_anywhere(t, IMG_FILES["compare"], "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

st.success("Done. Displaying precomputed artifacts — no training or inference is run here.")
