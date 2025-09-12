#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
streamlit_app.py

What this app shows
--------------------------------------
This is a *read-only* viewer for artifacts you already computed elsewhere.
For each ticker (e.g., SBUX, PFE) it displays:
  1) Three pre-rendered plots for the last-5-day price demo:
     a) OHLCV vs Actual
     b) All inputs vs Actual
     c) OHLCV vs All inputs vs Actual
  2) A small metrics panel showing RMSE and the learned ensemble weights.
     - RMSE is taken from outputs/last5/summary.json (if available).
     - The weights (w_xgb, w_arima) come from per-ticker metrics JSON.

Where files are searched
------------------------
- First we try local paths in the deployed environment:
    outputs/last5/<TICKER>/...      (new layout)
    data/runs/last5/<TICKER>/...    (legacy layout)
    <repo_root>/outputs/...         (repo mirror when running locally)
    <repo_root>/data/...            (legacy mirror)
- If not found locally, we fetch from GitHub "raw" URLs:
    https://raw.githubusercontent.com/<owner>/<repo>/<branch>/...

No training or inference is performed here.
This app only *reads* existing images and JSON files and displays them.
"""

from pathlib import Path
from io import BytesIO
import sys, os, json
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd


# =============================================================================
# Locate the repository root and ensure "src" is importable
# (works on Streamlit Cloud and when running locally)
# =============================================================================

# Path to this file
HERE = Path(__file__).resolve()

# We expect the app file to live at: repo_root/src/mm_ensemble/streamlit_app.py
# So, repo_root = HERE.parents[2]
REPO_ROOT = HERE.parents[2]

# Make sure "src" is on sys.path so we can do: from mm_ensemble.utils.paths import ...
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Prefer the shared paths utility (lets users override with env vars).
# If it's not importable, fall back to repo-root/data and repo-root/outputs.
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR = REPO_ROOT / "data"
    OUTPUTS_DIR = REPO_ROOT / "outputs"


# =============================================================================
# GitHub raw fallback (public). You can override with environment variables.
# =============================================================================

GITHUB_REPO_SLUG = os.getenv("GITHUB_REPO_SLUG", "Poulami-Nandi/mm_ensemble_pred")
GITHUB_BRANCH    = os.getenv("GITHUB_BRANCH", "main")
RAW_PREFIX = f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/{GITHUB_BRANCH}"


# =============================================================================
# Filenames for artifacts we expect to find per ticker
# (PRICE plots and their metrics JSON)
# =============================================================================

# The three image files we expect (already rendered by your training scripts)
IMG_FILES = {
    "ohlcv_only": "actual_vs_pred_ohlcv_only_PRICE.png",
    "all_inputs": "actual_vs_pred_all_inputs_PRICE.png",
    "compare":    "compare_all_vs_ohlcv_PRICE.png",
}

# Per-ticker metrics (for weights; RMSE will be overridden from summary.json)
JSON_FILES = {
    "all_inputs": "metrics_all_inputs.json",
    "ohlcv_only": "metrics_ohlcv_only.json",
}


# =============================================================================
# Streamlit page configuration and header
# =============================================================================

st.set_page_config(page_title="mm_ensemble — Last-5 price demo", layout="wide")
st.title("Multimodal Ensemble — Last-5 Day Price Demo (Static Artifacts)")
st.caption(f"DATA_DIR = {DATA_DIR}")
st.caption(f"OUTPUTS_DIR = {OUTPUTS_DIR}")
st.caption(f"GitHub fallback: {GITHUB_REPO_SLUG}@{GITHUB_BRANCH}")


# =============================================================================
# General helpers: date normalization, file fetching, etc.
# =============================================================================

def _norm_ds(s: pd.Series) -> pd.Series:
    """
    Normalize a column of dates/times to simple daily (no timezone) timestamps.
    This keeps charts and merges predictable.
    """
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")


def _candidate_local_paths(ticker: str, rel_name: str) -> List[Path]:
    """
    Build a list of *local* paths to try for a given file (image or JSON).
    We include new layout, legacy layout, and repo-mirror paths.
    """
    return [
        OUTPUTS_DIR / "last5" / ticker / rel_name,                  # new local
        DATA_DIR    / "runs"  / "last5" / ticker / rel_name,        # legacy local
        REPO_ROOT   / "outputs" / "last5" / ticker / rel_name,      # repo mirror (for local dev)
        REPO_ROOT   / "data"    / "runs"  / "last5" / ticker / rel_name,  # legacy repo mirror
    ]


def _load_local_bytes(ticker: str, rel_name: str) -> Optional[bytes]:
    """
    Try to read a file as bytes from any of the candidate local paths.
    Returns bytes on success, or None if all attempts fail.
    """
    for p in _candidate_local_paths(ticker, rel_name):
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                # Keep trying other candidates
                pass
    return None


def _fetch_raw_bytes(url: str) -> Optional[bytes]:
    """
    Download raw bytes from a URL (used for GitHub raw).
    Returns bytes on success, None on failure.
    """
    try:
        import requests
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


def _json_local_or_raw(ticker: str, rel_name: str) -> dict:
    """
    Try to load JSON from local paths first.
    If not found, try GitHub raw in both 'new' and 'legacy' folders.
    Returns a dict (empty dict on failure).
    """
    # 1) Local first
    b = _load_local_bytes(ticker, rel_name)
    if b:
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return {}

    # 2) GitHub raw fallback: try new layout, then legacy layout
    for base in (f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"):
        url = f"{RAW_PREFIX}/{base}/{rel_name}"
        try:
            jb = _fetch_raw_bytes(url)
            if jb:
                return json.loads(jb.decode("utf-8"))
        except Exception:
            continue

    # Nothing worked
    return {}


def _show_image_anywhere(ticker: str, rel_name: str, caption: str):
    """
    Display an image:
      - Try local bytes first.
      - If missing, try GitHub raw (new then legacy paths).
      - If still missing, show a warning listing the paths we tried.
    """
    # 1) Local
    b = _load_local_bytes(ticker, rel_name)
    if b:
        st.image(BytesIO(b), caption=caption, use_container_width=True)
        return

    # 2) GitHub raw
    tried = []
    for base in (f"outputs/last5/{ticker}", f"data/runs/last5/{ticker}"):
        url = f"{RAW_PREFIX}/{base}/{rel_name}"
        tried.append(url)
        b = _fetch_raw_bytes(url)
        if b:
            st.image(BytesIO(b), caption=f"{caption} (from GitHub)", use_container_width=True)
            return

    # 3) If all failed, explain exactly what we tried
    st.warning("Missing image:\n" + "\n".join(
        [f"• {p}" for p in _candidate_local_paths(ticker, rel_name)]
        + [f"• {u}" for u in tried]
    ))


def _discover_tickers() -> List[str]:
    """
    Try to auto-discover which tickers are available by scanning local folders.
    If we can't find any, just return ["SBUX", "PFE"] as a sensible default.
    """
    for root in (OUTPUTS_DIR / "last5", DATA_DIR / "runs" / "last5", REPO_ROOT / "outputs" / "last5"):
        if root.exists():
            ts = sorted([d.name for d in root.iterdir() if d.is_dir()])
            if ts:
                return ts
    return ["SBUX", "PFE"]


def _metric_tiles(met: dict, title: str):
    """
    Draw three small metric tiles for:
      - RMSE (from summary.json override if present)
      - w_xgb
      - w_arima

    'met' is expected to be the per-config metrics dict, e.g. the contents of
    metrics_all_inputs.json or metrics_ohlcv_only.json (possibly with RMSE overwritten).
    """
    rmse = met.get("rmse_last5")
    w    = met.get("weights") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"RMSE ({title})",
        "n/a" if rmse is None else (f"{rmse:.6f}" if isinstance(rmse, (int, float)) else str(rmse)),
    )
    c2.metric(
        f"w_xgb ({title})",
        "n/a" if w.get("w_xgb") is None else str(w.get("w_xgb")),
    )
    c3.metric(
        f"w_arima ({title})",
        "n/a" if w.get("w_arima") is None else str(w.get("w_arima")),
    )


# =============================================================================
# Load summary.json once and extract RMSEs to override per-ticker metrics
# =============================================================================

def _load_summary_rmse() -> Dict[str, Dict[str, float]]:
    """
    Read outputs/last5/summary.json (locally or from GitHub) and return:
      { "<TICKER>": { "all_inputs": <rmse>, "ohlcv_only": <rmse> }, ... }

    We *only* use this to override the RMSE shown in the metric tiles.
    """
    # Try local copies first
    local_candidates = [
        OUTPUTS_DIR / "last5" / "summary.json",
        REPO_ROOT / "outputs" / "last5" / "summary.json",
    ]

    payload = None
    for p in local_candidates:
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                continue

    # GitHub raw fallback
    if payload is None:
        url = f"{RAW_PREFIX}/outputs/last5/summary.json"
        jb = _fetch_raw_bytes(url)
        if jb:
            try:
                payload = json.loads(jb.decode("utf-8"))
            except Exception:
                payload = None

    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(payload, dict):
        return out

    # Expected shape:
    # {"tickers": {"PFE": {"all_inputs":{"rmse_last5":...},"ohlcv_only":{"rmse_last5":...}}, ...}}
    tickers = payload.get("tickers", {})
    if isinstance(tickers, dict):
        for t, entry in tickers.items():
            rm_all = entry.get("all_inputs", {}).get("rmse_last5")
            rm_ohl = entry.get("ohlcv_only", {}).get("rmse_last5")
            one: Dict[str, float] = {}
            if isinstance(rm_all, (int, float)):
                one["all_inputs"] = float(rm_all)
            if isinstance(rm_ohl, (int, float)):
                one["ohlcv_only"] = float(rm_ohl)
            if one:
                out[t] = one
    return out


# =============================================================================
# Optional: load and plot full close price history (if present)
# =============================================================================

def _read_parquet_from_bytes(b: bytes) -> Optional[pd.DataFrame]:
    """Helper to read a Parquet file from in-memory bytes (for GitHub raw)."""
    try:
        return pd.read_parquet(BytesIO(b))
    except Exception:
        return None


def _read_csv_from_bytes(b: bytes) -> Optional[pd.DataFrame]:
    """Helper to read a CSV file from in-memory bytes (for GitHub raw)."""
    try:
        return pd.read_csv(BytesIO(b))
    except Exception:
        return None


def _load_prices_df(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load a simple two-column time series for plotting close prices:
      columns: ds, Close

    Tries local Parquet/CSV first, then GitHub raw.
    If neither exists, return None so the UI can skip the price chart.
    """
    local_candidates = [
        DATA_DIR / ticker / "prices.parquet",
        DATA_DIR / ticker / "prices.csv",
        REPO_ROOT / "data" / ticker / "prices.parquet",
        REPO_ROOT / "data" / ticker / "prices.csv",
    ]

    df, src = None, None

    # Local files first
    for p in local_candidates:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                src = f"local: {p}"
                break
            except Exception:
                continue

    # GitHub raw fallback
    if df is None:
        for rp in (f"data/{ticker}/prices.parquet", f"data/{ticker}/prices.csv"):
            url = f"{RAW_PREFIX}/{rp}"
            b = _fetch_raw_bytes(url)
            if not b:
                continue
            df = _read_parquet_from_bytes(b) if rp.endswith(".parquet") else _read_csv_from_bytes(b)
            if df is not None and not df.empty:
                src = f"github: {url}"
                break

    # If nothing worked, just skip
    if df is None or df.empty:
        return None

    # Normalize and pick the close column
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    df["ds"] = _norm_ds(df["ds"])

    price_col = None
    if "close" in df.columns:
        price_col = "close"
    elif "adj_close" in df.columns:
        price_col = "adj_close"
    if price_col is None:
        return None

    # Keep only ds + Close, sorted, and remember where it came from
    df = df[["ds", price_col]].dropna().sort_values("ds")
    df = df.rename(columns={price_col: "Close"})
    df.attrs["source"] = src
    return df


# =============================================================================
# Main UI: discover tickers, show metrics, show plots, optional price chart
# =============================================================================

rmse_summary = _load_summary_rmse()   # Only used to override RMSE in tiles
tickers = _discover_tickers()         # Try to find tickers locally (fallback to ["SBUX","PFE"])

for t in tickers:
    st.header(t)

    # Optional: full-history close price chart (if we can find prices.*)
    prices = _load_prices_df(t)
    if prices is not None and not prices.empty:
        st.subheader("Close price — full history")
        # Streamlit convenience chart; the x-axis will be the index
        st.line_chart(prices.set_index("ds")["Close"])
        if prices.attrs.get("source"):
            st.caption(f"source: {prices.attrs['source']}")

    # Load per-ticker metrics JSON (for weights, etc.)
    met_all = _json_local_or_raw(t, JSON_FILES["all_inputs"])
    met_ohl = _json_local_or_raw(t, JSON_FILES["ohlcv_only"])

    # Override ONLY the RMSE values using the summary.json (if present)
    if t in rmse_summary:
        if "all_inputs" in rmse_summary[t]:
            met_all = dict(met_all or {})
            met_all["rmse_last5"] = rmse_summary[t]["all_inputs"]
        if "ohlcv_only" in rmse_summary[t]:
            met_ohl = dict(met_ohl or {})
            met_ohl["rmse_last5"] = rmse_summary[t]["ohlcv_only"]

    # Show the metric tiles (RMSE + weights)
    _metric_tiles(met_all or {}, "All inputs")
    _metric_tiles(met_ohl or {}, "OHLCV only")

    # The three plots for each ticker (images generated offline)
    st.subheader("a) OHLCV vs Actual")
    _show_image_anywhere(t, IMG_FILES["ohlcv_only"], "OHLCV vs Actual (Last 5 trading days)")

    st.subheader("b) All inputs vs Actual")
    _show_image_anywhere(t, IMG_FILES["all_inputs"], "All inputs vs Actual (Last 5 trading days)")

    st.subheader("c) OHLCV vs All inputs vs Actual")
    _show_image_anywhere(t, IMG_FILES["compare"], "Compare: Actual vs OHLCV vs All (Last 5 trading days)")

# Final note to the user: we didn't run any training here
st.success("Done. Displaying precomputed artifacts — no training or inference is run here.")
