#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys, os, json
from typing import Optional, List
import streamlit as st
import pandas as pd

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

# ===== NEW: load close-price history (local → GitHub raw) =====
def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _read_parquet_from_bytes(b: bytes) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(BytesIO(b))
    except Exception:
        return None

def _read_csv_from_bytes(b: bytes) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(BytesIO(b))
    except Exception:
        return None

def _load_prices_df(ticker: str) -> Optional[pd.DataFrame]:
    """
    Try local first:
      data/<TICKER>/prices.parquet
      data/<TICKER>/prices.csv
      (repo-root mirrors too)
    If missing, try GitHub raw for the same paths.
    """
    local_candidates = [
        DATA_DIR / ticker / "prices.parquet",
        DATA_DIR / ticker / "prices.csv",
        REPO_ROOT / "data" / ticker / "prices.parquet",
        REPO_ROOT / "data" / ticker / "prices.csv",
    ]
    # Local
    for p in local_candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                src = f"local: {p}"
                break
            except Exception:
                continue
    else:
        df, src = None, None

    # Remote (GitHub raw)
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

    if df is None or df.empty:
        return None

    # Normalize
    if "ds" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "ds"})
    df["ds"] = _norm_ds(df["ds"])

    # Prefer raw close; fallback to adj_close
    price_col = "close" if "close" in df.columns else ("adj_close" if "adj_close" in df.columns else None)
    if price_col is None:
        return None

    df = df[["ds", price_col]].dropna().sort_values("ds")
    df = df.rename(columns={price_col: "Close"})
    df.attrs["source"] = src
    return df

# --------- UI: SBUX + PFE, three prediction plots + price history ----------
tickers = _discover_tickers()
for t in tickers:
    st.header(t)

    # NEW: Close price history
    prices = _load_prices_df(t)
    if prices is not None and not prices.empty:
        st.subheader("Close price — full history")
        st.line_chart(prices.set_index("ds")["Close"])
        if prices.attrs.get("source"):
            st.caption(f"source: {prices.attrs['source']}")
    else:
        st.info("Close price file not found (tried local repo and GitHub raw).")

    # Metrics & prediction plots (unchanged)
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
