#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys, os
import streamlit as st

# --- SETTINGS (you can tweak in sidebar too) ---
DEFAULT_REPO = "Poulami-Nandi/mm_ensemble_pred"
DEFAULT_BRANCH = "main"
TICKER = "PFE"
FILENAME = "actual_vs_pred_all_inputs.png"   # change to e.g. "actual_vs_pred_ohlcv_only.png" or "compare_all_vs_ohlcv.png"

st.set_page_config(page_title="PFE plot loader", layout="centered")
st.title("PFE — Single Plot Loader")

# Sidebar knobs
repo = st.sidebar.text_input("GitHub repo (owner/name)", value=DEFAULT_REPO)
branch = st.sidebar.text_input("Git branch", value=DEFAULT_BRANCH)
fname = st.sidebar.text_input("File name", value=FILENAME)
st.sidebar.caption("Try other options:\n- actual_vs_pred_ohlcv_only.png\n- compare_all_vs_ohlcv.png")

# --- Resolve repo root & local search paths ---
here = Path(__file__).resolve()
repo_root = here.parents[2]  # .../repo_root/src/mm_ensemble/show_one_plot_pfe.py
outputs_new = repo_root / "outputs" / "last5" / TICKER / fname
runs_legacy = repo_root / "data" / "runs" / "last5" / TICKER / fname

local_candidates = [outputs_new, runs_legacy]

# --- Helper: load local bytes if exists ---
def _load_local_bytes() -> bytes | None:
    for p in local_candidates:
        if p.exists():
            st.caption(f"Loaded locally: {p}")
            return p.read_bytes()
    return None

# --- Helper: fetch from GitHub raw (bytes) ---
def _fetch_raw_bytes() -> bytes | None:
    try:
        import requests
    except Exception:
        return None
    for base in (f"outputs/last5/{TICKER}", f"data/runs/last5/{TICKER}"):
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{base}/{fname}"
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and r.content:
                st.caption(f"Loaded from GitHub raw: {url}")
                return r.content
        except Exception:
            pass
    return None

# --- Try local first, then GitHub raw ---
img_bytes = _load_local_bytes()
if img_bytes is None:
    img_bytes = _fetch_raw_bytes()

# --- Show or explain what's missing ---
if img_bytes:
    st.image(BytesIO(img_bytes), caption=f"{TICKER} — {fname}", use_container_width=True)
else:
    st.error("Could not load the image from local paths or GitHub.")
    st.write("Local paths checked:")
    for p in local_candidates:
        st.code(str(p))
    st.write("GitHub raw tried:")
    st.code(f"https://raw.githubusercontent.com/{repo}/{branch}/outputs/last5/{TICKER}/{fname}")
    st.code(f"https://raw.githubusercontent.com/{repo}/{branch}/data/runs/last5/{TICKER}/{fname}")
    st.info("If your repo is private, raw URLs won’t work. Commit the images into the repo and ensure the paths above exist in the deployed environment.")
