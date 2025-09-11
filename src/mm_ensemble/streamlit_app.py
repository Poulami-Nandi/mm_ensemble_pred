#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys, os
import streamlit as st

# ---- CONFIG: exact file you showed ----
REPO_SLUG = "Poulami-Nandi/mm_ensemble_pred"
BRANCH    = "main"
TICKER    = "PFE"
FILENAME  = "compare_all_vs_ohlcv_PRICE.png"  # <- note the _PRICE suffix

st.set_page_config(page_title=f"{TICKER} plot loader", layout="centered")
st.title(f"{TICKER} — Show a single plot (local → GitHub raw)")

# Try local file first (repo checkout)
here = Path(__file__).resolve()
repo_root = here.parents[2]  # repo_root/src/mm_ensemble/show_one_plot_pfe.py
local_path = repo_root / "outputs" / "last5" / TICKER / FILENAME

img_bytes = None
if local_path.exists():
    st.caption(f"Loaded locally: {local_path}")
    img_bytes = local_path.read_bytes()
else:
    # Remote raw URL form (NOT /blob/)
    raw_url = f"https://raw.githubusercontent.com/{REPO_SLUG}/{BRANCH}/outputs/last5/{TICKER}/{FILENAME}"
    st.caption(f"Loading from GitHub raw: {raw_url}")
    try:
        import requests
        r = requests.get(raw_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            img_bytes = r.content
        else:
            st.error(f"GitHub raw returned {r.status_code} for:\n{raw_url}")
    except Exception as e:
        st.error(f"Request failed: {e}")

if img_bytes:
    st.image(BytesIO(img_bytes), caption=f"{TICKER} — {FILENAME}", use_container_width=True)
else:
    st.error("Could not load the image from local path or GitHub raw.")
    st.write("Checked local path:")
    st.code(str(local_path))
    st.write("GitHub raw URL tried:")
    st.code(f"https://raw.githubusercontent.com/{REPO_SLUG}/{BRANCH}/outputs/last5/{TICKER}/{FILENAME}")
