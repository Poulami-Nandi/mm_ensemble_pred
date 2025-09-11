#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from io import BytesIO
import sys, os, json
import streamlit as st

# --------- Config (change if you want a different plot) ----------
TICKER   = "PFE"
FILENAME = "actual_vs_pred_all_inputs.png"  # or: "actual_vs_pred_ohlcv_only.png", "compare_all_vs_ohlcv.png"

# --------- Page ----------
st.set_page_config(page_title=f"{TICKER} — single plot", layout="centered")
st.title(f"{TICKER} — Single Plot Loader")

# Sidebar controls
st.sidebar.header("Source & Options")
source = st.sidebar.radio("Image source", ["Local (repo files)", "GitHub Raw"], index=0)
repo_slug = st.sidebar.text_input("GitHub repo (owner/name)", value="Poulami-Nandi/mm_ensemble_pred")
branch    = st.sidebar.text_input("Branch", value="main")
gh_token  = st.sidebar.text_input("GitHub token (optional, for private repos)", value="", type="password")

# --------- Resolve repo root and candidate local paths ----------
here = Path(__file__).resolve()
# expected layout: repo_root/src/mm_ensemble/show_one_plot_pfe.py  => parents[2] is repo_root
repo_root = here.parents[2]
src_dir   = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Prefer shared paths (if present), otherwise use repo-root-relative defaults
try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:
    DATA_DIR    = repo_root / "data"
    OUTPUTS_DIR = repo_root / "outputs"

local_candidates = [
    OUTPUTS_DIR / "last5" / TICKER / FILENAME,          # new layout
    DATA_DIR    / "runs" / "last5" / TICKER / FILENAME, # legacy layout
    repo_root   / "outputs" / "last5" / TICKER / FILENAME,  # repo-relative mirrors
    repo_root   / "data" / "runs" / "last5" / TICKER / FILENAME,
]

# --------- Helpers ----------
def load_local_bytes() -> bytes | None:
    """Return file bytes from the first existing local path; else None."""
    for p in local_candidates:
        if p.exists():
            st.caption(f"Loaded locally: {p}")
            return p.read_bytes()
    return None

def fetch_raw_bytes() -> bytes | None:
    """Fetch bytes from GitHub raw. Works for public repos; private requires token."""
    try:
        import requests
    except Exception:
        st.error("The requests package is not available in this environment.")
        return None

    headers = {"User-Agent": "streamlit-mm-ensemble/1.0"}
    if gh_token.strip():
        headers["Authorization"] = f"Bearer {gh_token.strip()}"

    tried = []
    for base in (f"outputs/last5/{TICKER}", f"data/runs/last5/{TICKER}"):
        url = f"https://github.com/{repo_slug}/{branch}/{base}/{FILENAME}"
        tried.append(url)
        try:
            r = requests.get(url, timeout=20, headers=headers)
            if r.status_code == 200 and r.content:
                st.caption(f"Loaded from GitHub raw: {url}")
                return r.content
        except Exception as e:
            st.warning(f"Fetch error from {url}: {e}")
    st.session_state["_tried_urls"] = tried
    return None

# --------- Load & Display ----------
img_bytes = None
if source == "Local (repo files)":
    img_bytes = load_local_bytes()
else:
    # Try local first so local dev works without network; then remote
    img_bytes = load_local_bytes() or fetch_raw_bytes()

if img_bytes:
    st.image(BytesIO(img_bytes), caption=f"{TICKER} — {FILENAME}", use_container_width=True)
else:
    st.error("Could not load the image from local paths or GitHub.")
    st.write("Local paths checked:")
    for p in local_candidates:
        st.code(str(p))

    if source != "Local (repo files)":
        st.write("GitHub raw tried:")
        for u in (st.session_state.get("_tried_urls") or []):
            st.code(u)

    st.info(
        "If the repo is private, pass a Personal Access Token (classic) with 'read:packages'/'repo' "
        "scope via the sidebar. Also ensure the file is actually committed to the repo path above "
        "(not .gitignored) and the filename matches exactly (case-sensitive)."
    )
