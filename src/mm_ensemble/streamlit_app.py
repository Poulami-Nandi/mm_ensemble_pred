#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ── Robust bootstrap so the app runs even if the package isn't installed ──
import sys
from pathlib import Path

def _ensure_pkg_on_path() -> None:
    """
    Make sure 'mm_ensemble' is importable by adding the correct folder to sys.path.
    Works for these common layouts:
      - repo_root/src/mm_ensemble/...
      - repo_root/mm_ensemble/...  (plain package without 'src')
    """
    here = Path(__file__).resolve()

    # Build a set of candidate directories to add to sys.path
    cand_paths = []
    parents = [here.parent] + list(here.parents)
    for p in parents:
        # Case A: the parent itself is the 'src' folder
        if (p / "mm_ensemble" / "utils" / "paths.py").exists():
            cand_paths.append(p)
        # Case B: there's a 'src' under this parent
        if (p / "src" / "mm_ensemble" / "utils" / "paths.py").exists():
            cand_paths.append(p / "src")

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in cand_paths:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    # Try adding each candidate and import
    for c in uniq:
        s = str(c)
        if s not in sys.path:
            sys.path.insert(0, s)
        try:
            import mm_ensemble  # noqa: F401
            return
        except Exception:
            continue

    # Final hint if nothing worked
    raise ModuleNotFoundError(
        "Could not import 'mm_ensemble'. "
        "Either install the repo (pip install -e .) or ensure PYTHONPATH includes the repo's 'src' folder."
    )

try:
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except ModuleNotFoundError:
    _ensure_pkg_on_path()
    from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore

# ── App starts here ──
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json

# Reuse your training/feature logic
try:
    import mm_ensemble.last5_ensemble_plots as lp
except ModuleNotFoundError:
    _ensure_pkg_on_path()
    import mm_ensemble.last5_ensemble_plots as lp

st.set_page_config(page_title="mm_ensemble — last-5 demo", layout="wide")

# ---------------------- helpers ----------------------
def _price_col(df: pd.DataFrame, pref: str = "auto") -> str:
    cand = [c.lower() for c in df.columns]
    if pref == "adj_close" and "adj_close" in cand:
        return "adj_close"
    if pref == "close" and "close" in cand:
        return "close"
    return "adj_close" if "adj_close" in cand else ("close" if "close" in cand else cand[0])

def _to_price_series(df: pd.DataFrame, dates, ret_pred, price_col_pref="auto"):
    if len(dates) == 0:
        return np.array([]), np.array([]), "close", float("nan")
    df2 = df.copy()
    df2["ds"] = pd.to_datetime(df2["ds"], errors="coerce")
    pcol = _price_col(df2, price_col_pref)
    mask_last5 = df2["ds"].isin(pd.to_datetime(dates))
    actual_price = pd.to_numeric(df2.loc[mask_last5, pcol], errors="coerce").values
    d0 = pd.to_datetime(dates[0])
    prev_mask = df2["ds"] < d0
    if not prev_mask.any():
        base = float(pd.to_numeric(df2[pcol], errors="coerce").dropna().iloc[0])
    else:
        base = float(pd.to_numeric(df2.loc[prev_mask, pcol], errors="coerce").dropna().iloc[-1])
    ret_pred = np.asarray(ret_pred, dtype=float)
    pred_price = np.zeros_like(ret_pred, dtype=float)
    p = base
    for i, r in enumerate(ret_pred):
        r = 0.0 if not math.isfinite(r) else r
        p = p * (1.0 + r)
        pred_price[i] = p
    return actual_price, pred_price, pcol, base

def _plot_series(dates, series_list, labels, title, y_label="Price", footer=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for s, lbl in zip(series_list, labels):
        ax.plot(dates, s, marker="o", linewidth=1.8, label=lbl)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    if footer:
        ax.text(0.0, -0.25, footer, transform=ax.transAxes, fontsize=9, alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

def _rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any(): return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

# ---------------------- sidebar ----------------------
st.sidebar.title("mm_ensemble — config")

# discover tickers in data/
available = []
if DATA_DIR.exists():
    for p in DATA_DIR.iterdir():
        if p.is_dir():
            available.append(p.name)
available = sorted(available) or ["SBUX", "PFE"]

tickers = st.sidebar.multiselect("Tickers", options=available, default=available)
feature_mode = st.sidebar.selectbox("Feature set", ["Both (compare)", "All inputs", "OHLCV only"], index=0)
target = st.sidebar.text_input("Target column", value="target_return_1d")
holdout_days = st.sidebar.number_input("Holdout days", min_value=3, max_value=10, value=5, step=1)
price_pref = st.sidebar.selectbox("Price column preference", ["auto", "adj_close", "close"], index=0)
plot_kind = st.sidebar.selectbox("Plot as", ["Price", "Return"], index=0)
save_artifacts = st.sidebar.checkbox("Save artifacts into outputs/last5", value=True)

lp.HOLDOUT_DAYS = int(holdout_days)

run_btn = st.sidebar.button("Run last-5 predictions")

st.title("Multimodal Ensemble — Last 5 Trading Days")
st.caption("Uses pre-saved inputs in `data/` and your ensemble (XGB+ARIMA with auto-weights). No downloads.")

# ---------------------- main ----------------------
if run_btn:
    out_base = OUTPUTS_DIR / "last5"
    out_base.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        st.subheader(f"Ticker: {t}")

        try:
            df = lp._load_dataset(DATA_DIR, t)
        except Exception as e:
            st.error(f"Failed to load dataset for {t}: {e}")
            continue

        feats_all   = lp._select_all_features(df, target)
        feats_ohlcv = lp._select_ohlcv_features(df, target)

        res_all = res_ohl = None
        if feature_mode in ("Both (compare)", "All inputs"):
            res_all = lp.train_and_predict_last5(df, feats_all, target)
        if feature_mode in ("Both (compare)", "OHLCV only"):
            res_ohl = lp.train_and_predict_last5(df, feats_ohlcv, target)

        def render_block(name, res):
            if res is None:
                return
            dates = pd.to_datetime(res["dates_last5"]).date
            y_true = res["y_true_last5"]
            y_pred = res["y_ens_last5"]
            w_xgb, w_ar = res["w_xgb"], res["w_arima"]

            if plot_kind == "Price":
                act_p, pred_p, pcol, base = _to_price_series(df, res["dates_last5"], y_pred, price_pref)
                rmse_price = _rmse(act_p, pred_p)
                title = f"{t} — {name} — Price (RMSE={rmse_price:.4f})"
                footer = f"Weights: XGB={w_xgb:.3f}, ARIMA={w_ar:.3f} • Baseline {pcol}={base:.2f}"
                _plot_series(dates, [act_p, pred_p], ["Actual", "Predicted"], title, y_label="Price", footer=footer)
                tbl = pd.DataFrame({
                    "ds": pd.to_datetime(res["dates_last5"]),
                    "actual_price": act_p,
                    "pred_price": pred_p,
                    "pred_return": y_pred,
                    "true_return": y_true,
                })
                st.dataframe(
                    tbl.style.format({"actual_price":"{:.2f}", "pred_price":"{:.2f}",
                                      "pred_return":"{:.6f}", "true_return":"{:.6f}"}),
                    use_container_width=True
                )
                if save_artifacts:
                    tdir = out_base / t
                    tdir.mkdir(parents=True, exist_ok=True)
                    suffix = "all_inputs" if "All" in name else "ohlcv_only"
                    tbl.to_csv(tdir / f"pred_last5_{suffix}_PRICE.csv", index=False)
                    (tdir / f"metrics_{suffix}.json").write_text(json.dumps({
                        "rmse_price": rmse_price,
                        "weights": {"w_xgb": float(w_xgb), "w_arima": float(w_ar)},
                        "feats_count": int(res["feats_count"]),
                        "train_len": int(res["train_len"]), "val_len": int(res["val_len"]), "test_len": int(res["test_len"])
                    }, indent=2), encoding="utf-8")

            else:
                rmse_ret = _rmse(y_true, y_pred)
                title = f"{t} — {name} — Return (RMSE={rmse_ret:.6f})"
                footer = f"Weights: XGB={w_xgb:.3f}, ARIMA={w_ar:.3f}"
                _plot_series(dates, [y_true, y_pred], ["Actual", "Predicted"], title, y_label="Daily Return", footer=footer)
                tbl = pd.DataFrame({
                    "ds": pd.to_datetime(res["dates_last5"]),
                    "true_return": y_true,
                    "pred_return": y_pred,
                })
                st.dataframe(tbl.style.format({"true_return":"{:.6f}", "pred_return":"{:.6f}"}), use_container_width=True)
                if save_artifacts:
                    tdir = out_base / t
                    tdir.mkdir(parents=True, exist_ok=True)
                    suffix = "all_inputs" if "All" in name else "ohlcv_only"
                    tbl.to_csv(tdir / f"pred_last5_{suffix}.csv", index=False)
                    (tdir / f"metrics_{suffix}.json").write_text(json.dumps({
                        "rmse_return": rmse_ret,
                        "weights": {"w_xgb": float(w_xgb), "w_arima": float(w_ar)},
                        "feats_count": int(res["feats_count"]),
                        "train_len": int(res["train_len"]), "val_len": int(res["val_len"]), "test_len": int(res["test_len"])
                    }, indent=2), encoding="utf-8")

        if feature_mode in ("All inputs", "Both (compare)"):
            st.markdown("### All inputs (ensemble)")
            render_block("All inputs", res_all)

        if feature_mode in ("OHLCV only", "Both (compare)"):
            st.markdown("### OHLCV only (ensemble)")
            render_block("OHLCV only", res_ohl)

        if feature_mode == "Both (compare)" and (res_all is not None) and (res_ohl is not None):
            st.markdown("### Compare (Actual vs OHLCV vs All)")
            dates = pd.to_datetime(res_all["dates_last5"]).date
            if plot_kind == "Price":
                act_p, pred_all_p, pcol, base = _to_price_series(df, res_all["dates_last5"], res_all["y_ens_last5"], price_pref)
                _,    pred_ohl_p, _, _        = _to_price_series(df, res_ohl["dates_last5"], res_ohl["y_ens_last5"], price_pref)
                rmse_all = _rmse(act_p, pred_all_p)
                rmse_ohl = _rmse(act_p, pred_ohl_p)
                title = f"{t} — Compare (Price) — All (RMSE={rmse_all:.4f}) vs OHLCV (RMSE={rmse_ohl:.4f})"
                footer = (f"Weights — All: XGB={res_all['w_xgb']:.3f}, ARIMA={res_all['w_arima']:.3f} • "
                          f"OHLCV: XGB={res_ohl['w_xgb']:.3f}, ARIMA={res_ohl['w_arima']:.3f}")
                _plot_series(dates, [act_p, pred_ohl_p, pred_all_p],
                             ["Actual", "Pred (OHLCV)", "Pred (All inputs)"], title, y_label="Price", footer=footer)
            else:
                y_true = res_all["y_true_last5"]
                y_all  = res_all["y_ens_last5"]
                y_ohl  = res_ohl["y_ens_last5"]
                rmse_all = _rmse(y_true, y_all)
                rmse_ohl = _rmse(y_true, y_ohl)
                title = f"{t} — Compare (Return) — All (RMSE={rmse_all:.6f}) vs OHLCV (RMSE={rmse_ohl:.6f})"
                footer = (f"Weights — All: XGB={res_all['w_xgb']:.3f}, ARIMA={res_all['w_arima']:.3f} • "
                          f"OHLCV: XGB={res_ohl['w_xgb']:.3f}, ARIMA={res_ohl['w_arima']:.3f}")
                _plot_series(dates, [y_true, y_ohl, y_all],
                             ["Actual", "Pred (OHLCV)", "Pred (All inputs)"], title, y_label="Daily Return", footer=footer)
        st.divider()
else:
    st.info("Configure options in the sidebar and click **Run last-5 predictions**.")
