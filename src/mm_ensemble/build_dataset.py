#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
# Module: mm_ensemble/build_dataset.py
# Overview:
#   Builds modeling dataset by merging features/targets from raw inputs and saving train/val/test splits.
# Notes:
#   - This file has been annotated with verbose comments for clarity.
#   - Logic is unchanged; only comments were added.
##############################################################################


"""
Build a modeling dataset by merging:
  - prices.parquet
  - trends.parquet
  - gdelt_daily.csv
  - fundamentals_daily.parquet

Outputs per ticker:
  data/<TICKER>/dataset.parquet
  data/<TICKER>/train.parquet
  data/<TICKER>/val.parquet
  data/<TICKER>/test.parquet

Features (examples):
  - Price-based: returns, log-returns, moving averages, rolling volatility
  - Google Trends: raw, moving averages, rolling z-score
  - News: log1p counts, 7D avg counts, 7D avg tone
  - Fundamentals (daily forward-filled): revenue, EPS, margins, TTM sums, ratios

Targets:
  - target_return_1d = pct change of (adj_close or close) next 1 trading day
  - target_return_5d = pct change next 5 trading days
  - target_up_1d = 1 if target_return_1d > 0 else 0

Time-safe splits (no shuffling):
  - mode "frac": train_frac + val_frac (test = remainder)
  - mode "date": train_end_date, val_end_date cutoffs

Usage examples:
  python build_dataset.py --root data --tickers SBUX PFE
  python build_dataset.py --root data --tickers SBUX --split-mode frac --train-frac 0.7 --val-frac 0.15
  python build_dataset.py --split-mode date --train-end 2025-04-01 --val-end 2025-07-01
"""

# Imports: stdlib, scientific stack, and optional ML/TS libs
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR

# ------------------------------- cfg -------------------------------


# --- Dataclass ---
# Configuration or typed records used across stages.
@dataclass

# --- Class `BuildCfg` ---
# Purpose: Data structure or component in the pipeline.
class BuildCfg:
    root: Path
    tickers: List[str]
    split_mode: str = "frac"  # 'frac' or 'date'
    train_frac: float = 0.7
    val_frac: float = 0.15
    train_end: Optional[str] = None
    val_end: Optional[str] = None
    price_col_pref: str = "auto"  # 'auto'|'adj_close'|'close'
    min_rows: int = 120  # sanity threshold before splitting

# --------------------------- time helpers --------------------------

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _lead(series: pd.Series, k: int) -> pd.Series:
    return series.shift(-k)

def _safe_price_column(df: pd.DataFrame, pref: str) -> str:
    cols = [c.lower() for c in df.columns]
    if pref == "adj_close" and "adj_close" in cols: return "adj_close"
    if pref == "close" and "close" in cols: return "close"
    if "adj_close" in cols: return "adj_close"
    if "close" in cols: return "close"
    raise KeyError("Expected 'adj_close' or 'close' in prices dataframe")

# ------------------------ feature engineering ----------------------

def _price_features(prices: pd.DataFrame, price_pref: str) -> pd.DataFrame:
    df = prices.copy()
    df["ds"] = _norm_ds(df["ds"])
    df = df.sort_values("ds").drop_duplicates(subset=["ds"])
    px_col = _safe_price_column(df, price_pref)

    # returns
    df["ret_1d"] = df[px_col].pct_change()
    df["logret_1d"] = np.log(df[px_col]).diff()
    df["ret_5d"] = df[px_col].pct_change(5)

    # rolling features
    for w in (5, 10, 20, 50):
        df[f"ma_{w}"] = df[px_col].rolling(w, min_periods=1).mean()
    df["vol_10"] = df["logret_1d"].rolling(10, min_periods=2).std()
    df["ma_5_over_20"] = df["ma_5"] / df["ma_20"]
    df["ma_20_over_50"] = df["ma_20"] / df["ma_50"]

    # targets (forward returns)
    df["target_return_1d"] = _lead(df[px_col], 1) / df[px_col] - 1.0
    df["target_return_5d"] = _lead(df[px_col], 5) / df[px_col] - 1.0
    df["target_up_1d"] = (df["target_return_1d"] > 0).astype("Int64")

    return df

def _trends_features(trends: pd.DataFrame) -> pd.DataFrame:
    if trends is None or trends.empty:
        return pd.DataFrame(columns=["ds","gt","gt_ma_7","gt_ma_28","gt_z_90"])
    df = trends.copy()
    df["ds"] = _norm_ds(df["ds"])
    if "gt" not in df.columns:
        # allow alternate name
        if "google_trend" in df.columns:
            df = df.rename(columns={"google_trend": "gt"})
        else:
            df["gt"] = np.nan
    df = df.sort_values("ds").drop_duplicates(subset=["ds"])
    # fill occasional gaps from weekly sampling
    df["gt"] = df["gt"].astype(float)
    df["gt"] = df["gt"].ffill()

    df["gt_ma_7"] = df["gt"].rolling(7, min_periods=1).mean()
    df["gt_ma_28"] = df["gt"].rolling(28, min_periods=1).mean()
    m = df["gt"].rolling(90, min_periods=10).mean()
    s = df["gt"].rolling(90, min_periods=10).std()
    df["gt_z_90"] = (df["gt"] - m) / s.replace(0, np.nan)
    return df[["ds","gt","gt_ma_7","gt_ma_28","gt_z_90"]]

def _news_features(news: pd.DataFrame) -> pd.DataFrame:
    if news is None or news.empty:
        return pd.DataFrame(columns=["ds","news_count","news_log1p","news_cnt_ma_7","tone_ma_7"])
    df = news.copy()
    df["ds"] = _norm_ds(df["ds"])
    for c in ("news_count","avg_tone"):
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values("ds").drop_duplicates(subset=["ds"])
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(int)
    df["avg_tone"] = pd.to_numeric(df["avg_tone"], errors="coerce")
    df["news_log1p"] = np.log1p(df["news_count"])
    df["news_cnt_ma_7"] = df["news_count"].rolling(7, min_periods=1).mean()
    df["tone_ma_7"] = df["avg_tone"].rolling(7, min_periods=1).mean()
    return df[["ds","news_count","news_log1p","news_cnt_ma_7","tone_ma_7"]]

def _load_optional_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)

def _load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)

# ------------------------------ builder ------------------------------

def build_one(cfg: BuildCfg, ticker: str) -> Tuple[Path, Path, Path, Path]:
    tdir = cfg.root / ticker
    prices = pd.read_parquet(tdir / "prices.parquet")
    trends = _load_optional_parquet(tdir / "trends.parquet")
    news   = _load_optional_csv(tdir / "gdelt_daily.csv")
    funda  = _load_optional_parquet(tdir / "fundamentals_daily.parquet")

    # base: price features + targets
    p = _price_features(prices, cfg.price_col_pref)

    # trends/news features
    tr = _trends_features(trends) if trends is not None else _trends_features(pd.DataFrame())
    nw = _news_features(news) if news is not None else _news_features(pd.DataFrame())

    # merge on daily trading days (left on prices)
    df = p.merge(tr, on="ds", how="left").merge(nw, on="ds", how="left")

    # fundamentals (already daily + ffilled by your builder)
    if funda is not None and not funda.empty:
        funda = funda.copy()
        funda["ds"] = _norm_ds(funda["ds"])
        # Ensure numeric
        for c in funda.columns:
            if c != "ds":
                funda[c] = pd.to_numeric(funda[c], errors="coerce")
        df = df.merge(funda, on="ds", how="left")

    # final ordering & cleaning
    df = df.sort_values("ds").reset_index(drop=True)

    # rows usable for training (need non-null targets)
    ready = df.dropna(subset=["target_return_1d", "target_return_5d"]).copy()

    # sanity check before splitting
    if len(ready) < cfg.min_rows:
        raise RuntimeError(f"{ticker}: not enough rows ({len(ready)}) to split; reduce horizon or min_rows")

    # time-safe split
    if cfg.split_mode == "frac":
        n = len(ready)
        n_train = int(n * cfg.train_frac)
        n_val   = int(n * (cfg.train_frac + cfg.val_frac))
        train = ready.iloc[:n_train]
        val   = ready.iloc[n_train:n_val]
        test  = ready.iloc[n_val:]
    else:
        # date mode
        if not cfg.train_end or not cfg.val_end:
            raise ValueError("split-mode 'date' requires --train-end and --val-end (YYYY-MM-DD)")
        train = ready.loc[ready["ds"] <= pd.Timestamp(cfg.train_end)]
        val   = ready.loc[(ready["ds"] > pd.Timestamp(cfg.train_end)) & (ready["ds"] <= pd.Timestamp(cfg.val_end))]
        test  = ready.loc[ready["ds"] > pd.Timestamp(cfg.val_end)]
        if train.empty or val.empty or test.empty:
            raise RuntimeError("Date splits produced an empty split; check --train-end/--val-end")

    # write
    out_full = tdir / "dataset.parquet"
    out_train = tdir / "train.parquet"
    out_val = tdir / "val.parquet"
    out_test = tdir / "test.parquet"

    df.to_parquet(out_full, index=False)
    train.to_parquet(out_train, index=False)
    val.to_parquet(out_val, index=False)
    test.to_parquet(out_test, index=False)

    return out_full, out_train, out_val, out_test

# ------------------------------- cli -------------------------------


# --- Function `main()` ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root data folder")
    ap.add_argument("--tickers", nargs="+", default=["SBUX","PFE"])
    ap.add_argument("--split-mode", choices=["frac","date"], default="frac")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--train-end", type=str, default=None, help="YYYY-MM-DD (date split)")
    ap.add_argument("--val-end", type=str, default=None, help="YYYY-MM-DD (date split)")
    ap.add_argument("--price-col-pref", choices=["auto","adj_close","close"], default="auto")
    ap.add_argument("--min-rows", type=int, default=120)
    args = ap.parse_args()

    cfg = BuildCfg(

# Configuration / constants / paths
        root=Path(args.root),
        tickers=[t.upper() for t in args.tickers],
        split_mode=args.split_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        train_end=args.train_end,
        val_end=args.val_end,
        price_col_pref=args.price_col_pref,
        min_rows=args.min_rows,
    )

    written = []
    for t in cfg.tickers:
        (out_full, out_train, out_val, out_test) = build_one(cfg, t)
        written += [str(out_full), str(out_train), str(out_val), str(out_test)]

    print("Wrote:")
    for w in written:
        print(" ", w)


# Entrypoint: parse CLI args and run main routine.
if __name__ == "__main__":
    main()
