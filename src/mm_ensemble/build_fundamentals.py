#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR

# ------------- helpers -------------

def _read_metadata(root: Path, ticker: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    mpath = root / ticker / "metadata.json"
    if mpath.exists():
        js = json.loads(mpath.read_text(encoding="utf-8"))
        start = pd.Timestamp(js.get("period_start"))
        end = pd.Timestamp(js.get("period_end"))
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"Bad metadata dates in {mpath}")
        return start, end
    # fallback: last 365 days
    end = pd.Timestamp.utcnow().floor("D")
    start = end - pd.Timedelta(days=365)
    return start, end

def _normalize_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _daily_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")

def _best_unit(units: Dict[str, list], prefer: List[str]) -> Optional[str]:
    """Pick a unit key present in `units` that best matches `prefer` list."""
    if not isinstance(units, dict):
        return None
    # exact first
    for p in prefer:
        if p in units: return p
    # fallback: fuzzy contains
    for p in prefer:
        for k in units.keys():
            if p.lower() in k.lower():
                return k
    # else choose the numerically longest/most common
    try:
        return max(units.keys(), key=lambda k: len(units[k]))
    except Exception:
        return None

def _extract_tag(facts: dict, ns: str, tag_candidates: List[str]) -> Optional[dict]:
    """Return the first tag dict found in the namespace (e.g., 'us-gaap')."""
    tree = facts.get("facts", {}).get(ns, {})
    for nm in tag_candidates:
        if nm in tree:
            return tree[nm]
    return None

def _flatten_numeric_series(tag_dict: dict, unit_pref: List[str]) -> pd.DataFrame:
    """
    Flatten a tag's 'units' entries into a DataFrame with columns:
      ds (period end), val, filed, form, fy, fp
    Only keep rows with no 'segment' (consolidated).
    """
    if not tag_dict or "units" not in tag_dict:
        return pd.DataFrame(columns=["ds","val","filed","form","fy","fp"])
    ukey = _best_unit(tag_dict["units"], unit_pref)
    if not ukey or ukey not in tag_dict["units"]:
        return pd.DataFrame(columns=["ds","val","filed","form","fy","fp"])
    rows = []
    for it in tag_dict["units"][ukey]:
        # skip segmented data
        if "segment" in it and it["segment"]:
            continue
        # use end date for both instant/duration
        ds = it.get("end") or it.get("date") or it.get("instant")
        if not ds:
            continue
        filed = it.get("filed")
        form = it.get("form")
        fy = it.get("fy")
        fp = it.get("fp")
        val = it.get("val")
        try:
            v = float(val)
        except Exception:
            continue
        rows.append({"ds": ds, "val": v, "filed": filed, "form": form, "fy": fy, "fp": fp})
    if not rows:
        return pd.DataFrame(columns=["ds","val","filed","form","fy","fp"])
    df = pd.DataFrame(rows)
    df["ds"] = _normalize_ds(df["ds"])
    # prefer 10-Q/10-K when duplicated on same date, else max filed
    form_pref = {"10-K": 2, "10-Q": 1}
    df["form_rank"] = df["form"].map(form_pref).fillna(0).astype(int)
    df["filed_ts"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.sort_values(["ds","form_rank","filed_ts"]).drop_duplicates(subset=["ds"], keep="last")
    return df[["ds","val"]].sort_values("ds")

def _ffill_to_daily(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, colname: str) -> pd.DataFrame:
    idx = _daily_index(start, end)
    base = pd.DataFrame({"ds": idx})
    out = base.merge(df.rename(columns={"val": colname}), on="ds", how="left").sort_values("ds")
    out[colname] = out[colname].astype(float).ffill()
    return out

def _ttm_from_quarterly(qdf: pd.DataFrame, col: str, periods: int = 4) -> pd.Series:
    """Assumes qdf[col] is non-daily (point-in-time per quarter) already forward-filled to daily."""
    # To avoid re-summing daily duplicates, downsample to quarter-ends, then re-expand.
    q = qdf.dropna(subset=[col]).copy()
    if q.empty:
        return pd.Series(index=qdf.index, dtype=float)
    # take last value each quarter on the daily grid
    q["quarter"] = q["ds"].dt.to_period("Q")
    q_end = q.groupby("quarter", as_index=False).agg({"ds":"max", col:"last"})
    q_end = q_end.set_index("ds").sort_index()
    q_end[f"ttm_{col}"] = q_end[col].rolling(periods, min_periods=1).sum()
    # reindex back to daily and ffill
    daily = pd.DataFrame(index=qdf["ds"])
    ttm = daily.join(q_end[[f"ttm_{col}"]], how="left")
    return ttm[f"ttm_{col}"].ffill()

# ------------- main transform -------------

GAAP = "us-gaap"
TAGS = {
    "revenue":        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income":     ["NetIncomeLoss"],
    "eps_diluted":    ["EarningsPerShareDiluted"],
    "shares_out":     ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
    "assets":         ["Assets"],
    "liabilities":    ["Liabilities"],
    "cash":           ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsPeriodIncreaseDecrease"],
    "op_cf":          ["NetCashProvidedByUsedInOperatingActivities"],
    "gross_profit":   ["GrossProfit"],
    "cost_of_revenue":["CostOfRevenue"],
}

UNIT_PREF = {
    "revenue": ["USD"],
    "net_income": ["USD"],
    "eps_diluted": ["USD/shares", "USD/share", "USD"],  # eps sometimes has odd unit strings
    "shares_out": ["shares"],
    "assets": ["USD"],
    "liabilities": ["USD"],
    "cash": ["USD"],
    "op_cf": ["USD"],
    "gross_profit": ["USD"],
    "cost_of_revenue": ["USD"],
}

def build_one_ticker(root: Path, ticker: str) -> Path:
    start, end = _read_metadata(root, ticker)
    fpath = root / ticker / "facts.json"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing facts.json for {ticker}: {fpath}")
    facts = json.loads(fpath.read_text(encoding="utf-8"))

    frames = []
    # extract each tag -> daily ffilled
    for key, candidates in TAGS.items():
        tag_dict = _extract_tag(facts, GAAP, candidates)
        df = _flatten_numeric_series(tag_dict, UNIT_PREF.get(key, ["USD"]))
        if df.empty:
            # make empty daily column
            empty = pd.DataFrame({"ds": _daily_index(start, end), key: np.nan})
            frames.append(empty)
            continue
        daily = _ffill_to_daily(df, start, end, key)
        frames.append(daily)

    # merge all daily cols
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="ds", how="outer")
    out = out.sort_values("ds").reset_index(drop=True)

    # Derived features
    # ttm sums for revenue, net income, eps_diluted (sum of last 4 qtrs)
    # Build quarter-end series by sampling the last known value each quarter from daily (approx.)
    for col in ["revenue", "net_income", "eps_diluted"]:
        ttm = _ttm_from_quarterly(out[["ds", col]].copy(), col, periods=4)
        out[f"ttm_{col}"] = ttm.values

    # ratios
    out["gross_margin"] = np.where(out["revenue"].abs() > 0, out["gross_profit"] / out["revenue"], np.nan)
    out["ocf_margin"]   = np.where(out["revenue"].abs() > 0, out["op_cf"] / out["revenue"], np.nan)
    out["debt_to_assets"] = np.where(out["assets"].abs() > 0, out["liabilities"] / out["assets"], np.nan)

    # write parquet
    out_dir = root / ticker
    out_path = out_dir / "fundamentals_daily.parquet"
    out.to_parquet(out_path, index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Root folder containing <TICKER>/facts.json & metadata.json")
    ap.add_argument("--tickers", nargs="+", default=["SBUX","PFE"])
    args = ap.parse_args()

    root = Path(args.root)
    written = []
    for t in args.tickers:
        p = build_one_ticker(root, t.upper())
        written.append(str(p))
    print("Wrote:", *written, sep="\n  ")

if __name__ == "__main__":
    main()