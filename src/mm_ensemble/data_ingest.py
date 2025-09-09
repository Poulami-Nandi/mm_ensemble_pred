#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone data ingestion for two demo tickers (SBUX, PFE) for the last 1 year.

Streams (free, programmatic):
  - OHLCV via yfinance
  - Google Trends via pytrends (chunked windows + retry/backoff)
  - News counts + average tone via GDELT 2.1 Doc API
  - Fundamentals via SEC EDGAR Company Facts (XBRL JSON)

Outputs are saved under ./data/<TICKER>/ as parquet/csv/json files.

Examples:
  python data_ingest.py --email YOU@domain.com --save prices
  python data_ingest.py --email YOU@domain.com --save trends --geo_trends US
  python data_ingest.py --email YOU@domain.com --save news
  python data_ingest.py --email YOU@domain.com --save facts
  python data_ingest.py --email YOU@domain.com --save all --skip-existing
"""
from pathlib import Path
if Path("data/.freeze_inputs").exists():
    print("Inputs are frozen at data/.freeze_inputs — skipping any download.")
    raise SystemExit(0)

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import json
import time
import random
import re

# --------------------- Lazy imports for optional deps ---------------------

def _lazy_import_yf():
    import importlib
    return importlib.import_module("yfinance")

def _lazy_import_pytrends():
    import importlib
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR
    return importlib.import_module("pytrends.request"), importlib.import_module("pytrends.exceptions")

# ------------------------------- Config ----------------------------------

@dataclass
class IngestConfig:
    tickers: List[str]
    email: str
    out_dir: str = "data"
    lookback_days: int = 365
    geo_trends: str = ""              # '' for worldwide, 'US' for US-only
    trends_window_days: int = 90
    trends_base_pause: float = 2.0
    trends_max_retries: int = 5
    save: str = "all"                 # one of: prices, trends, news, facts, all
    skip_existing: bool = False

# ------------------------- Utilities & IO -------------------------

def daterange_utc(days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.utcnow().floor("D")
    start = end - pd.Timedelta(days=days)
    return start, end

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Path, skip_existing: bool = False):
    ensure_dir(path.parent)
    if skip_existing and path.exists():
        return
    df.to_parquet(path, index=False)

def save_csv(df: pd.DataFrame, path: Path, skip_existing: bool = False):
    ensure_dir(path.parent)
    if skip_existing and path.exists():
        return
    df.to_csv(path, index=False)

def write_text(txt: str, path: Path, skip_existing: bool = False):
    ensure_dir(path.parent)
    if skip_existing and path.exists():
        return
    path.write_text(txt, encoding="utf-8")

def to_daily_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")

def normalize_ds(series: pd.Series) -> pd.Series:
    """Normalize any datetime series to naive (no tz) daily timestamps."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    s = s.dt.tz_convert(None).dt.floor("D")
    return s

# ------------------------ Robust column normalizer ------------------------

def _flatten_col(c) -> str:
    """Flatten yfinance tuples and normalize."""
    if isinstance(c, tuple):
        parts = [str(x) for x in c if x is not None and str(x) != ""]
        s = "_".join(parts)
    else:
        s = str(c)
    s = s.strip().lower().replace(" ", "_")
    return s

def _pick_col(df_cols: List[str], base: str) -> str:
    """
    Pick a column for a given base name, accepting:
      - exact: base
      - suffix: f"{something}_{base}"
      - prefix: f"{base}_{something}"
    'adj_close' also accepts 'adjclose' variants.
    """
    cols = df_cols
    candidates = []

    def add_if(pred):
        for c in cols:
            if pred(c):
                candidates.append(c)

    add_if(lambda c: c == base)                    # exact
    add_if(lambda c: c.endswith("_" + base))       # suffix
    add_if(lambda c: c.startswith(base + "_"))     # prefix

    if base == "adj_close":
        add_if(lambda c: c == "adjclose")
        add_if(lambda c: c.endswith("_adjclose"))
        add_if(lambda c: c.startswith("adjclose_"))

    seen, dedup = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            dedup.append(c)

    if not dedup:
        raise KeyError(f"Missing '{base}' column. Available columns: {cols}")
    return dedup[0]

# ---------------------------- OHLCV (yfinance) ----------------------------

def fetch_ohlcv(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    yf = _lazy_import_yf()
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,   # keep Adj Close
        progress=False
    )
    if df is None or len(df) == 0:
        raise ValueError(f"yfinance returned no rows for {ticker}")

    df = df.reset_index()
    df.columns = [_flatten_col(c) for c in df.columns]

    if "date" in df.columns and "ds" not in df.columns:
        df = df.rename(columns={"date": "ds"})
    if "ds" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "ds"})
    df["ds"] = normalize_ds(df["ds"])

    cols = list(df.columns)
    open_c  = _pick_col(cols, "open")
    high_c  = _pick_col(cols, "high")
    low_c   = _pick_col(cols, "low")
    close_c = _pick_col(cols, "close")
    adj_c   = _pick_col(cols, "adj_close")
    vol_c   = _pick_col(cols, "volume")

    out = df[["ds", open_c, high_c, low_c, close_c, adj_c, vol_c]].copy()
    out.columns = ["ds", "open", "high", "low", "close", "adj_close", "volume"]
    return out

# ---------------------- Google Trends (pytrends) ----------------------

def _pytrends_chunk(keyword: str, geo: str, start: pd.Timestamp, end: pd.Timestamp,
                    base_pause: float, max_retries: int) -> pd.DataFrame:
    """Fetch one window with retry/backoff to avoid 429."""
    (TrendReqMod, PTEx) = _lazy_import_pytrends()
    TrendReq = TrendReqMod.TrendReq

    attempt = 0
    while True:
        try:
            pytrends = TrendReq(hl="en-US", tz=0)
            timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
            pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                return pd.DataFrame(columns=["ds", "gt"])
            df = df.reset_index().rename(columns={"date": "ds", keyword: "gt"})
            df["ds"] = normalize_ds(df["ds"])
            return df[["ds", "gt"]]
        except PTEx.TooManyRequestsError:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(base_pause * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
        except requests.RequestException:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(base_pause * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))

def fetch_trends_chunked(keyword: str, geo: str, start: pd.Timestamp, end: pd.Timestamp,
                         window_days: int, base_pause: float, max_retries: int) -> pd.DataFrame:
    """Pull Trends in chunks (e.g., 90-day windows) and merge."""
    pieces, s = [], start
    while s <= end:
        e = min(s + pd.Timedelta(days=window_days - 1), end)
        df = _pytrends_chunk(keyword, geo, s, e, base_pause, max_retries)
        pieces.append(df)
        time.sleep(base_pause + random.uniform(0.0, 0.3))  # polite pause
        s = e + pd.Timedelta(days=1)

    if not pieces:
        idx = to_daily_index(start, end)
        return pd.DataFrame({"ds": idx, "gt": np.nan})

    all_df = pd.concat(pieces, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["ds"], keep="last").sort_values("ds")

    idx = to_daily_index(start, end)
    left = pd.DataFrame({"ds": idx})
    left["ds"] = normalize_ds(left["ds"])
    all_df["ds"] = normalize_ds(all_df["ds"])
    out = left.merge(all_df, on="ds", how="left")
    return out

# -------------------------- GDELT (Doc API) --------------------------

def gdelt_query(keyword: str, timespan_days: int) -> List[Dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": keyword,
        "format": "json",
        "maxrecords": 250,
        "sort": "DateDesc",
        "timespan": f"{timespan_days}d",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])

def gdelt_daily_counts_and_tone(keyword: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = (end - start).days + 1
    arts = gdelt_query(keyword, days)

    idx = to_daily_index(start, end)
    idx_df = pd.DataFrame({"ds": idx})
    idx_df["ds"] = normalize_ds(idx_df["ds"])

    if not arts:
        return pd.DataFrame({"ds": idx_df["ds"], "news_count": 0, "avg_tone": np.nan})

    df = pd.DataFrame(arts)

    # tone as Series (not float) when missing
    if "tone" in df.columns:
        df["tone"] = pd.to_numeric(df["tone"], errors="coerce").fillna(0.0)
    else:
        df["tone"] = pd.Series(0.0, index=df.index)

    # Parse date, normalize
    if "seendate" in df.columns:
        ds = pd.to_datetime(df["seendate"], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    elif "publishedAt" in df.columns:
        ds = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    else:
        ds = pd.NaT
    df["ds"] = normalize_ds(ds)

    df = df.dropna(subset=["ds"])
    if df.empty:
        return pd.DataFrame({"ds": idx_df["ds"], "news_count": 0, "avg_tone": np.nan})

    # Some payloads lack 'url'; count rows instead if absent
    if "url" in df.columns:
        agg = df.groupby("ds").agg(news_count=("url", "count"), avg_tone=("tone", "mean")).reset_index()
    else:
        agg = df.groupby("ds").agg(news_count=("tone", "size"),  avg_tone=("tone", "mean")).reset_index()

    out = idx_df.merge(agg, on="ds", how="left")
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["avg_tone"]   = out["avg_tone"].astype(float)
    return out

# ------------------- SEC EDGAR (Company Facts JSON) -------------------

def sec_headers(email: str) -> Dict[str, str]:
    return {
        "User-Agent": f"DataIngest/1.0 (+{email})",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def _http_get_json(url: str, email: str, timeout: int = 30) -> Optional[Any]:
    r = requests.get(url, headers=sec_headers(email), timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None

def load_sec_ticker_map(email: str) -> pd.DataFrame:
    """
    Robust ticker→CIK map with multiple fallbacks:
      1) company_tickers.json  (classic dict of dicts)
      2) company_tickers_exchange.json (list of dicts)
      3) include/ticker.txt (plain text 'ticker|cik')
    """
    # 1) classic JSON
    js = _http_get_json("https://www.sec.gov/files/company_tickers.json", email)
    if isinstance(js, dict) and js:
        rows = []
        for _, v in js.items():
            if "ticker" in v and ("cik_str" in v or "cik" in v):
                cik = str(v.get("cik_str", v.get("cik"))).strip()
                rows.append({"ticker": v["ticker"].upper(), "title": v.get("title", ""), "cik": cik})
        if rows:
            df = pd.DataFrame(rows)
            df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
            return df[["ticker", "title", "cik"]]

    # 2) exchange JSON
    js2 = _http_get_json("https://www.sec.gov/files/company_tickers_exchange.json", email)
    if isinstance(js2, list) and js2:
        rows = []
        for v in js2:
            if "ticker" in v and ("cik_str" in v or "cik" in v):
                cik = str(v.get("cik_str", v.get("cik"))).strip()
                rows.append({"ticker": v["ticker"].upper(), "title": v.get("title", ""), "cik": cik})
        if rows:
            df = pd.DataFrame(rows)
            df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
            return df[["ticker", "title", "cik"]]

    # 3) plain text fallback
    r = requests.get("https://www.sec.gov/include/ticker.txt", headers=sec_headers(email), timeout=30)
    if r.status_code == 200:
        lines = r.text.strip().splitlines()
        rows = []
        for line in lines:
            parts = re.split(r"[|\s,;]+", line.strip())
            if len(parts) >= 2:
                tkr, cik = parts[0].upper(), re.sub(r"\D", "", parts[1])
                if tkr and cik:
                    rows.append({"ticker": tkr, "title": "", "cik": cik})
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="last")
            df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
            return df[["ticker", "title", "cik"]]

    raise RuntimeError("Failed to build SEC ticker map from standard sources.")

def get_cik_from_search_index(ticker: str, email: str) -> Optional[str]:
    """
    Use the SEC 'search-index' API to find a CIK for a ticker symbol.
    Endpoint: https://efts.sec.gov/LATEST/search-index?keys=<TICKER>
    """
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {"keys": ticker}
    r = requests.get(url, params=params, headers=sec_headers(email), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    try:
        js = r.json()
    except Exception:
        return None
    # The JSON shape can vary. Look for first doc with a CIK-ish value.
    # Common spots: js['hits']['hits'][i]['_source']['ciks'] or ['cik']
    def _extract(jsobj) -> Optional[str]:
        if not isinstance(jsobj, dict):
            return None
        hits = jsobj.get("hits")
        if isinstance(hits, dict):
            arr = hits.get("hits")
            if isinstance(arr, list):
                for h in arr:
                    src = h.get("_source", {})
                    # ciks may be list of strings, sometimes 'cik'
                    if "ciks" in src and isinstance(src["ciks"], list) and src["ciks"]:
                        v = re.sub(r"\D", "", str(src["ciks"][0]))
                        if v:
                            return v.zfill(10)
                    if "cik" in src:
                        v = re.sub(r"\D", "", str(src["cik"]))
                        if v:
                            return v.zfill(10)
        return None
    return _extract(js)

def get_cik_for_ticker(ticker: str, email: str) -> str:
    """
    Try multiple sources with light retry. Final fallback: built-in map for demo tickers.
    """
    last_err = None
    for attempt in range(2):
        try:
            df = load_sec_ticker_map(email)
            row = df.loc[df["ticker"].str.upper() == ticker.upper()]
            if not row.empty:
                return row.iloc[0]["cik"]
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    # Try search-index per ticker
    try:
        cik = get_cik_from_search_index(ticker, email)
        if cik:
            return cik
    except Exception as e:
        last_err = e

    # Final hard-coded fallback for your demo tickers
    builtin = {"SBUX": "0000829224", "PFE": "0000078003"}
    if ticker.upper() in builtin:
        return builtin[ticker.upper()]

    raise last_err if last_err else RuntimeError(f"Unable to resolve CIK for ticker {ticker}")

def fetch_company_facts(cik: str, email: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=sec_headers(email), timeout=30)
    r.raise_for_status()
    return r.json()

# ----------------------------- Orchestrator -----------------------------

def ingest_one_ticker(ticker: str, keyword: str, cfg: IngestConfig):
    out_dir = Path(cfg.out_dir) / ticker
    ensure_dir(out_dir)
    start, end = daterange_utc(cfg.lookback_days)

    # 1) OHLCV
    if cfg.save in ("prices", "all"):
        ohlcv = fetch_ohlcv(ticker, start, end)
        save_parquet(ohlcv, out_dir / "prices.parquet", cfg.skip_existing)

    # 2) Google Trends
    if cfg.save in ("trends", "all"):
        trends = fetch_trends_chunked(
            keyword=keyword,
            geo=cfg.geo_trends,
            start=start,
            end=end,
            window_days=cfg.trends_window_days,
            base_pause=cfg.trends_base_pause,
            max_retries=cfg.trends_max_retries,
        )
        save_parquet(trends, out_dir / "trends.parquet", cfg.skip_existing)

    # 3) GDELT News
    if cfg.save in ("news", "all"):
        news = gdelt_daily_counts_and_tone(keyword, start, end)
        save_csv(news, out_dir / "gdelt_daily.csv", cfg.skip_existing)

    # 4) SEC Company Facts (raw JSON dump)
    if cfg.save in ("facts", "all"):
        cik = get_cik_for_ticker(ticker, cfg.email)
        facts = fetch_company_facts(cik, cfg.email)
        write_text(json.dumps(facts, ensure_ascii=False), out_dir / "facts.json", cfg.skip_existing)

    # Metadata (always update)
    meta = {
        "ticker": ticker,
        "keyword": keyword,
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "save": cfg.save,
        "trends_chunk_days": cfg.trends_window_days,
    }
    write_text(json.dumps(meta, indent=2), out_dir / "metadata.json", False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SBUX", "PFE"])
    parser.add_argument("--email", required=True, help="Contact email for SEC User-Agent header")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--geo_trends", default="")
    parser.add_argument("--save", choices=["prices", "trends", "news", "facts", "all"], default="all",
                        help="Select which data to fetch/save (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing output files")
    # Optional tuning knobs
    parser.add_argument("--trends_window_days", type=int, default=90)
    parser.add_argument("--trends_base_pause", type=float, default=2.0)
    parser.add_argument("--trends_max_retries", type=int, default=5)
    args = parser.parse_args()

    cfg = IngestConfig(
        tickers=args.tickers,
        email=args.email,
        out_dir=args.out_dir,
        geo_trends=args.geo_trends,
        trends_window_days=args.trends_window_days,
        trends_base_pause=args.trends_base_pause,
        trends_max_retries=args.trends_max_retries,
        save=args.save,
        skip_existing=args.skip_existing,
    )
    brand_map = {"SBUX": "Starbucks", "PFE": "Pfizer"}
    for t in cfg.tickers:
        ingest_one_ticker(t.upper(), brand_map.get(t.upper(), t.upper()), cfg)

if __name__ == "__main__":
    main()