#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from mm_ensemble.utils.paths import DATA_DIR, OUTPUTS_DIR

def _norm_ds(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None).dt.floor("D")

def _load_preds(tdir: Path, model: str):
    df = pd.read_parquet(tdir / "predictions.parquet")
    df["ds"] = _norm_ds(df["ds"])
    if model == "ensemble": col = "y_ens"
    elif model == "xgb": col = "y_xgb"
    elif model == "arima": col = "y_arima"
    else: raise ValueError("model must be ensemble|xgb|arima")
    return df[["ds","split","y_true",col]].rename(columns={col:"yhat"})

def _load_prices(tdir: Path, price_col: str):
    px = pd.read_parquet(tdir / "prices.parquet")
    px["ds"] = _norm_ds(px["ds"])
    if price_col not in px.columns:
        if price_col == "adj_close" and "close" in px.columns:
            price_col = "close"
        else:
            raise KeyError(f"{price_col} not found in prices")
    return px[["ds", price_col]].rename(columns={price_col: "px"})

def _metrics_from_curve(curve: pd.Series, freq_per_year=252):
    rets = curve.pct_change().dropna()
    if rets.empty:
        return {"CAGR":0,"Sharpe":0,"MaxDD":0,"HitRate":0,"Len":0}
    total_ret = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    n_years = max(1e-9, len(rets) / freq_per_year)
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    sharpe = float(np.sqrt(freq_per_year) * rets.mean() / (rets.std() + 1e-12))
    # Max drawdown
    roll_max = curve.cummax()
    dd = (curve / roll_max - 1.0)
    maxdd = float(dd.min())
    hit = float((rets > 0).mean())
    return {"CAGR":float(cagr), "Sharpe":float(sharpe), "MaxDD":maxdd, "HitRate":hit, "Len":int(len(rets))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--model", choices=["ensemble","xgb","arima"], default="ensemble")
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--threshold", type=float, default=0.0, help="Go long if yhat > threshold, else flat")
    ap.add_argument("--tcost-bps", type=float, default=5.0, help="Transaction cost in basis points per entry/exit")
    ap.add_argument("--price-col", choices=["adj_close","close"], default="adj_close")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    tdir = Path(args.root) / args.ticker.upper()
    preds = _load_preds(tdir, args.model)
    preds = preds.loc[preds["split"] == args.split].copy().sort_values("ds")

    prices = _load_prices(tdir, args.price_col).sort_values("ds")
    df = preds.merge(prices, on="ds", how="inner").copy()

    # daily return from price
    df["ret_px"] = df["px"].pct_change()

    # position: long if pred > threshold, else 0
    df["pos"] = (df["yhat"] > args.threshold).astype(float)

    # turnover & costs (bps converted to decimal)
    df["turnover"] = df["pos"].diff().abs().fillna(df["pos"].abs())
    cost = args.tcost_bps / 10000.0
    df["strategy_ret"] = df["pos"].shift(1).fillna(0) * df["ret_px"] - df["turnover"] * cost

    # equity curve
    df["equity"] = (1 + df["strategy_ret"]).cumprod()

    metrics = _metrics_from_curve(df["equity"])
    metrics.update({
        "model": args.model, "split": args.split, "threshold": args.threshold,
        "tcost_bps": args.tcost_bps, "price_col": args.price_col
    })

    if args.out_csv:
        df[["ds","y_true","yhat","pos","ret_px","strategy_ret","equity"]].to_csv(args.out_csv, index=False)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()