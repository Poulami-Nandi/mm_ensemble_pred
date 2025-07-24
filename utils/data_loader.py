import pandas as pd
import os

def load_ohlcv(stock):
    file_map = {
        "MSFT": "data/MSFT_OHLCV.csv",
        "TSLA": "data/TSLA_OHLCV.csv"
    }
    df = pd.read_csv(file_map[stock], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def load_trends(stock):
    file_map = {
        "MSFT": "data/MSFT_trends.csv",
        "TSLA": "data/TSLA_trends.csv"
    }
    df = pd.read_csv(file_map[stock], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def load_and_prepare_data(stock):
    ohlcv = load_ohlcv(stock)
    trends = load_trends(stock)

    # Merge on Date
    df = pd.merge(ohlcv, trends, on="Date", how="inner")

    # Rename Google Trends column to 'Trend' if not already
    if 'Trend' not in df.columns:
        trend_col = [col for col in df.columns if col.lower() not in ['date', 'open', 'high', 'low', 'close', 'volume']]
        if trend_col:
            df.rename(columns={trend_col[0]: 'Trend'}, inplace=True)

    # Drop missing values if any
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)
    return df
