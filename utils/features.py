import pandas as pd
import numpy as np

def add_technical_indicators(df):
    df = df.copy()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Averages
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Bollinger Bands
    df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # Volatility (Rolling Std)
    df['Volatility_10'] = df['Close'].rolling(window=10).std()

    # Google Trends Momentum
    if 'Trend' in df.columns:
        df['Trend_Momentum_3'] = df['Trend'] - df['Trend'].shift(3)
        df['Trend_MA_7'] = df['Trend'].rolling(window=7).mean()

    # Drop NA rows after indicator computation
    df.dropna(inplace=True)

    return df
