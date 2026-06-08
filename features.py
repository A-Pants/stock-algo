import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler

# ------------------

def prepare_data(df, window = 20):
    # Create label: 1 if tommorow's close is higher than today's
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"] * 1.03).astype(int)
    # print(df["target"].value_counts(normalize=True))  # Check the split of our data (0 vs 1)
    df.dropna(inplace = True)

    # Features and Labels
    feature_cols = ["Open", "High", "Low", "Close", "Volume", 
                    "sma_20", "sma_50", "rsi", "macd", "macd_signal", "bb_high", "bb_low"]
    
    X_raw = df[feature_cols].values
    y_raw = df["target"].values

    # Normalize features to 0-1 range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Sliding windows
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window : i]) # Grab rows i - window to i
        y.append(y_raw[i])

    return np.array(X), np.array(y)

# ------------------

def add_indicators(df):
    # Simple Moving Average (SMA)
    df["sma_20"] = ta.trend.sma_indicator(df["Close"], window = 20)
    df["sma_50"] = ta.trend.sma_indicator(df["Close"], window = 50)

    # RSI
    df["rsi"] = ta.momentum.rsi(df["Close"], window = 14)

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window = 20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df.dropna(inplace = True)

    return df