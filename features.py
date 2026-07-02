import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve

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

    # ATR
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14, fillna=False)
    df["ATR"] = atr.average_true_range()
    df["ATR"] /= df["Close"]

    df.dropna(inplace = True)

    return df

# ------------------

def prepare_data(df, window = 20):
    # Create label: 1 if tommorow's close is higher than today's
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"] * 1.03).astype(int)
    # print(df["target"].value_counts(normalize=True))  # Check the split of our data (0 vs 1)
    df.dropna(inplace = True)

    # Features and Labels
    feature_cols = ["Open", "High", "Low", "Close", "Volume", 
                    "sma_20", "sma_50", "rsi", "macd", "macd_signal", "bb_high", "bb_low", "ATR"]
    
    X_raw = df[feature_cols].values
    y_raw = df["target"].values
    dates = df.index

    # Normalize features to 0-1 range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Sliding windows
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window : i]) # Grab rows i - window to i
        y.append(y_raw[i])

    dates = dates[window:]

    return np.array(X), np.array(y), np.array(dates)

# ------------------

# X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np, dates_list = split_data(X, y, dates, split)
def split_data(X, y, dates, train_test_split, val_size = 0.0):
    if val_size > 0.0:
        # Train/Test split not using sklearn
        val_split = int(len(X) * train_test_split)
        train_split = int(val_split * (1 - val_size))

        dates_list = dates[val_split:]

        X_train_np = X[:train_split]
        X_val_np = X[train_split : val_split]
        X_test_np = X[val_split:]

        y_train_np = y[:train_split]
        y_val_np = y[train_split : val_split]
        y_test_np = y[val_split:]

        return X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np, dates_list

    else:
        # Train/Test split not using sklearn
        split = int(len(X) * train_test_split)

        dates_list = dates[split:]

        X_train_np = X[:split]
        X_test_np = X[split:]
        y_train_np = y[:split]
        y_test_np = y[split:]

        return X_train_np, None, X_test_np, y_train_np, None, y_test_np, dates_list
    
# ---------------------

def calculate_weights(data_length, decay_rate):
    return np.exp(decay_rate * np.arange(data_length))

# ---------------------

def calculate_prediction_threshold(y_test_np, predictions_np):

    precision, recall, thresholds = precision_recall_curve(y_test_np, predictions_np)
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    best_f1_index = np.argmax(f1_scores)

    best_f1_score = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]

    return best_threshold, best_f1_score