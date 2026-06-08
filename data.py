import yfinance as yf
import pandas as pd

# --------------------

def get_data(ticker, start = "2015-01-01", end = "2024-12-31"):
    df = yf.download(ticker, start = start, end = end, auto_adjust = True)
    df.columns = df.columns.get_level_values(0)
    df.dropna(inplace = True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df