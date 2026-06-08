import ta
import matplotlib.pyplot as plt

from data import get_data

# --------------------

ticker = input("Enter ticker sybol: ")
df = get_data(ticker)

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

print(df.columns.tolist())
print(df.shape)

# --------------------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (14, 12), sharex = True)

# Price + Bollinger Bands + SMAs
ax1.plot(df.index, df["Close"], label = "Close", linewidth = 1.2)
ax1.plot(df.index, df["sma_20"], label = "SMA 20", linestyle = "--")
ax1.plot(df.index, df["sma_50"], label = "SMA 50", linestyle = "--")
ax1.fill_between(df.index, df["bb_low"], df["bb_high"], alpha = 0.15, label = "Bollinger Bands")
ax1.set_title("Price (USD)")
ax1.legend(fontsize = 8)
ax1.grid(alpha = 0.3)

# RSI
ax2.plot(df.index, df["rsi"], color = "purple", linewidth = 1)
ax2.axhline(70, color = "red", linestyle = "--", alpha = 0.8)
ax2.axhline(30, color = "green", linestyle = "--", alpha = 0.8)
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.grid(alpha = 0.3)

# MACD
ax3.plot(df.index, df["macd"], label="MACD", linewidth = 1)
ax3.plot(df.index, df["macd_signal"], label = "Signal", linewidth = 1, linestyle = "--")
ax3.axhline(0, color = "gray", linewidth = 0.8)
ax3.set_ylabel("MACD")
ax3.legend(fontsize = 8)
ax3.grid(alpha = 0.3)

plt.tight_layout()
plt.savefig("day2_plot.png", dpi = 150)