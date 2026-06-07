import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --------------------

ticker = "NVDA"
df = yf.download(ticker, start = "2022-01-01", end = "2024-12-31", auto_adjust = True)

print(df.head())
print(df.shape)

# Drop rows with missing values
df.dropna(inplace = True)
df.columns = df.columns.get_level_values(0)

# Columns we care about
df = df[["Open", "High", "Low", "Close", "Volume"]]

print(df.dtypes)
print(df.isnull().sum())

# ---------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (14, 8), sharex = True)

# Price
ax1.plot(df.index, df["Close"], color = "royalblue", linewidth = 1.2)
ax1.set_title(f"{ticker} Closing Price")
ax1.set_ylabel("Price (USD)")
ax1.grid(alpha = 0.3)


# Volume
ax2.fill_between(df.index, df["Volume"], color="gray", alpha=0.5)
ax2.set_ylabel("Shares Traded")
ax2.grid(alpha = 0.3)

plt.tight_layout()

confirm = input("Save plot? (y): ")
if confirm.lower() == "y":
    plt.savefig("day1_plot.png", dpi = 150)
elif confirm.lower() != "skip":
    confirm = input("Enter y for yes -or- skip to skip: ")
    if confirm.lower() == "y":
        plt.savefig("day1_plot.png", dpi = 150)

plt.show()