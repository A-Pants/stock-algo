import torch
import numpy as np
import matplotlib.pyplot as plt

from data import get_data
from features import add_indicators, prepare_data, split_data
from model import LSTMModel

# ------------------

# Get the data
ticker = input("Enter ticker sybol: ")
df = get_data(ticker)

df = add_indicators(df)

X, y, dates = prepare_data(df)

input_size = X.shape[2]

X_train, X_val, X_test, y_train, y_val, y_test, dates_list = split_data(X, y, dates, 0.8)

X_test = torch.tensor(X_test, dtype=torch.float32)

# ------------------

# Model
model = LSTMModel(input_size = input_size)

# Load weights
model.load_state_dict(torch.load("model.pth"))

# ------------------

# Eval
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test))

    predicted_labels = (predictions > 0.19930227).float()

    predictions_np = predictions.numpy()
    predicted_labels_np = predicted_labels.numpy()

# Flatten
predicted_labels_np = predicted_labels_np.flatten()
predictions_np = predictions_np.flatten()

# ------------------

# Plot
fig, axs = plt.subplots(nrows = 6, ncols = 1, sharex = 'all', gridspec_kw = {"height_ratios": [3, 1, 1, 1, 1, 1]}, 
                        figsize=(20, 16))

start_date = df.loc[dates_list[0]:]
buys = np.where(predicted_labels_np == 1)[0]
up_days = np.where(y_test == 1)[0]

# Price + Bollinger Bands + buy signals + actual 3% up days
# Volume
# RSI
# MACD
# Prediction probabilities
# P&L curve

# ---------------

axs[0].plot(dates_list, start_date["Close"], color = 'blue', label = 'Close')
axs[0].plot(dates_list, start_date["bb_high"], color = 'gray', label = 'BB High')
axs[0].plot(dates_list, start_date["bb_low"], color = 'gray', label = 'BB Low')

print(f"Total buy signals: {buys.shape[0]}")
print(f"Total up days: {up_days.shape[0]}")
print(f"Total test days: {len(dates_list)}")

axs[0].scatter(dates_list[buys], start_date["Close"].iloc[buys], c ='green', label = 'Buy Signal')
axs[0].scatter(dates_list[up_days], start_date["Close"].iloc[up_days], c ='orange', label = '3% up days')
axs[0].legend()

# ----------------

up_mask = start_date["Close"] > start_date["Open"]
up_days = start_date[up_mask]
down_days = start_date[~up_mask]

axs[1].bar(up_days.index, up_days["Volume"], width=0.8, color = 'green', label = 'Up Days')
axs[1].bar(down_days.index, down_days["Volume"], width=0.8, color = 'red', label = 'Down Days')
axs[1].legend()

# ----------------


axs[2].axhline(y = 70, color = 'red', label = 'Overbought')
axs[2].axhline(y = 30, color = 'green', label = 'Oversold')
axs[2].plot(dates_list, start_date["rsi"], color = 'purple', label = 'RSI')
axs[2].legend()

# ----------------

difference = start_date['macd'] - start_date['macd_signal']
up_mask = difference > 0
up_days = start_date[up_mask]
down_days = start_date[~up_mask]

axs[3].axhline(y = 0, color = 'grey')
axs[3].plot(dates_list, start_date["macd"], color = 'blue', label = 'MACD')
axs[3].plot(dates_list, start_date['macd_signal'], color = 'orange', label = 'MACD Signal')
axs[3].bar(up_days.index, difference[up_mask], color = 'green', label = 'Bullish')
axs[3].bar(down_days.index, difference[~up_mask], color = 'red', label = 'Bearish')
axs[3].legend()

# -----------------

axs[4].axhline(y = 0.5, color = 'green', label = 'High Confidence', linestyle = "--")
axs[4].axhline(y = 0.4, color = 'yellow', label = 'Medium Confidence', linestyle = "--")
axs[4].axhline(y = 0.3, color = 'red', label = 'Low Confidence', linestyle = "--")
axs[4].plot(dates_list, predictions_np, color = 'blue', label = 'Prediction Probability')
axs[4].legend()

# -----------------

profit_ratio = (start_date["Close"].iloc[buys] / start_date["Open"].iloc[buys]) - 1
profit = np.zeros(len(dates_list))
profit[buys] = 1000 * profit_ratio.values
profit = profit.cumsum()

buy_and_hold_ratio = 1000 / start_date["Open"].iloc[0]
buy_and_hold_profit = start_date["Close"] * buy_and_hold_ratio - 1000

axs[5].plot(dates_list, profit, color = 'blue', label = 'P&L')
axs[5].plot(buy_and_hold_profit, color = 'pink', label = 'Buy and Hold Profit')
axs[5].legend()

# -----------------

fig.suptitle(f"{ticker} ML Trading Model")
plt.tight_layout()
plt.savefig("day4_plot.png", dpi = 150)