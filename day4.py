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

X_train, X_val, X_test, y_train, y_val, y_test, dates_list = split_data(X, y, dates, 0.8)

X_test = torch.tensor(X_test, dtype=torch.float32)

# ------------------

# Model
model = LSTMModel()

# Load weights
model.load_state_dict(torch.load("model.pth"))

# ------------------

# Eval
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test))

    predicted_labels = (predictions > 0.3).float()

    predictions_np = predictions.numpy()
    predicted_labels_np = predicted_labels.numpy()

# Flatten
predicted_labels_np = predicted_labels_np.flatten()
predictions_np = predictions_np.flatten()

# ------------------

# Plot
fig, axs = plt.subplots(nrows = 6, ncols = 1, sharex = 'all', gridspec_kw = {"height_ratios": [3, 1, 1, 1, 1, 1]}, figsize=(20, 16))

start_date = df.loc[dates_list[0]:]
buys = np.where(predicted_labels_np == 1)[0]
up_days = np.where(y_test == 1)[0]

# Price + Bollinger Bands + buy signals + actual 3% up days
# Volume
# RSI
# MACD
# Prediction probabilities
# P&L curve

axs[0].plot(dates_list, start_date["Close"], c = 'blue', label = 'Close')
axs[0].plot(dates_list, start_date["bb_high"], c = 'gray', label = 'BB High')
axs[0].plot(dates_list, start_date["bb_low"], c = 'gray', label = 'BB Low')

print(f"Total buy signals: {buys.shape[0]}")
print(f"Total up days: {up_days.shape[0]}")
print(f"Total test days: {len(dates_list)}")

axs[0].scatter(dates_list[buys], start_date["Close"].iloc[buys], c='green', label = 'Buy Signal')
axs[0].scatter(dates_list[up_days], start_date["Close"].iloc[up_days], c='orange', label = '3% up days')
axs[0].legend()

plt.savefig("day4_plot.png", dpi = 150)