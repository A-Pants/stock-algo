import torch
import torch.nn as nn
import ta

import numpy as np

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report

from data import get_data
from features import prepare_data, split_data, add_indicators, calculate_weights, calculate_prediction_threshold
from model import LSTMModel

# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------

# List of stocks to check
tickers = [
    "NVDA", "TSM",                          # Semiconductors
    "TSLA", "META", "NFLX",                 # High-volatility tech
    "MSFT", "AAPL", "GOOGL", "AMZN",        # Big tech
    "BAC",  "JPM",                          # Financial
    "GE",                                   # Industrial
    "COP",  "OXY"                           # Energy
    ]

# Empty lists to collect yfinance data
X_train_list = []
# X_val_list = []
X_test_list = []

y_train_list = []
# y_val_list = []
y_test_list = []

for ticker in tickers:
    df = get_data(ticker)

    df = add_indicators(df)

    X_new, y_new, dates = prepare_data(df)

    X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new, dates_list_new = split_data(X_new, y_new, dates, 0.8)

    X_train_list.append(X_train_new)
    # X_val_list.append(X_val_new)
    X_test_list.append(X_test_new)

    y_train_list.append(y_train_new)
    # y_val_list.append(y_val_new)
    y_test_list.append(y_test_new)

# Numpy array of training, validation(currently unused), and test data
X_train_np = np.concatenate(X_train_list)
# X_val_np = np.concatenate(X_val_list)
X_test_np = np.concatenate(X_test_list)

y_train_np = np.concatenate(y_train_list)
# y_val_np = np.concatenate(y_val_list)
y_test_np = np.concatenate(y_test_list)

input_size = X_train_np.shape[2]

# -------------------

X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

# -------------------

# Weigh the data
sample_weights = calculate_weights(X_train.shape[0], decay_rate = 0.0005)  # Decay rate should be between 0.0005 and 0.001
sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(X_train_np), replacement = True)

# Data Loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, sampler = sampler, batch_size = 32)

# Model
model = LSTMModel(input_size = input_size)
model.to(device)

# Loss
pos_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

# Optimizer
# 0.0001 & 0.001 are too small of steps
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# -----------------

# Train
epochs = 50

for epoch in range(epochs):
    # Ensure dropout is active during training
    model.train()
    # Loss per epoch
    epoch_loss = 0

    # Training loop
    for X_batch, y_batch in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()

        # Backwards pass
        loss.backward()

        # Update weights
        optimizer.step()

    # Print each epochs results
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ------------------

# Save the model
torch.save(model.state_dict(), "model.pth")

# ------------------

# Eval
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test))  # Added sigmoid here because of weighted loss

    print(f"Min prediction: {predictions.min().item():.4f}")
    print(f"Max prediction: {predictions.max().item():.4f}")
    print(f"Mean prediction: {predictions.mean().item():.4f}")

    predictions_np = predictions.squeeze().cpu().numpy()

    best_threshold, best_f1_score = calculate_prediction_threshold(y_test_np, predictions_np)

    print(best_threshold, best_f1_score)

    predicted_labels = (predictions > 0.3).float()
    accuracy = (predicted_labels == y_test).float().mean().item()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test.cpu().numpy(), predicted_labels.cpu().numpy()))