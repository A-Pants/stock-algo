import torch
import torch.nn as nn
import ta

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from data import get_data
from features import prepare_data, add_indicators
from model import LSTMModel

# -------------------

# Get the data
ticker = input("Enter ticker sybol: ")
df = get_data(ticker)

df = add_indicators(df)

# Dates is unused, I updated prepare_data so I'm putting it here so I can retrain/save the day3 model
X, y, dates = prepare_data(df)

# -------------------

# Split the data
# Train/Test split using sklearn
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

# Train/Test split not using sklearn
split = int(len(X) * 0.8)

X_train_np = X[:split]
X_test_np = X[split:]
y_train_np = y[:split]
y_test_np = y[split:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

# -------------------

# Data Loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

# Model
model = LSTMModel()

# Loss
pos_weight = torch.tensor([2.0])
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

    predicted_labels = (predictions > 0.3).float()
    accuracy = (predicted_labels == y_test).float().mean().item()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test.numpy(), predicted_labels.numpy()))