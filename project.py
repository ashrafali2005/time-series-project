

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)



def generate_synthetic_timeseries(n=3000):
    """
    Generates a realistic multivariate dataset:
    - target: synthetic financial/energy-like series
    - features: noise, seasonality, trend, exogenous signals
    """
    rng = np.random.default_rng(42)

    t = np.arange(n)

    # Components
    trend = t * 0.001
    season = 2 * np.sin(t / 24) + np.sin(t / 7)
    noise = rng.normal(0, 0.3, n)
    spikes = (rng.random(n) < 0.01) * rng.normal(5, 2, n)

    target = 10 + trend + season + noise + spikes

    # Extra features
    feature1 = np.sin(t / 48) + rng.normal(0, 0.1, n)
    feature2 = np.cos(t / 16) + rng.normal(0, 0.1, n)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=n, freq="H"),
        "value": target,
        "feature1": feature1,
        "feature2": feature2,
    })

    return df


SEQ_LEN = 48
PRED_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
PATIENCE = 7


def preprocess(df):
    df = df.copy()

    df["hour"] = df.timestamp.dt.hour
    df["day"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"value_lag{lag}"] = df["value"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "value"]]

    X = df[feature_cols].values
    y = df["value"].values.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    return df, X_scaled, y_scaled, feature_cols, x_scaler, y_scaler


def create_sequences(X, y):
    xs, ys = [], []
    for i in range(len(X) - SEQ_LEN - PRED_HORIZON):
        xs.append(X[i:i + SEQ_LEN])
        ys.append(y[i + SEQ_LEN])
    return np.array(xs), np.array(ys)


class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                            batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.attn_weights = None

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H)
        out_t = out.transpose(0, 1)     # (T, B, H)
        attn_out, w = self.attn(out_t, out_t, out_t)
        self.attn_weights = w.detach().cpu()
        out2 = attn_out.transpose(0, 1)
        return self.fc(out2[:, -1, :])



def train(model, train_loader, val_loader):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=3, verbose=True
    )

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                val_losses.append(crit(pred, yb).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train {np.mean(tr_losses):.4f} | Val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping!")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader, scaler):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            pred = model(Xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    preds = scaler.inverse_transform(np.vstack(preds))
    trues = scaler.inverse_transform(np.vstack(trues))

    return (
        math.sqrt(mean_squared_error(trues, preds)),
        mean_absolute_error(trues, preds),
        trues,
        preds
    )


df = generate_synthetic_timeseries()
df, Xs, ys, feature_cols, xsc, ysc = preprocess(df)
Xseq, yseq = create_sequences(Xs, ys)

n = len(Xseq)
tr_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_ds = TSDataset(Xseq[:tr_end], yseq[:tr_end])
val_ds = TSDataset(Xseq[tr_end:val_end], yseq[tr_end:val_end])
test_ds = TSDataset(Xseq[val_end:], yseq[val_end:])

train_loader = DataLoader(train_ds, BATCH_SIZE, True)
val_loader = DataLoader(val_ds, BATCH_SIZE, False)
test_loader = DataLoader(test_ds, BATCH_SIZE, False)

input_dim = Xseq.shape[-1]

# Train baseline LSTM
print("\n=== TRAINING BASELINE LSTM ===")
lstm = LSTMModel(input_dim)
lstm = train(lstm, train_loader, val_loader)

# Train attention model
print("\n=== TRAINING ATTENTION MODEL ===")
attn = AttentionModel(input_dim)
attn = train(attn, train_loader, val_loader)

# Evaluation
lstm_rmse, lstm_mae, y_true, y_lstm = evaluate(lstm, test_loader, ysc)
attn_rmse, attn_mae, _, y_attn = evaluate(attn, test_loader, ysc)

print("\nAblation Study:")
print("LSTM     → RMSE:", lstm_rmse, " MAE:", lstm_mae)
print("ATTN     → RMSE:", attn_rmse, " MAE:", attn_mae)

# Plot predictions
plt.figure(figsize=(12, 5))
plt.plot(y_true, label="Actual")
plt.plot(y_lstm, label="LSTM Pred")
plt.plot(y_attn, label="Attention Pred")
plt.title("Test Predictions")
plt.legend()
plt.show()

# Attention weights heatmap
w = attn.attn_weights.mean(dim=0)[0].numpy()
plt.imshow(w, cmap="viridis", aspect="auto")
plt.colorbar(label="Attention Weight")
plt.title("Attention Heatmap (Sample)")
plt.show()
