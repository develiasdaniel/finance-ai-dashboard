import os
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
@dataclass
class Config:
    output_dir: str = "outputs/lstm_pytorch_daily_v2"
    plots_dir: str = "outputs/lstm_pytorch_daily_v2/plots"
    csv_path: str = "../data/lstm_daily_dataset.csv"

    date_col: str = "date_day"
    client_col: str = "client_id"
    target_col: str = "daily_spend"

    feature_cols = [
        "tx_count", "unique_mcc", "unique_merchants",
        "online_ratio", "chip_ratio",
        "day_of_week", "is_weekend", "month", "day_of_month"
    ]

    client_id: int = 1098

    # split
    train_ratio: float = 0.8

    # evaluation: last N days
    n_days_eval: int = 7

    # outlier handling
    clip_outliers: bool = True
    clip_q_low: float = 0.01
    clip_q_high: float = 0.95

    # LSTM specific
    lookback: int = 14  # Use last 14 days to predict next day

    # ========== HYPERPARAMETERS TO EXPERIMENT ==========
    lstm_units: int = 64  # Try: 32, 64, 128, 256
    n_lstm_layers: int = 1  # Try: 1, 2, 3
    dense_units: int = 32  # Try: 16, 32, 64
    dropout_rate: float = 0.2  # Try: 0.1, 0.2, 0.3, 0.5
    use_dropout: bool = True

    # Training
    batch_size: int = 32  # Try: 16, 32, 64
    epochs: int = 100  # Try: 50, 100, 200
    learning_rate: float = 0.001  # Try: 0.001, 0.0001, 0.01

    # Early stopping
    early_stopping: bool = True
    patience: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LSTM Model (PyTorch)
# =========================
class LSTMForecastModel(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_size=1, lstm_units=64, n_lstm_layers=1,
                 dense_units=32, dropout_rate=0.2):
        super(LSTMForecastModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=n_lstm_layers,
            dropout=dropout_rate if n_lstm_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(lstm_units, dense_units)
        self.dense2 = nn.Linear(dense_units, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, lookback, n_features)
        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        out = self.dropout(last_out)
        out = self.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


# =========================
# Utils
# =========================
def ensure_dirs(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)


def load_preprocessed_daily(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.target_col]).copy()

    # filter client
    df = df[df[cfg.client_col] == cfg.client_id].copy()

    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    return df


def preprocess_target(df: pd.DataFrame, cfg: Config):
    y = df[cfg.target_col].copy()

    clip_info = {}
    if cfg.clip_outliers:
        q_low = y.quantile(cfg.clip_q_low)
        q_high = y.quantile(cfg.clip_q_high)
        y = y.clip(lower=q_low, upper=q_high)
        clip_info = {"q_low_value": float(q_low), "q_high_value": float(q_high)}
    else:
        clip_info = {"q_low_value": None, "q_high_value": None}

    df[cfg.target_col] = y
    return df, clip_info


def scale_data(df: pd.DataFrame, cfg: Config):
    n = len(df)
    split_idx = int(n * cfg.train_ratio)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = df[cfg.feature_cols].iloc[:split_idx]
    train_target = df[[cfg.target_col]].iloc[:split_idx]

    feature_scaler.fit(train_features)
    target_scaler.fit(train_target)

    features_scaled = feature_scaler.transform(df[cfg.feature_cols])
    target_scaled = target_scaler.transform(df[[cfg.target_col]]).flatten()

    return features_scaled, target_scaled, feature_scaler, target_scaler, split_idx


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(target) - lookback):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()

    return total_loss / len(val_loader)


def train_lstm(model, train_loader, val_loader, cfg: Config, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("   Training LSTM (PyTorch)...")

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"     Epoch {epoch + 1}/{cfg.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if cfg.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"     Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_state)
                    break

    print(f"   ✓ Trained for {len(train_losses)} epochs")

    history = {'loss': train_losses, 'val_loss': val_losses}
    return history


@torch.no_grad()
def predict(model, X_test, device):
    model.eval()
    X_test = X_test.to(device)
    outputs = model(X_test)
    return outputs.cpu().numpy().flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def main():
    cfg = Config()
    ensure_dirs(cfg)

    print("\n" + "=" * 70)
    print("LSTM Daily Forecasting Pipeline (PyTorch)".center(70))
    print("=" * 70 + "\n")
    print(f"🔧 Device: {cfg.device}\n")

    print("1) Loading data...")
    df = load_preprocessed_daily(cfg)
    print(f"   ✓ Rows: {len(df)} | Client: {cfg.client_id}")

    print("\n2) Preprocessing target (clipping)...")
    df, clip_info = preprocess_target(df, cfg)

    print("\n3) Scaling data (features + target)...")
    features_scaled, target_scaled, f_scaler, t_scaler, split_idx = scale_data(df, cfg)

    print("\n4) Creating sequences...")
    X, y = create_sequences(features_scaled, target_scaled, cfg.lookback)

    n_train_seq = int(len(y) * cfg.train_ratio)
    X_train, y_train_seq = X[:n_train_seq], y[:n_train_seq]
    X_test, y_test_seq = X[n_train_seq:], y[n_train_seq:]

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_seq.reshape(-1, 1))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    print(f"   ✓ Train sequences: {X_train_tensor.shape}")
    print(f"   ✓ Test sequences: {X_test_tensor.shape}")

    print("\n5) Building LSTM model...")
    model = LSTMForecastModel(
        input_size=len(cfg.feature_cols),
        lstm_units=cfg.lstm_units,
        n_lstm_layers=cfg.n_lstm_layers,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate
    ).to(cfg.device)

    print("\n6) Training LSTM...")
    history = train_lstm(model, train_loader, val_loader, cfg, cfg.device)

    print("\n7) LSTM inference on test set...")
    y_pred_scaled = predict(model, X_test_tensor, cfg.device)

    y_pred_raw = t_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_raw = t_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    dates_all = df[cfg.date_col].iloc[cfg.lookback:]
    dates_test = dates_all.iloc[n_train_seq:n_train_seq + len(y_pred_raw)]

    print("\n8) Evaluating metrics on full test set...")
    metrics = compute_metrics(y_test_raw, y_pred_raw)

    print(f"   ✓ MAE: ${metrics['mae']:.2f}")
    print(f"   ✓ RMSE: ${metrics['rmse']:.2f}")
    print(f"   ✓ MAPE: {metrics['mape']:.2f}%")

if __name__ == "__main__":
    main()