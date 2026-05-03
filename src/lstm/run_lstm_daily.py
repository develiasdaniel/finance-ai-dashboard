import os
import json
import warnings
from dataclasses import dataclass

import joblib
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
    csv_path: str = "../data/lstm_daily_dataset_86.csv"
    #csv_path: str = "../data/lstm_daily_dataset.csv"

    date_col: str = "date_day"
    client_col: str = "client_id"
    target_col: str = "daily_spend"

    feature_cols = [
        "tx_count", "unique_mcc", "unique_merchants",
        "online_ratio", "chip_ratio",
        "day_of_week", "is_weekend", "month", "day_of_month"
    ]

    #client_id: int = 1098
    client_id: int = 86

    # split
    train_ratio: float = 0.8

    # evaluation: last N days
    n_days_eval: int = 20

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
    return {"mae": float(mae), "rmse": float(rmse)}



def save_artifacts(cfg: Config, model, scalers: dict, metrics: dict, predictions_df: pd.DataFrame, metadata: dict):
    model_path = os.path.join(cfg.output_dir, "lstm_model.pth")
    scalers_path = os.path.join(cfg.output_dir, "scalers.pkl")
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    preds_path = os.path.join(cfg.output_dir, "predictions.csv")
    meta_path = os.path.join(cfg.output_dir, "metadata.json")

    # Guardar modelo de PyTorch y scalers
    torch.save(model.state_dict(), model_path)
    joblib.dump(scalers, scalers_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_df.to_csv(preds_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Model saved: {model_path}")
    print(f"   Scalers saved: {scalers_path}")
    print(f"   Metrics saved: {metrics_path}")
    print(f"   Predictions saved: {preds_path}")
    print(f"   Metadata saved: {meta_path}")


# =========================
# Visualizations (Matched to ARIMA)
# =========================
def plot_eda(cfg: Config, y_raw: pd.Series):
    print("   Generating EDA plots...")
    plt.figure(figsize=(8, 5))
    box = plt.boxplot(y_raw.values, labels=["Expenses"])

    min_val, max_val = y_raw.min(), y_raw.max()
    median = y_raw.median()
    q1, q3 = y_raw.quantile(0.25), y_raw.quantile(0.75)

    plt.text(1.1, min_val, f"Min: {min_val:.1f}", fontsize=10, color="blue")
    plt.text(1.1, max_val, f"Max: {max_val:.1f}", fontsize=10, color="blue")
    plt.text(1.1, median, f"Median: {median:.1f}", fontsize=10, color="orange")
    plt.text(1.1, q1, f"Q1: {q1:.1f}", fontsize=10, color="green")
    plt.text(1.1, q3, f"Q3: {q3:.1f}", fontsize=10, color="green")

    plt.title("Exploratory Data Analysis: Daily Expenses Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Daily Expenses")
    plt.ylabel("Expense ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "00_eda_boxplot.png"), dpi=150)
    plt.close()
    print("     ✓ 00_eda_boxplot.png")


def plot_series_with_split(cfg: Config, y_raw_series: pd.Series, split_idx: int):
    print("   Generating main plots...")
    plt.figure(figsize=(14, 5))
    plt.plot(y_raw_series.index, y_raw_series.values, label="Daily Expense (raw)", linewidth=1.5, alpha=0.8)
    plt.axvline(y_raw_series.index[split_idx], color="red", linestyle="--", linewidth=2,
                label="Train/Test split")
    plt.title("Daily Expense Time Series (Complete History)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Amount ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "01_series_with_split.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 01_series_with_split.png")


def plot_training_history(cfg: Config, history: dict):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title('LSTM Training History (Loss)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "01b_training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 01b_training_history.png")


def plot_forecast_vs_actual(cfg: Config, y_test_series: pd.Series, y_pred_raw: np.ndarray,
                            y_test_eval_series: pd.Series = None, y_pred_eval_raw: np.ndarray = None):
    if y_test_eval_series is not None:
        y_test_plot = y_test_eval_series
        y_pred_plot = y_pred_eval_raw
        title_suffix = f"(Last {len(y_test_eval_series)} Days - Zoomed View)"
    else:
        y_test_plot = y_test_series
        y_pred_plot = y_pred_raw
        title_suffix = f"(Full Test Set: {len(y_test_series)} Days)"

    plt.figure(figsize=(14, 6))
    plt.plot(y_test_plot.index, y_test_plot.values, marker="o", label="Actual (Test)",
             linewidth=2, markersize=8, color="blue")
    plt.plot(y_test_plot.index, y_pred_plot, marker="s", label="LSTM Forecast",
             linewidth=2, markersize=8, color="orange", alpha=0.9)
    plt.title(f"LSTM Forecast vs Actual {title_suffix}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Amount ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if y_test_eval_series is not None:
        plt.savefig(os.path.join(cfg.plots_dir, "02_forecast_vs_actual_7days.png"), dpi=150, bbox_inches="tight")
        print("     ✓ 02_forecast_vs_actual_7days.png (zoomed view)")
    else:
        plt.savefig(os.path.join(cfg.plots_dir, "02_forecast_vs_actual_full.png"), dpi=150, bbox_inches="tight")
        print("     ✓ 02_forecast_vs_actual_full.png (full test)")
    plt.close()


def plot_residuals(cfg: Config, y_test_series: pd.Series, y_pred_raw: np.ndarray,
                   y_test_eval_series: pd.Series = None, y_pred_eval_raw: np.ndarray = None):
    if y_test_eval_series is not None:
        y_test_plot = y_test_eval_series
        y_pred_plot = y_pred_eval_raw
        title_suffix = f"(Last {len(y_test_eval_series)} Days)"
    else:
        y_test_plot = y_test_series
        y_pred_plot = y_pred_raw
        title_suffix = f"(Full Test Set: {len(y_test_series)} Days)"

    residuals = y_test_plot.values - y_pred_plot

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(residuals)), residuals,
            color=["green" if r >= 0 else "red" for r in residuals], alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    plt.title(f"Residuals (Actual - Forecast) {title_suffix}", fontsize=14, fontweight="bold")
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Residual ($)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if y_test_eval_series is not None:
        plt.savefig(os.path.join(cfg.plots_dir, "03_residuals_7days.png"), dpi=150, bbox_inches="tight")
        print("     ✓ 03_residuals_7days.png (zoomed view)")
    else:
        plt.savefig(os.path.join(cfg.plots_dir, "03_residuals_full.png"), dpi=150, bbox_inches="tight")
        print("     ✓ 03_residuals_full.png (full test)")
    plt.close()


def plot_metrics(cfg: Config, metrics: dict):
    print("   Generating metrics visualization...")
    mae, rmse = metrics['mae'], metrics['rmse']

    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ['MAE\n($)', 'RMSE\n($)']
    metrics_values = [mae, rmse]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title('LSTM Model Evaluation Metrics (Test Set)', fontsize=14, fontweight="bold")
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        height = bar.get_height()
        label = f'{value:.2f}%' if i == 2 else f'${value:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2., height, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "04_metrics_bar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 04_metrics_bar_chart.png")

    # Metrics summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['MAE', f'${mae:.2f}', f'Average absolute error per day'],
        ['RMSE', f'${rmse:.2f}', f'Root Mean Squared Error (penalizes large errors)'],
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.15, 0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(3):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

    plt.title('LSTM Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "05_metrics_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 05_metrics_table.png")


def main():
    cfg = Config()
    ensure_dirs(cfg)

    print("\n" + "=" * 70)
    print("LSTM Daily Forecasting Pipeline (PyTorch)".center(70))
    print("=" * 70 + "\n")
    print(f"🔧 Device: {cfg.device}\n")

    print("1) Loading preprocessed daily data...")
    df = load_preprocessed_daily(cfg)
    print(f"   ✓ Loaded {len(df)} days of data | Client: {cfg.client_id}")
    print(f"   Date range: {df[cfg.date_col].min().date()} to {df[cfg.date_col].max().date()}")
    n_zeros = (df[cfg.target_col] == 0).sum()
    print(f"   ✓ Zeros count in dataseries: {n_zeros} ({n_zeros / len(df) * 100:.2f}%)")

    # Create a Pandas Series mapped to dates for ARIMA-like plotting
    y_raw_series = df.set_index(cfg.date_col)[cfg.target_col].copy()

    print("\n2) Exploratory Data Analysis (EDA):")
    print(f"   Mean daily expense: ${y_raw_series.mean():.2f}")
    print(f"   Median daily expense: ${y_raw_series.median():.2f}")
    print(f"   Std Dev: ${y_raw_series.std():.2f}")
    print(f"   Min: ${y_raw_series.min():.2f}")
    print(f"   Max: ${y_raw_series.max():.2f}")

    print("\n3) Preprocessing target (clip)...")
    df, clip_info = preprocess_target(df, cfg)
    print(f"   ✓ Clipping: q_low=${clip_info['q_low_value']:.2f}, q_high=${clip_info['q_high_value']:.2f}")

    print("\n4) Scaling and Temporal split...")
    features_scaled, target_scaled, f_scaler, t_scaler, split_idx = scale_data(df, cfg)

    print("   Creating sequences...")
    X, y = create_sequences(features_scaled, target_scaled, cfg.lookback)

    n_train_seq = int(len(y) * cfg.train_ratio)
    X_train, y_train_seq = X[:n_train_seq], y[:n_train_seq]
    X_test, y_test_seq = X[n_train_seq:], y[n_train_seq:]

    print(f"   ✓ Train: {len(y_train_seq)} sequences ({cfg.train_ratio * 100:.0f}%)")
    print(f"   ✓ Test: {len(y_test_seq)} sequences ({(1 - cfg.train_ratio) * 100:.0f}%)")

    # Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_seq.reshape(-1, 1))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

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

    print("\n7) Evaluating metrics on FULL TEST SET...")
    y_pred_scaled = predict(model, X_test_tensor, cfg.device)

    y_pred_raw = t_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_raw = t_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # Create Series for full test set (aligned with dates)
    dates_all = df[cfg.date_col].iloc[cfg.lookback:]
    dates_test = dates_all.iloc[n_train_seq:n_train_seq + len(y_pred_raw)].values

    y_pred_full_series = pd.Series(y_pred_raw, index=dates_test, name="y_pred")
    y_test_full_series = pd.Series(y_test_raw, index=dates_test, name="y_true")

    # Extract last 7 days for visualization
    y_pred_eval = y_pred_raw[-cfg.n_days_eval:]
    y_test_eval_series = y_test_full_series.iloc[-cfg.n_days_eval:]

    metrics = compute_metrics(y_test_full_series.values, y_pred_full_series.values)
    print(f"   ✓ MAE: ${metrics['mae']:.2f}")
    print(f"   ✓ RMSE: ${metrics['rmse']:.2f}")

    predictions_df = pd.DataFrame({
        "date": dates_test.astype(str),
        "y_true": y_test_full_series.values,
        "y_pred": y_pred_full_series.values,
        "residual": y_test_full_series.values - y_pred_full_series.values,
        "abs_error": np.abs(y_test_full_series.values - y_pred_full_series.values),
    })

    metadata = {
        "csv_path": cfg.csv_path,
        "n_total_points": len(df),
        "n_train_seq": len(y_train_seq),
        "n_test_seq": len(y_test_seq),
        "n_visualization_window": cfg.n_days_eval,
        "train_ratio": cfg.train_ratio,
        "lookback": cfg.lookback,
        "data_stats": {
            "mean_daily_expense": float(y_raw_series.mean()),
            "max_daily_expense": float(y_raw_series.max()),
            "min_daily_expense": float(y_raw_series.min()),
            "std_daily_expense": float(y_raw_series.std()),
        },
        "clip_outliers": cfg.clip_outliers,
        "clip_info": clip_info,
        "hyperparameters": {
            "lstm_units": cfg.lstm_units,
            "n_lstm_layers": cfg.n_lstm_layers,
            "dense_units": cfg.dense_units,
            "dropout_rate": cfg.dropout_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate
        },
        "metrics_evaluated_on": "full_test_set",
        "metrics": metrics,
    }

    print("\n8) Saving artifacts...")
    save_artifacts(cfg, model, {"feature_scaler": f_scaler, "target_scaler": t_scaler},
                   metrics, predictions_df, metadata)

    print("\n9) Generating plots...")
    plot_eda(cfg, y_raw_series)
    plot_series_with_split(cfg, y_raw_series, split_idx)
    plot_training_history(cfg, history)

    # Plot BOTH: full test + zoomed N days
    plot_forecast_vs_actual(cfg, y_test_full_series, y_pred_raw)
    plot_forecast_vs_actual(cfg, y_test_full_series, y_pred_raw, y_test_eval_series, y_pred_eval)

    plot_residuals(cfg, y_test_full_series, y_pred_raw)
    plot_residuals(cfg, y_test_full_series, y_pred_raw, y_test_eval_series, y_pred_eval)

    plot_metrics(cfg, metrics)

    print("\n" + "=" * 70)
    print("  Pipeline completed successfully!".center(70))
    print("=" * 70)
    print(f"- Artifacts: {cfg.output_dir}")
    print(f"- Plots: {cfg.plots_dir}")
    print("\nGenerated plots:")
    print("  00_eda_boxplot.png")
    print("  01_series_with_split.png")
    print("  01b_training_history.png")
    print("  02_forecast_vs_actual_full.png (metrics computed on this)")
    print("  02_forecast_vs_actual_7days.png (zoomed visualization)")
    print("  03_residuals_full.png")
    print("  03_residuals_7days.png")
    print("  04_metrics_bar_chart.png")
    print("  05_metrics_table.png")
    print()


if __name__ == "__main__":
    main()