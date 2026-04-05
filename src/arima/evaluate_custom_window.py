import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
PREDICTIONS_PATH = "outputs/arima_daily_v2/predictions.csv"
METADATA_PATH = "outputs/arima_daily_v2/metadata.json"
OUTPUT_DIR = "outputs/arima_evaluation_custom"
PLOTS_DIR = "outputs/arima_evaluation_custom/plots"

N_DAYS_EVAL = 20  # 7, 14, 20, 30


# =========================
# Utils
# =========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate MAE, RMSE, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def plot_forecast_vs_actual(y_test_series: pd.Series, y_pred_raw: np.ndarray, n_days: int):
    """Plot forecast vs actual."""
    plt.figure(figsize=(14, 6))
    plt.plot(
        y_test_series.index,
        y_test_series.values,
        marker="o",
        label="Actual (Test)",
        linewidth=2,
        markersize=8,
        color="blue",
    )
    plt.plot(
        y_test_series.index,
        y_pred_raw,
        marker="s",
        label="ARIMA Forecast (1-step rolling)",
        linewidth=2,
        markersize=8,
        color="orange",
        alpha=0.9,
    )
    plt.title(f"ARIMA Forecast vs Actual (Last {n_days} Days)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Amount ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"01_forecast_vs_actual_{n_days}days.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ 01_forecast_vs_actual_{n_days}days.png")


def plot_residuals(y_test_series: pd.Series, y_pred_raw: np.ndarray, n_days: int):
    """Plot residuals."""
    residuals = y_test_series.values - y_pred_raw

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(residuals)), residuals,
            color=["green" if r >= 0 else "red" for r in residuals], alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    plt.title(f"Residuals (Actual - Forecast) - {n_days} Days",
              fontsize=14, fontweight="bold")
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Residual ($)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"02_residuals_{n_days}days.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ 02_residuals_{n_days}days.png")


def plot_metrics(metrics: dict, n_days: int):
    """Plot metrics."""
    mae = metrics['mae']
    rmse = metrics['rmse']
    mape = metrics['mape']

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ['MAE\n($)', 'RMSE\n($)', 'MAPE\n(%)']
    metrics_values = [mae, rmse, mape]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
    ax.set_title(f'ARIMA Model Evaluation Metrics ({n_days} Days)',
                 fontsize=14, fontweight="bold")
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        height = bar.get_height()
        label = f'{value:.2f}%' if i == 2 else f'${value:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2., height, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"03_metrics_{n_days}days.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ 03_metrics_{n_days}days.png")


def main():
    ensure_dirs()

    print("\n" + "=" * 70)
    print(f"Custom Window Evaluation ({N_DAYS_EVAL} Days)".center(70))
    print("=" * 70 + "\n")

    # Load metadata for info only
    print("1) Loading metadata info...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"   ✓ Model: ARIMA{metadata['selected_order']}")
    print(f"   ✓ Seasonal: SARIMA{metadata['selected_seasonal_order']}")

    # Load predictions
    print(f"\n2) Loading predictions from CSV...")
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])

    print(f"   ✓ Loaded {len(predictions_df)} predictions")

    # Extract last N days
    print(f"\n3) Extracting last {N_DAYS_EVAL} days from test set...")
    predictions_eval = predictions_df.iloc[-N_DAYS_EVAL:].copy()

    print(f"   ✓ Date range: {predictions_eval['date'].min().date()} to {predictions_eval['date'].max().date()}")

    # Convert to Series
    y_test_series = pd.Series(
        predictions_eval['y_true'].values,
        index=predictions_eval['date'].values,
        name="y_true"
    )
    y_pred_raw = predictions_eval['y_pred'].values

    # Compute metrics
    print(f"\n4) Evaluating metrics on {N_DAYS_EVAL} days...")
    metrics = compute_metrics(y_test_series.values, y_pred_raw)
    print(f"   ✓ MAE: ${metrics['mae']:.2f}")
    print(f"   ✓ RMSE: ${metrics['rmse']:.2f}")
    print(f"   ✓ MAPE: {metrics['mape']:.2f}%")

    # Save metrics
    print(f"\n5) Saving results...")
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{N_DAYS_EVAL}days.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "n_days": N_DAYS_EVAL,
            "date_range": f"{predictions_eval['date'].min().date()} to {predictions_eval['date'].max().date()}",
            "metrics": metrics
        }, f, indent=2)
    print(f"   ✓ Metrics saved: {metrics_path}")

    # Save predictions
    preds_path = os.path.join(OUTPUT_DIR, f"predictions_{N_DAYS_EVAL}days.csv")
    predictions_eval.to_csv(preds_path, index=False)
    print(f"   ✓ Predictions saved: {preds_path}")

    # Plot
    print(f"\n6) Generating plots...")
    plot_forecast_vs_actual(y_test_series, y_pred_raw, N_DAYS_EVAL)
    plot_residuals(y_test_series, y_pred_raw, N_DAYS_EVAL)
    plot_metrics(metrics, N_DAYS_EVAL)

    # Summary table
    print(f"\n7) Summary (last {N_DAYS_EVAL} days):")
    summary_df = pd.DataFrame({
        "Date": predictions_eval['date'].dt.strftime("%Y-%m-%d"),
        "Actual": predictions_eval['y_true'].round(2),
        "Forecast": predictions_eval['y_pred'].round(2),
        "Error": predictions_eval['residual'].round(2),
        "Abs Error": predictions_eval['abs_error'].round(2),
    })
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  Custom evaluation completed!".center(70))
    print("=" * 70)
    print(f"- Results: {OUTPUT_DIR}")
    print(f"- Plots: {PLOTS_DIR}")
    print()


if __name__ == "__main__":
    main()