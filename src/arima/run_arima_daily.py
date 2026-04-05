import os
import json
import warnings
from dataclasses import dataclass
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
@dataclass
class Config:
    output_dir: str = "outputs/arima_daily_v2"
    plots_dir: str = "outputs/arima_daily_v2/plots"
    csv_path: str = "../data/processed_expense.csv"
    date_col: str = "Date"
    amount_col: str = "Expense"

    # daily frequency
    freq: str = "D"

    # split
    train_ratio: float = 0.8

    # evaluation: last N days
    n_days_eval: int = 7

    # outlier handling (recommended for daily data)
    clip_outliers: bool = True
    clip_q_low: float = 0.01
    clip_q_high: float = 0.99

    # transformation
    use_log1p: bool = True

    # model
    seasonal: bool = True
    m: int = 7  # weekly seasonality in daily data


# =========================
# Utils
# =========================
def ensure_dirs(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)


def load_preprocessed_daily(cfg: Config) -> pd.DataFrame:
    """Load preprocessed CSV (no aggregation)."""
    df = pd.read_csv(cfg.csv_path)

    # parse columns
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df[cfg.amount_col] = pd.to_numeric(df[cfg.amount_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col, cfg.amount_col]).copy()

    # Sort by date
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    # Create series with date index
    daily = df.set_index(cfg.date_col)[[cfg.amount_col]].copy()
    daily.columns = ["y"]
    daily.index.name = "date"

    # Fill missing dates with 0 (days with no recorded expense)
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq=cfg.freq)
    daily = daily.reindex(full_idx, fill_value=0.0)

    return daily


def preprocess_target(daily: pd.DataFrame, cfg: Config):
    """Apply clipping and log transform."""
    y = daily["y"].copy()

    # clip outliers
    clip_info = {}
    if cfg.clip_outliers:
        q_low = y.quantile(cfg.clip_q_low)
        q_high = y.quantile(cfg.clip_q_high)
        y = y.clip(lower=q_low, upper=q_high)
        clip_info = {"q_low_value": float(q_low), "q_high_value": float(q_high)}
    else:
        clip_info = {"q_low_value": None, "q_high_value": None}

    # log1p transform
    if cfg.use_log1p:
        y_model = np.log1p(y)
    else:
        y_model = y.copy()

    return y, y_model, clip_info


def temporal_split(y_model: pd.Series, train_ratio: float = 0.8):
    """Temporal split (80/20)."""
    n = len(y_model)
    split_idx = int(n * train_ratio)
    y_train = y_model.iloc[:split_idx].copy()
    y_test_full = y_model.iloc[split_idx:].copy()
    return y_train, y_test_full, split_idx


def inverse_transform(pred_model: np.ndarray, cfg: Config):
    """Inverse log1p if applied."""
    if cfg.use_log1p:
        return np.expm1(pred_model)
    return pred_model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """MAE, RMSE, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE robust to zeros
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def rolling_forecast_1step(model, y_test: pd.Series, cfg: Config):
    """
    Rolling forecast 1-step: predict full test set day by day.
    Returns predictions in original scale for entire test set.
    """
    y_pred_raw_list = []
    model_rolling = deepcopy(model)

    print("   Rolling forecast 1-step (full test set):")
    for i in range(len(y_test)):
        # predict 1 step ahead
        forecast_step = model_rolling.predict(n_periods=1)
        forecast_value = forecast_step[0] if isinstance(forecast_step, np.ndarray) else forecast_step.iloc[0]

        # inverse transform to original scale
        pred_raw = inverse_transform(np.array([forecast_value]), cfg)[0]
        y_pred_raw_list.append(pred_raw)

        # update with observed real data (walk-forward)
        model_rolling.update(y_test.iloc[i : i + 1])

        if (i + 1) % max(1, len(y_test) // 5) == 0:
            print(f"     {i + 1}/{len(y_test)} forecasted")

    y_pred_raw = np.array(y_pred_raw_list)
    return y_pred_raw



def save_artifacts(cfg: Config, model, metrics: dict, predictions_df: pd.DataFrame, metadata: dict):
    """Save model, metrics, predictions and metadata."""
    model_path = os.path.join(cfg.output_dir, "arima_model.pkl")
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    preds_path = os.path.join(cfg.output_dir, "predictions.csv")
    meta_path = os.path.join(cfg.output_dir, "metadata.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_df.to_csv(preds_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Model saved: {model_path}")
    print(f"   Metrics saved: {metrics_path}")
    print(f"   Predictions saved: {preds_path}")
    print(f"   Metadata saved: {meta_path}")


def plot_eda(cfg: Config, y_raw: pd.Series):
    """EDA: Boxplot with statistical annotations."""
    print("   Generating EDA plots...")
    plt.figure(figsize=(8, 5))

    # Create boxplot
    box = plt.boxplot(y_raw.values, labels=["Expenses"])

    # Calculate stats
    min_val = y_raw.min()
    max_val = y_raw.max()
    median = y_raw.median()
    q1 = y_raw.quantile(0.25)
    q3 = y_raw.quantile(0.75)

    # Add annotations
    plt.text(1.1, min_val, f"Min: {min_val:.1f}", fontsize=10, color="blue")
    plt.text(1.1, max_val, f"Max: {max_val:.1f}", fontsize=10, color="blue")
    plt.text(1.1, median, f"Median: {median:.1f}", fontsize=10, color="orange")
    plt.text(1.1, q1, f"Q1: {q1:.1f}", fontsize=10, color="green")
    plt.text(1.1, q3, f"Q3: {q3:.1f}", fontsize=10, color="green")

    # Titles and labels
    plt.title("Exploratory Data Analysis: Daily Expenses Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Daily Expenses")
    plt.ylabel("Expense ($)")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "00_eda_boxplot.png"), dpi=150)
    plt.close()

    print("     ✓ 00_eda_boxplot.png")


def plot_series_with_split(cfg: Config, y_raw: pd.Series, split_idx: int):
    """Plot 1: Complete series + train/test split."""
    print("   Generating main plots...")

    plt.figure(figsize=(14, 5))
    plt.plot(y_raw.index, y_raw.values, label="Daily Expense (raw)", linewidth=1.5, alpha=0.8)
    plt.axvline(y_raw.index[split_idx], color="red", linestyle="--", linewidth=2,
                label="Train/Test split (80/20)")
    plt.title("Daily Expense Time Series (Complete History)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Amount ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "01_series_with_split.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 01_series_with_split.png")


def plot_forecast_vs_actual(cfg: Config, y_test_series: pd.Series, y_pred_raw: np.ndarray,
                            y_test_eval_series: pd.Series = None, y_pred_eval_raw: np.ndarray = None):
    """
    Plot 2: Forecast vs Actual.
    - If y_test_eval provided: plot only last 7 days: cleaner visualization
    - If not: plot full test set
    """
    # Use evaluation set if provided: last 7 days for clarity
    if y_test_eval_series is not None:
        y_test_plot = y_test_eval_series
        y_pred_plot = y_pred_eval_raw
        title_suffix = f"(Last {len(y_test_eval_series)} Days - Zoomed View)"
    else:
        y_test_plot = y_test_series
        y_pred_plot = y_pred_raw
        title_suffix = f"(Full Test Set: {len(y_test_series)} Days)"

    plt.figure(figsize=(14, 6))
    plt.plot(
        y_test_plot.index,
        y_test_plot.values,
        marker="o",
        label="Actual (Test)",
        linewidth=2,
        markersize=8,
        color="blue",
    )
    plt.plot(
        y_test_plot.index,
        y_pred_plot,
        marker="s",
        label="ARIMA Forecast (1-step rolling)",
        linewidth=2,
        markersize=8,
        color="orange",
        alpha=0.9,
    )
    plt.title(f"ARIMA Forecast vs Actual {title_suffix}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Amount ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if y_test_eval_series is not None:
        plt.savefig(os.path.join(cfg.plots_dir, "02_forecast_vs_actual_7days.png"),
                    dpi=150, bbox_inches="tight")
        print("     ✓ 02_forecast_vs_actual_7days.png (zoomed view)")
    else:
        plt.savefig(os.path.join(cfg.plots_dir, "02_forecast_vs_actual_full.png"),
                    dpi=150, bbox_inches="tight")
        print("     ✓ 02_forecast_vs_actual_full.png (full test)")
    plt.close()


def plot_residuals(cfg: Config, y_test_series: pd.Series, y_pred_raw: np.ndarray,
                   y_test_eval_series: pd.Series = None, y_pred_eval_raw: np.ndarray = None):
    """
    Plot 3: Residuals.
    - If y_test_eval provided: plot only last 7 days
    - If not: plot full test set
    """
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
    plt.title(f"Residuals (Actual - Forecast) {title_suffix}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Residual ($)", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if y_test_eval_series is not None:
        plt.savefig(os.path.join(cfg.plots_dir, "03_residuals_7days.png"),
                    dpi=150, bbox_inches="tight")
        print("     ✓ 03_residuals_7days.png (zoomed view)")
    else:
        plt.savefig(os.path.join(cfg.plots_dir, "03_residuals_full.png"),
                    dpi=150, bbox_inches="tight")
        print("     ✓ 03_residuals_full.png (full test)")
    plt.close()

def plot_metrics(cfg: Config, metrics: dict):
    """Plot final metrics: Table 1 (MAE, RMSE, MAPE)."""
    print("   Generating metrics visualization...")

    mae = metrics['mae']
    rmse = metrics['rmse']
    mape = metrics['mape']

    # Metrics bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ['MAE\n($)', 'RMSE\n($)', 'MAPE\n(%)']
    metrics_values = [mae, rmse, mape]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
    ax.set_title('ARIMA Model Evaluation Metrics (Test Set)',
                 fontsize=14, fontweight="bold")
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        height = bar.get_height()
        label = f'{value:.2f}%' if i == 2 else f'${value:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2., height, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "04_metrics_bar_chart.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 04_metrics_bar_chart.png")

    # Metrics summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['MAE', f'${mae:.2f}', f'Average absolute error per day'],
        ['RMSE', f'${rmse:.2f}', f'Root Mean Squared Error (penalizes large errors)'],
        ['MAPE', f'{mape:.2f}%', f'Mean Absolute Percentage Error'],
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.15, 0.15, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else '#ffffff')

    plt.title('ARIMA Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.plots_dir, "05_metrics_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("     ✓ 05_metrics_table.png")


def main():
    cfg = Config()
    ensure_dirs(cfg)

    print("\n" + "=" * 70)
    print("ARIMA Daily Forecasting Pipeline (Preprocessed Data)".center(70))
    print("=" * 70 + "\n")

    print("1) Loading preprocessed daily data...")
    daily = load_preprocessed_daily(cfg)
    print(f"   ✓ Loaded {len(daily)} days of data")
    print(f"   Date range: {daily.index.min().date()} to {daily.index.max().date()}")

    print("\n2) Exploratory Data Analysis (EDA):")
    print(f"   Mean daily expense: ${daily['y'].mean():.2f}")
    print(f"   Median daily expense: ${daily['y'].median():.2f}")
    print(f"   Std Dev: ${daily['y'].std():.2f}")
    print(f"   Min: ${daily['y'].min():.2f}")
    print(f"   Max: ${daily['y'].max():.2f}")

    print("\n3) Preprocessing target (clip/log)...")
    y_raw, y_model, clip_info = preprocess_target(daily, cfg)
    print(f"   ✓ Clipping: q_low=${clip_info['q_low_value']:.2f}, q_high=${clip_info['q_high_value']:.2f}")
    print(f"   ✓ Log1p transform applied: {cfg.use_log1p}")

    print("\n4) Temporal split (80% train / 20% test)...")
    n = len(y_model)
    split_idx = int(n * cfg.train_ratio)

    y_train = y_model.iloc[:split_idx].copy()
    y_test_full = y_model.iloc[split_idx:].copy()

    print(f"   ✓ Train: {len(y_train)} days ({cfg.train_ratio * 100:.0f}%)")
    print(f"   ✓ Test: {len(y_test_full)} days ({(1 - cfg.train_ratio) * 100:.0f}%)")

    # Evaluation subset: last 7 days for visualization
    y_test_eval = y_test_full.iloc[-cfg.n_days_eval:]
    print(f"   ✓ Evaluation window: last {cfg.n_days_eval} days (for visualization)")
    print(f"   ✓ Eval date range: {y_test_eval.index.min().date()} to {y_test_eval.index.max().date()}")

    print("\n5) Training ARIMA/SARIMA with auto_arima...")
    model = auto_arima(
        y_train,
        seasonal=cfg.seasonal,
        m=cfg.m,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        start_P=0,
        start_Q=0,
        max_P=2,
        max_Q=2,
        d=None,
        D=None,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
    )

    print(f"   ✓ Selected ARIMA order: {model.order}")
    print(f"   ✓ Selected Seasonal order: {model.seasonal_order}")
    print(f"   ✓ AIC: {model.aic():.2f}")

    print("\n6) Rolling forecast (1-step) on FULL TEST SET...")
    y_pred_raw = rolling_forecast_1step(model, y_test_full, cfg)
    y_test_full_raw = inverse_transform(y_test_full.values, cfg)

    # Create Series for full test set
    y_pred_full_series = pd.Series(y_pred_raw, index=y_test_full.index, name="y_pred")
    y_test_full_series = pd.Series(y_test_full_raw, index=y_test_full.index, name="y_true")

    # Extract last 7 days for visualization
    y_pred_eval = y_pred_raw[-cfg.n_days_eval:]
    y_test_eval_raw = inverse_transform(y_test_eval.values, cfg)
    y_test_eval_series = pd.Series(y_test_eval_raw, index=y_test_eval.index, name="y_true")

    print("\n7) Evaluating metrics on FULL TEST SET (20%)...")
    metrics = compute_metrics(y_test_full_series.values, y_pred_full_series.values)
    print(f"   ✓ MAE: ${metrics['mae']:.2f}")
    print(f"   ✓ RMSE: ${metrics['rmse']:.2f}")
    print(f"   ✓ MAPE: {metrics['mape']:.2f}%")

    # predictions table (full test)
    predictions_df = pd.DataFrame(
        {
            "date": y_test_full.index.astype(str),
            "y_true": y_test_full_series.values,
            "y_pred": y_pred_full_series.values,
            "residual": y_test_full_series.values - y_pred_full_series.values,
            "abs_error": np.abs(y_test_full_series.values - y_pred_full_series.values),
        }
    )

    metadata = {
        "csv_path": cfg.csv_path,
        "n_total_points": int(len(y_raw)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test_full)),
        "n_visualization_window": cfg.n_days_eval,
        "train_ratio": cfg.train_ratio,
        "freq": cfg.freq,
        "data_stats": {
            "mean_daily_expense": float(y_raw.mean()),
            "max_daily_expense": float(y_raw.max()),
            "min_daily_expense": float(y_raw.min()),
            "std_daily_expense": float(y_raw.std()),
        },
        "clip_outliers": cfg.clip_outliers,
        "clip_q_low": cfg.clip_q_low,
        "clip_q_high": cfg.clip_q_high,
        "clip_info": clip_info,
        "use_log1p": cfg.use_log1p,
        "seasonal": cfg.seasonal,
        "m": cfg.m,
        "selected_order": model.order,
        "selected_seasonal_order": model.seasonal_order,
        "forecast_method": "rolling_1step",
        "metrics_evaluated_on": "full_test_set",
        "metrics": metrics,
    }

    print("\n8) Saving artifacts...")
    save_artifacts(cfg, model, metrics, predictions_df, metadata)

    print("\n9) Generating plots...")
    plot_eda(cfg, y_raw)
    plot_series_with_split(cfg, y_raw, split_idx)

    # Plot BOTH: full test + zoomed 7 days
    plot_forecast_vs_actual(cfg, y_test_full_series, y_pred_raw)  # Full
    plot_forecast_vs_actual(cfg, y_test_full_series, y_pred_raw,
                            y_test_eval_series, y_pred_eval)  # Zoomed 7 days

    plot_residuals(cfg, y_test_full_series, y_pred_raw)  # Full
    plot_residuals(cfg, y_test_full_series, y_pred_raw,
                   y_test_eval_series, y_pred_eval)  # Zoomed 7 days

    plot_metrics(cfg, metrics)

    print("\n" + "=" * 70)
    print("  Pipeline completed successfully!".center(70))
    print("=" * 70)
    print(f"- Artifacts: {cfg.output_dir}")
    print(f"- Plots: {cfg.plots_dir}")
    print("\nGenerated plots:")
    print("  00_eda_boxplot.png")
    print("  01_series_with_split.png")
    print("  02_forecast_vs_actual_full.png (metrics computed on this)")
    print("  02_forecast_vs_actual_7days.png (zoomed visualization)")
    print("  03_residuals_full.png")
    print("  03_residuals_7days.png")
    print("  04_metrics_bar_chart.png")
    print("  05_metrics_table.png")
    print()

if __name__ == "__main__":
    main()