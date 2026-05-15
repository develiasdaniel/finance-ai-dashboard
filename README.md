# finance-ai-dashboard

**Intelligent Financial Prediction and Lifestyle Assistant**

This repository contains Python scripts for predicting daily financial expenses using two main approaches: automatic ARIMA models and PyTorch LSTM models. The typical workflow involves running daily predictions on previously processed time series data.

---

## Main Scripts

- `src/arima/run_arima_daily.py`: Predicts daily time series using automatic ARIMA.
- `src/lstm/run_lstm_daily.py`: Predicts daily time series using a PyTorch LSTM model.

---

## 1. How to Run `run_arima_daily.py`

**Location:** `src/arima/run_arima_daily.py`

This script takes a daily expense CSV file and fits an automatic ARIMA model to predict the next day's expense. Preprocessing includes options for outlier handling and target transformation.

### Example usage:
```bash
cd src/arima
python run_arima_daily.py
```

Most settings can be adjusted in the `Config` class at the top of the script.

### Main configurable hyperparameters

- `output_dir`: Directory where results are saved.
- `csv_path`: Path to the preprocessed daily data CSV.
- `date_col`: Name of the date column.
- `amount_col`: Name of the expense column.
- `train_ratio`: Portion of data used for training.
- `n_days_eval`: Number of days at the end for evaluation.
- `clip_outliers`: (`True/False`) Enables outlier clipping.
- `clip_q_low` and `clip_q_high`: Quantiles for outlier clipping.
- `use_log1p`: (`True/False`) Enables log1p transformation of the target.
- `seasonal`: (`True/False`) Whether to use seasonality.
- `m`: Seasonality period (default 7 for weekly patterns in daily data).

---

## 2. How to Run `run_lstm_daily.py`

**Location:** `src/lstm/run_lstm_daily.py`

This script trains a multi-layer LSTM model in PyTorch for daily consumption forecasting, using several features. You can configure which client to analyze and experiment with different hyperparameters.

### Example usage:
```bash
cd src/lstm
python run_lstm_daily.py
```

All settings can be configured in the `Config` class at the beginning of the script.

### Main configurable hyperparameters

- `output_dir`: Directory where results are saved.
- `csv_path`: Path to the feature-rich dataset CSV.
- `client_id`: Client ID to analyze.
- `train_ratio`: Portion of data used for training.
- `n_days_eval`: Number of evaluation days at the end.
- `clip_outliers`, `clip_q_low`, `clip_q_high`: Outlier handling (as in ARIMA).
- `lookback`: Number of days used as input to the LSTM.
- `lstm_units`: Number of units in the LSTM layer(s).
- `n_lstm_layers`: Number of stacked LSTM layers.
- `dense_units`: Units in the final dense layer.
- `dropout_rate`, `use_dropout`: Regularization settings.
- `batch_size`: Training batch size.
- `epochs`: Number of training epochs.
- `learning_rate`: Optimizer learning rate.
- `early_stopping`: Enable early stopping if no improvement.
- `patience`: Number of epochs for early stopping patience.
- `device`: `"cuda"` for GPU or `"cpu"` (automatically set if CUDA is available).

---

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Updated requirements.txt

```
scikit-learn
pmdarima
joblib
matplotlib
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
torch
```

---

## Notes

- Data files should be located according to each script's `csv_path`.
- Models and results are saved in the output folders defined in each script.
- Set the `Config` class in each script to customize your workflow.

---