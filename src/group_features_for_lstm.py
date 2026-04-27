import pandas as pd

# Load data
transactions = pd.read_csv("data/transactions_data.csv")

# Keep only selected clients
selected_clients = [1428, 1439, 473, 397, 517, 897, 618, 1494, 955, 1744]
#selected_clients = [1428]
selected_clients = [1098]
transactions = transactions[transactions["client_id"].isin(selected_clients)].copy()

# Clean amount
transactions["amount"] = (
    transactions["amount"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# Parse datetime
transactions["date"] = pd.to_datetime(transactions["date"])
transactions["date_day"] = transactions["date"].dt.date

# Daily aggregation per client
daily = (
    transactions
    .groupby(["client_id", "date_day"])
    .agg(
        daily_spend=("amount", "sum"),
        tx_count=("amount", "size"),
        unique_mcc=("mcc", "nunique"),
        unique_merchants=("merchant_id", "nunique"),
        online_ratio=("use_chip", lambda x: (x == "Online Transaction").mean()),
        chip_ratio=("use_chip", lambda x: (x == "Chip Transaction").mean()),
    )
    .reset_index()
)

# Time features
daily["date_day"] = pd.to_datetime(daily["date_day"])
daily["day_of_week"] = daily["date_day"].dt.weekday
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
daily["month"] = daily["date_day"].dt.month
daily["day_of_month"] = daily["date_day"].dt.day

# Save dataset
daily.to_csv("data/lstm_daily_dataset.csv", index=False)
print("Saved lstm_daily_dataset.csv with shape:", daily.shape)