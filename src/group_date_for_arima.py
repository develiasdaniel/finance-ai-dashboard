import pandas as pd

# Load data
transactions = pd.read_csv("data/transactions_data.csv")

# ---- filter for ONE user ----
user_id = 1428
transactions = transactions[transactions["client_id"] == user_id].copy()

print("Rows for this user:", len(transactions))

# ---- clean amount ----
transactions["amount"] = (
    transactions["amount"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# ---- parse datetime ----
transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
transactions["date_day"] = transactions["date"].dt.date

# ---- group by day ----
grouped = (
    transactions
    .groupby(["client_id", "date_day"], as_index=False)
    .agg(
        total_amount=("amount", "sum"),
        transaction_count=("id", "count")
    )
)

print(grouped)

# ---- save ----
grouped.to_csv(f"data/output_user_{user_id}.csv", index=False)