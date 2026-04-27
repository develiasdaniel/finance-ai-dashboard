import pandas as pd

# Load data
transactions = pd.read_csv("data/transactions_data.csv", sep=",")
users = pd.read_csv("data/users_data.csv", sep=",")

# Clean and convert numeric fields
users["current_age"] = pd.to_numeric(users["current_age"], errors="coerce")

# Clean income: remove $ and commas
users["yearly_income"] = (
    users["yearly_income"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# Filter users by age (19-35) and income (20k-40k)
filtered_users = users[
    (users["current_age"].between(19, 35)) &
    (users["yearly_income"].between(20000, 40000))
]

# Get filtered user IDs
user_ids = filtered_users["id"].unique()

# Filter transactions by those users
transactions_filtered = transactions[transactions["client_id"].isin(user_ids)].copy()

# Clean amount
transactions_filtered["amount"] = (
    transactions_filtered["amount"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# Convert date
transactions_filtered["date"] = pd.to_datetime(transactions_filtered["date"])

# Aggregate by user: record count and average daily spend
daily_avg = (
    transactions_filtered
    .groupby(["client_id", transactions_filtered["date"].dt.date])["amount"]
    .sum()
    .reset_index()
    .groupby("client_id")["amount"]
    .mean()
    .rename("avg_daily_spend")
)

counts = (
    transactions_filtered
    .groupby("client_id")
    .size()
    .rename("transaction_count")
)

result = pd.concat([counts, daily_avg], axis=1).reset_index()
result.to_csv("data/group_by_users_transactions.csv", index=False)
print(result)

print(result)