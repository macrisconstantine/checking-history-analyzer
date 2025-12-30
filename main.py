import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
INPUT_FILE = "your_checking_account_statement.csv"  
EXPORT_MONTHLY_CSV = True
# ----------------------------

# Load data
df = pd.read_csv(INPUT_FILE)

# Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Parse dates
df["transaction_date"] = pd.to_datetime(df["transaction_date"]).round(2)

# Ensure numeric amounts
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"]).round(2)

# Money in / out
df["money_in"] = df.apply(
    lambda r: r.transaction_amount if r.transaction_type.lower() == "credit" else 0,
    axis=1
)
df["money_out"] = df.apply(
    lambda r: r.transaction_amount if r.transaction_type.lower() == "debit" else 0,
    axis=1
)

# Monthly grouping
df["year_month"] = df["transaction_date"].dt.to_period("M")

monthly = df.groupby("year_month").agg(
    total_in=("money_in", "sum"),
    total_out=("money_out", "sum"),
    transactions=("transaction_amount", "count")
).reset_index().round(2)

monthly["net"] = (monthly["total_in"] - monthly["total_out"]).round(2)

# ---------- OVERALL STATS ----------
total_in = df["money_in"].sum()
total_out = df["money_out"].sum()
net = total_in - total_out

# Compute average monthly inflow and outflow
avg_monthly_in = monthly["total_in"].mean()
avg_monthly_out = monthly["total_out"].mean()
avg_monthly_net = monthly["net"].mean()

largest_deposit = df[df.money_in > 0]["money_in"].max()
largest_withdrawal = df[df.money_out > 0]["money_out"].max()

# Trend (simple linear direction)
trend = "increasing" if monthly["net"].iloc[-1] > monthly["net"].iloc[0] else "decreasing"

# ---------- OUTPUT ----------
print("\n===== FINANCIAL SUMMARY =====\n")

print(f"Date range: {df.transaction_date.min().date()} → {df.transaction_date.max().date()}")
print(f"Total deposited: {total_in:,.2f}")
print(f"Total spent: {total_out:,.2f}")
print(f"Net cash flow: {net:,.2f}\n")

print("----- Monthly Averages -----")
print(f"Average monthly deposits: {avg_monthly_in:,.2f}")
print(f"Average monthly spending: {avg_monthly_out:,.2f}")
print(f"Average monthly net: {avg_monthly_net:,.2f}\n")

print("----- Extremes -----")
print(f"Largest single deposit: {largest_deposit:,.2f}")
print(f"Largest single withdrawal: {largest_withdrawal:,.2f}\n")

print("----- Trend -----")
print(f"Overall net cash flow trend is {trend}\n")

print("----- Monthly Breakdown -----")
print(monthly.to_string(index=False))

# Export CSV if desired
if EXPORT_MONTHLY_CSV:
    out_file = Path(INPUT_FILE).with_name("monthly_summary.csv")
    monthly.to_csv(out_file, index=False)
    print(f"\nMonthly summary exported to: {out_file}")
