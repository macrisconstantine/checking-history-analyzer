import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
INPUT_FILE = "your_checking_account_statement.csv" 
EXPORT_MONTHLY_CSV = True

# Updated categories based on your actual CSV data
CATEGORIES = {
    "Payroll/Income": ["payroll", "monthly interest", "wire deposit"],
    "Investments/Crypto": ["binance", "fid bkg svc", "moneyline"],
    "Education/Uni": ["georgia tech"],
    "Credit Card PMT": ["cardmember serv", "citi card"],
    "Zelle/Social": ["zelle"],
    "Government/Fees": ["secretary of s"],
    "Savings Transfer": ["360 performance"]
}

def categorize_description(desc):
    if pd.isna(desc): return "Miscellaneous"
    desc = str(desc).lower()
    for cat, keywords in CATEGORIES.items():
        if any(kw in desc for kw in keywords):
            return cat
    return "Miscellaneous"

# ---------- DATA LOADING ----------
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 1. Map your specific column to 'description'
df = df.rename(columns={'transaction_description': 'description'})

# 2. Fix Date Warning (using the format in your CSV: MM/DD/YY)
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%m/%d/%y')

# 3. Clean numeric data
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"]).abs().round(2)

# Categorization
df["category"] = df["description"].apply(categorize_description)
df["year_month"] = df["transaction_date"].dt.to_period("M")

# Split Money In/Out
df["money_in"] = df.apply(lambda r: r.transaction_amount if r.transaction_type.lower() == "credit" else 0, axis=1)
df["money_out"] = df.apply(lambda r: r.transaction_amount if r.transaction_type.lower() == "debit" else 0, axis=1)

# ---------- ANALYSIS ----------

# Monthly Aggregation
monthly = df.groupby("year_month").agg(
    income=("money_in", "sum"),
    expenses=("money_out", "sum"),
    net_flow=("transaction_amount", lambda x: (df.loc[x.index, "money_in"] - df.loc[x.index, "money_out"]).sum())
).reset_index()

# Category Breakdown
cat_totals = df.groupby("category")["money_out"].sum().sort_values(ascending=False)

# Specific Zelle Insight (Who are you paying/receiving from most?)
zelle_activity = df[df['description'].str.contains('Zelle', case=False, na=False)]

# ---------- OUTPUT ----------
print("\n===== ANALYSIS COMPLETE =====")
print(f"Tracking period: {df.transaction_date.min().date()} to {df.transaction_date.max().date()}")

print("\n--- Spending by Category ---")
print(cat_totals.to_string())

print("\n--- Zelle Traffic ---")
if not zelle_activity.empty:
    print(zelle_activity[['transaction_date', 'description', 'transaction_amount']].to_string(index=False))

print("\n--- Monthly Overview ---")
print(monthly.to_string(index=False))

if EXPORT_MONTHLY_CSV:
    out_file = Path(INPUT_FILE).with_name("financial_summary.csv")
    monthly.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")