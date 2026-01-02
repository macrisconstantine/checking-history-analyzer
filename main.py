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

# Map specific column to 'description' and parse dates
df = df.rename(columns={'transaction_description': 'description'})
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format='%m/%d/%y')

# Clean numeric data & Round to 2
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"]).abs().round(2)

# Categorization
df["category"] = df["description"].apply(categorize_description)
df["year_month"] = df["transaction_date"].dt.to_period("M")

# Split Money In/Out
df["money_in"] = df.apply(lambda r: r.transaction_amount if r.transaction_type.lower() == "credit" else 0, axis=1).round(2)
df["money_out"] = df.apply(lambda r: r.transaction_amount if r.transaction_type.lower() == "debit" else 0, axis=1).round(2)

# ---------- ANALYSIS ----------

# Monthly Aggregation
monthly = df.groupby("year_month").agg(
    income=("money_in", "sum"),
    expenses=("money_out", "sum")
).reset_index()

# Calculate Net Flow and Round
monthly["net_flow"] = (monthly["income"] - monthly["expenses"]).round(2)
monthly["income"] = monthly["income"].round(2)
monthly["expenses"] = monthly["expenses"].round(2)

# Category Breakdown for console output
cat_totals = df.groupby("category")["money_out"].sum().sort_values(ascending=False).round(2)

# ---------- OUTPUT ----------
print("\n--- Monthly Overview ---")
print(monthly.to_string(index=False))

if EXPORT_MONTHLY_CSV:
    out_file = Path(INPUT_FILE).with_name("financial_summary.csv")
    # Ensuring the CSV also respects the 2-decimal rounding
    monthly.to_csv(out_file, index=False, float_format='%.2f')
    print(f"\nSaved rounded results to: {out_file}")