import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
INPUT_FILE = "your_checking_account_statement.csv" 
EXPORT_MONTHLY_CSV = True

# Simple keyword mapping for categorization
CATEGORIES = {
    "Housing": ["rent", "mortgage", "property tax"],
    "Food": ["starbucks", "walmart", "grocery", "uber eats", "restaurant", "mcdonalds"],
    "Utilities": ["electric", "water", "internet", "verizon", "at&t", "utility"],
    "Transport": ["gas", "shell", "uber", "lyft", "parking", "auto"],
    "Income": ["payroll", "deposit", "dividend", "interest"],
    "Entertainment": ["netflix", "spotify", "steam", "hulu", "disney+"]
}

def categorize_description(desc):
    desc = str(desc).lower()
    for cat, keywords in CATEGORIES.items():
        if any(kw in desc for kw in keywords):
            return cat
    return "Miscellaneous"

# ---------- DATA LOADING ----------
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"]).abs().round(2)

# Standardize Type
df["is_credit"] = df["transaction_type"].str.lower() == "credit"
df["category"] = df["description"].apply(categorize_description)
df["year_month"] = df["transaction_date"].dt.to_period("M")

# Split In/Out
df["money_in"] = df.apply(lambda r: r.transaction_amount if r.is_credit else 0, axis=1)
df["money_out"] = df.apply(lambda r: r.transaction_amount if not r.is_credit else 0, axis=1)

# ---------- NEW FEATURES ----------

# 1. Savings Rate Calculation
# Savings Rate = (Net / Total In) * 100
monthly = df.groupby("year_month").agg(
    total_in=("money_in", "sum"),
    total_out=("money_out", "sum")
).reset_index()

monthly["net"] = monthly["total_in"] - monthly["total_out"]
monthly["savings_rate_pct"] = (monthly["net"] / monthly["total_in"] * 100).round(2)

# 2. Category Breakdown (Where is the money going?)
spending_by_cat = df[df.money_out > 0].groupby("category")["money_out"].sum().sort_values(ascending=False)

# 3. Recurring Expense Detection (Potential Subscriptions)
# Finds items with the exact same description and amount appearing multiple times
recurring = df[df.money_out > 0].groupby(["description", "transaction_amount"]).size()
subscriptions = recurring[recurring >= 3] # Appears 3+ times in the dataset

# ---------- ADVANCED OUTPUT ----------
print("\n" + "="*40)
print("       ADVANCED FINANCIAL REPORT")
print("="*40)

print(f"\n[SAVINGS RATE] Overall: {(monthly['net'].sum() / monthly['total_in'].sum() * 100):.2f}%")
print("Top Spending Categories:")
print(spending_by_cat.head(5).to_string())

print("\n[RECURRING PAYMENTS] Possible Subscriptions:")
if not subscriptions.empty:
    print(subscriptions.to_string())
else:
    print("No clear recurring patterns detected.")

print("\n[MONTHLY PERFORMANCE]")
print(monthly[["year_month", "total_in", "total_out", "savings_rate_pct"]].to_string(index=False))

if EXPORT_MONTHLY_CSV:
    out_file = Path(INPUT_FILE).with_name("detailed_financials.csv")
    monthly.to_csv(out_file, index=False)