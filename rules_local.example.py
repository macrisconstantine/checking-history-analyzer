"""
Site-specific classification rules.

Copy this file to `rules_local.py` and edit it. `rules_local.py` is gitignored,
so patterns that name your employer, school, charity, landlord or bank stay on
your machine instead of going into a public repository.

`analyze.py` tries these before its own built-in rules, so they can override
anything there. That ordering is what lets a foreign-ATM pass-through rule beat
the generic "Cash (ATM)" spending rule.

Each entry is a 4-tuple:

    (bucket, category, direction, patterns)

    bucket     "income", "spending", "transfer" or "passthrough"
    category   the label shown in the report
    direction  "credit", "debit", or None to match either
    patterns   lowercase substrings matched against the transaction description

Uncomment and adapt. Anything matching no rule at all shows up in the report's
"Uncategorised" section, which is the quickest way to find what you still need.
"""

LOCAL_RULES: list[tuple] = [
    # Cash withdrawn abroad and handed to someone else is not your spending.
    # ("passthrough", "Cash out for family (foreign ATM)", "debit",
    #  ["some foreign bank", "another foreign bank"]),

    # ("spending", "Charity / giving", "debit", ["the charity's ach name"]),
    # ("spending", "Tuition",          "debit", ["the school's ach name"]),
    # ("spending", "Rent",             "debit", ["the landlord's ach name"]),

    # A savings or brokerage account you also own - money moved, not spent.
    # ("transfer", "Savings", None, ["nickname on the transfer"]),
]
