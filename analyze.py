#!/usr/bin/env python3
"""
checking-history-analyzer - full analysis of a Capital One 360 Checking export.

Usage
-----
    python analyze.py export.csv
    python analyze.py export.csv --by-year
    python analyze.py export.csv --by-year --calendar-years --top 20

Expects the Capital One "Download Transactions" CSV, whose columns are:
    Account Number, Transaction Description, Transaction Date,
    Transaction Type, Transaction Amount, Balance

Rows arrive newest-first and amounts are unsigned; direction comes from
Transaction Type (Credit / Debit).

Why this is more than a sum of debits
-------------------------------------
Most of the money moving through a checking account like this one is neither
income nor spending:

  * transfers to a brokerage, a savings account or a crypto exchange are money
    you still own - an internal transfer, not an expense;
  * money that arrives only to be handed straight on to someone else was never
    yours at all - a pass-through, on neither side of the ledger.

So every row is bucketed as income / spending / internal transfer /
pass-through, and only the paycheck counts as income. The four buckets are
reconciled against the statement's own net balance change, so nothing can be
quietly dropped.

Years
-----
`--by-year` splits into 12-month windows anchored on the first transaction
rather than on 1 January, so a window can line up with when the income actually
started instead of cutting the first year in half. `--calendar-years` switches
to Jan-Dec.

Local rules
-----------
Patterns that name a real employer, school, charity or bank belong in
rules_local.py, which is gitignored. See rules_local.example.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

INCOME, SPENDING, TRANSFER, PASSTHROUGH = "income", "spending", "transfer", "passthrough"

BUCKET_LABEL = {
    INCOME: "Income",
    SPENDING: "Spending",
    TRANSFER: "Internal transfer",
    PASSTHROUGH: "Pass-through",
}

# Site-specific patterns name a real employer, school, charity or bank, so they
# live in rules_local.py, which is gitignored. Copy rules_local.example.py to
# start one. Local rules are tried first, so they override anything below - which
# is what lets a foreign-ATM pass-through rule beat the generic "Cash (ATM)" rule.
def _load_local_rules() -> list[tuple]:
    """Read LOCAL_RULES out of rules_local.py sitting next to this script.

    Loaded by path rather than by import name so it works whatever the current
    directory is, and under interpreter flags such as -P and -I that strip the
    script's own directory from sys.path.
    """
    path = Path(__file__).resolve().with_name("rules_local.py")
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location("rules_local", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "LOCAL_RULES", []))


LOCAL_RULES: list[tuple] = _load_local_rules()

# (bucket, category, direction, patterns) - first match wins.
# direction is "credit", "debit", or None for either.
RULES = list(LOCAL_RULES) + [
    # Internal transfers: money that moved but is still yours.
    (TRANSFER, "Fidelity (brokerage)",   None,     ["fid bkg svc", "moneyline"]),
    (TRANSFER, "Capital One savings",    None,     ["360 performance savings"]),
    (TRANSFER, "Crypto (Binance)",       None,     ["binance", "bam trading"]),

    # The paycheck is the only thing treated as real income.
    (INCOME,   "Payroll",                "credit", ["payroll"]),

    # Pass-through: money that was never really yours, in either direction.
    (PASSTHROUGH, "Money in (Zelle)",                  "credit", ["zelle money received"]),
    (PASSTHROUGH, "Checks & wires in",                 "credit", ["check deposit", "wire deposit"]),
    (PASSTHROUGH, "Tax refunds",                       "credit", ["treas 310", "tax ref",
                                                                  "ilsttaxrfd", "state of ill"]),
    (PASSTHROUGH, "Interest",                          "credit", ["interest paid",
                                                                  "interest adjustment"]),
    (PASSTHROUGH, "Refunds & adjustments",             "credit", ["card adjustment", "ticketmaster",
                                                                  "lopez et"]),
    (PASSTHROUGH, "Other money in",                    "credit", ["instant transfer received",
                                                                  "paypal"]),

    # Spending. Specific ACH payees first, generic card purchases last.
    (SPENDING, "Taxes paid",             "debit",  ["usataxpymt"]),
    (SPENDING, "Credit card payments",   "debit",  ["cardmember serv", "citi card", "citi autopay",
                                                    "elan web", "discover e-payment"]),
    (SPENDING, "Cash (ATM)",             "debit",  ["atm withdrawal"]),
    (SPENDING, "Zelle sent",             "debit",  ["zelle money sent"]),
    (SPENDING, "City & government fees", "debit",  ["chipaypar", "city of chicago", "il sos",
                                                    "secretary of s"]),
    (SPENDING, "International transfer", "debit",  ["wise"]),
    (SPENDING, "Card purchases",         "debit",  ["card purchase"]),
    (SPENDING, "Other outflow",          "debit",  ["paper payment"]),
]

# Sub-categories applied only to "Card purchases".
CARD_SUBCATS = [
    ("Travel & airlines",  ["united", "delta", "american ", "booking", "bkg "]),
    ("Transit & parking",  ["ventra", "mbta", "keolis", "parkwhiz", "park chicago"]),
    ("Dining & bars",      ["wendy", "chipotle", "panera", "raising canes", "longhorn", "andy s",
                            "coco blu", "emporium", "cira cabra", "levy ", "pistachio", "binnys"]),
    ("Shopping & online",  ["amazon", "kohls", "eyebuydirect", "walgreens", "minuteman"]),
    ("Entertainment",      ["steam", "act cpd", "pp entertainment"]),
    ("Health",             ["minuteclinic"]),
    ("Fines & fees",       ["photoenforcement", "illinois secretary"]),
    ("Services",           ["freetaxusa"]),
]

MERCHANT_PREFIX = re.compile(r"^(?:digital|debit)\s+card\s+purchase\s*-\s*", re.I)
TRAILING_STATE = re.compile(r"\s+(?:[A-Z]{2}|US)$")
RAISE_THRESHOLD = 5.00  # dollars; smaller paycheck wobbles are withholding noise


# --------------------------------------------------------------- loading


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Account Number": str})
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"transaction_description", "transaction_date",
                "transaction_type", "transaction_amount", "balance"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected column(s): {', '.join(sorted(missing))}")

    df = df.rename(columns={"transaction_description": "description"})
    df["date"] = pd.to_datetime(df["transaction_date"], format="%m/%d/%y")
    df["amount"] = pd.to_numeric(df["transaction_amount"]).abs().round(2)
    df["balance"] = pd.to_numeric(df["balance"]).round(2)
    df["is_credit"] = df["transaction_type"].str.strip().str.lower().eq("credit")

    # The export is newest-first; reverse it to get statement order.
    df = df.iloc[::-1].reset_index(drop=True)
    df["signed"] = (df["amount"] * np.where(df["is_credit"], 1, -1)).round(2)
    df["month"] = df["date"].dt.to_period("M")
    return df


# ---------------------------------------------------------- categorizing


def classify(desc: str, is_credit: bool) -> tuple[str, str]:
    low = str(desc).lower()
    want = "credit" if is_credit else "debit"
    for bucket, category, direction, patterns in RULES:
        if direction and direction != want:
            continue
        if any(p in low for p in patterns):
            return bucket, category
    return (PASSTHROUGH, "Uncategorised in") if is_credit else (SPENDING, "Uncategorised out")


def card_subcat(desc: str) -> str:
    low = str(desc).lower()
    for name, patterns in CARD_SUBCATS:
        if any(p in low for p in patterns):
            return name
    return "Other card spend"


def clean_merchant(desc: str) -> str:
    """'Debit Card Purchase - AMAZON MARK ZD16T1842 SEATTLE WA' -> 'AMAZON MARK SEATTLE'."""
    s = MERCHANT_PREFIX.sub("", str(desc)).strip()
    s = re.sub(r"\b(?=\S*\d)\S{4,}\b", " ", s)        # order ids, terminal ids, masked accounts
    s = TRAILING_STATE.sub("", s.strip(" ,-"))
    s = re.sub(r"\s{2,}", " ", s).strip(" ,-")
    return s or str(desc)


def person(desc: str) -> str:
    """Normalise a Zelle counterparty: 'sent to ALEX J SMITH' -> 'ALEX SMITH'."""
    m = re.search(r"zelle money (?:sent to|received from)\s+(.*)", str(desc), re.I)
    if not m:
        return ""
    parts = m.group(1).upper().split()
    if not parts:
        return ""
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else parts[0]


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [classify(d, c) for d, c in zip(df["description"], df["is_credit"])]
    df["bucket"] = [b for b, _ in pairs]
    df["category"] = [c for _, c in pairs]
    df["subcategory"] = [
        card_subcat(d) if cat == "Card purchases" else cat
        for d, cat in zip(df["description"], df["category"])
    ]
    df["counterparty"] = df["description"].map(person)
    return df


def flag_reversals(df: pd.DataFrame) -> pd.DataFrame:
    """Match 'Card Adjustment (Credit)' rows to the purchase they reverse, and net both out."""
    df["is_reversal"] = False
    stop = {"CHECKING", "CARD", "ADJUSTMENT", "SIGNATURE", "CREDIT", "DEBIT", "DIGITAL", "PURCHASE"}
    adjustments = df[df["description"].str.contains("card adjustment", case=False, na=False)]
    for idx, row in adjustments.iterrows():
        tokens = set(re.findall(r"[A-Za-z]{4,}", row["description"].upper())) - stop
        window = df[(~df["is_credit"])
                    & (df["amount"] == row["amount"])
                    & ((df["date"] - row["date"]).abs() <= pd.Timedelta(days=5))
                    & (~df["is_reversal"])]
        for cand_idx, cand in window.iterrows():
            cand_tokens = set(re.findall(r"[A-Za-z]{4,}", cand["description"].upper())) - stop
            if tokens & cand_tokens:
                df.loc[[idx, cand_idx], "is_reversal"] = True
                break
    return df


def reconcile(df: pd.DataFrame) -> list[str]:
    """Every row's balance should equal the previous balance plus this row's signed amount."""
    breaks = []
    bal = df["balance"].tolist()
    sig = df["signed"].tolist()
    for i in range(1, len(df)):
        expected = round(bal[i - 1] + sig[i], 2)
        if abs(expected - bal[i]) > 0.01:
            r = df.iloc[i]
            breaks.append(f"{r['date']:%Y-%m-%d}  {r['description'][:44]} - "
                          f"expected {expected:,.2f}, got {bal[i]:,.2f}")
    return breaks


# ------------------------------------------------------- document model


@dataclass
class Block:
    kind: str
    payload: dict = field(default_factory=dict)


class Doc:
    """A renderer-agnostic report: an ordered list of blocks."""

    def __init__(self, title: str, subtitle: str = "") -> None:
        self.title = title
        self.subtitle = subtitle
        self.blocks: list[Block] = []

    def _add(self, kind: str, **payload) -> None:
        self.blocks.append(Block(kind, payload))

    def section(self, title: str, note: str = "") -> None:
        self._add("section", title=title, note=note)

    def sub(self, title: str, note: str = "") -> None:
        self._add("sub", title=title, note=note)

    def note(self, text: str) -> None:
        self._add("note", text=text)

    def stats(self, items: list[tuple[str, str, str]]) -> None:
        """items: (label, value, note)"""
        self._add("stats", items=items)

    def kv(self, pairs: list[tuple[str, str]]) -> None:
        self._add("kv", pairs=pairs)

    def table(self, headers: list[str], rows: list[list[str]], aligns: str,
              bars: list[float] | None = None, emphasise_last: bool = False) -> None:
        self._add("table", headers=headers, rows=rows, aligns=aligns,
                  bars=bars, emphasise_last=emphasise_last)

    def chart(self, labels: list[str], series: list[tuple[str, list[float]]]) -> None:
        self._add("chart", labels=labels, series=series)


# ------------------------------------------------------------ formatting


def money(x: float, sign: bool = False) -> str:
    s = f"${abs(x):,.2f}"
    if x < 0:
        return "-" + s
    return ("+" + s) if sign else s


def pct(x: float) -> str:
    return f"{x:.1f}%"


# ---------------------------------------------------------- text render


BLOCKS = "▏▎▍▌▋▊▉█"
WIDTH = 86
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def vlen(s: str) -> int:
    return len(ANSI_RE.sub("", str(s)))


def pad(s: str, w: int, align: str) -> str:
    gap = max(0, w - vlen(s))
    return (str(s) + " " * gap) if align == "l" else (" " * gap + str(s))


def text_bar(fraction: float, width: int = 20) -> str:
    if not fraction or fraction <= 0:
        return ""
    units = max(0.0, min(1.0, fraction)) * width
    full = int(units)
    out = "█" * full
    idx = int(round((units - full) * 8))
    if idx:
        out += BLOCKS[idx - 1]
    return out or "▏"


def render_text(doc: Doc) -> str:
    out: list[str] = []
    out.append("")
    out.append("╭" + "─" * (WIDTH - 2) + "╮")
    out.append("│ " + pad(doc.title.upper(), WIDTH - 4, "l") + " │")
    if doc.subtitle:
        out.append("│ " + pad(doc.subtitle, WIDTH - 4, "l") + " │")
    out.append("╰" + "─" * (WIDTH - 2) + "╯")

    for b in doc.blocks:
        p = b.payload
        if b.kind == "section":
            out.append("")
            out.append("┏" + "━" * (WIDTH - 2) + "┓")
            out.append("┃ " + pad(p["title"].upper(), WIDTH - 4, "l") + " ┃")
            out.append("┗" + "━" * (WIDTH - 2) + "┛")
            if p["note"]:
                out.append("  " + p["note"])
        elif b.kind == "sub":
            out.append("")
            label = f" {p['title']} "
            out.append("  ──" + label + "─" * max(0, WIDTH - 6 - len(label)))
            if p["note"]:
                out.append("  " + p["note"])
        elif b.kind == "note":
            out.append("  " + p["text"])
        elif b.kind == "stats":
            out.append("")
            for label, value, note in p["items"]:
                line = "  " + pad(label, 26, "l") + pad(value, 16, "r")
                if note:
                    line += "   " + note
                out.append(line)
        elif b.kind == "kv":
            out.append("")
            for k, v in p["pairs"]:
                out.append("  " + pad(k, 26, "l") + v)
        elif b.kind == "table":
            out.append("")
            headers, rows, aligns = p["headers"], p["rows"], p["aligns"]
            bars = p["bars"]
            cols = len(headers)
            widths = [max([vlen(headers[i])] + [vlen(r[i]) for r in rows]) for i in range(cols)]
            sep = "  "
            out.append("  " + sep.join(pad(headers[i], widths[i], aligns[i]) for i in range(cols)))
            out.append("  " + "─" * (sum(widths) + len(sep) * (cols - 1)))
            for n, row in enumerate(rows):
                line = "  " + sep.join(pad(row[i], widths[i], aligns[i]) for i in range(cols))
                if bars:
                    line += "  " + text_bar(bars[n])
                if p["emphasise_last"] and n == len(rows) - 1:
                    out.append("  " + "─" * (sum(widths) + len(sep) * (cols - 1)))
                out.append(line)
        elif b.kind == "chart":
            out.append("")
            labels, series = p["labels"], p["series"]
            peak = max((max(v) if v else 0) for _, v in series) or 1
            for i, lab in enumerate(labels):
                for name, vals in series:
                    out.append("  " + pad(lab if name == series[0][0] else "", 9, "l")
                               + pad(name[:8], 9, "l")
                               + pad(f"{vals[i]:,.0f}", 11, "r") + "  "
                               + text_bar(vals[i] / peak, 28))
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------- html render


CSS = """
:root{
  --bg:#fbfbf9; --fg:#1b1b19; --muted:#6f6e68; --line:#e7e4dd; --card:#fff;
  --pos:#2f7d4f; --neg:#b4452f; --accent:#3f6ea8; --accent2:#c98a3c; --bar:#cfd8e4;
  --shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#141519; --fg:#e9e7e1; --muted:#9b998f; --line:#2a2c33; --card:#1c1e24;
    --pos:#62bd87; --neg:#e3775c; --accent:#7aa8dc; --accent2:#d9a763; --bar:#343a45;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --bg:#141519; --fg:#e9e7e1; --muted:#9b998f; --line:#2a2c33; --card:#1c1e24;
  --pos:#62bd87; --neg:#e3775c; --accent:#7aa8dc; --accent2:#d9a763; --bar:#343a45;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:980px; margin:0 auto; padding:48px 16px 96px}
header.top{margin-bottom:32px; padding-bottom:24px; border-bottom:2px solid var(--line)}
h1{font-size:30px; line-height:1.2; margin:0 0 6px; letter-spacing:-.02em}
.sub{color:var(--muted); font-size:15px}
h2{
  font-size:12px; letter-spacing:.11em; text-transform:uppercase; color:var(--muted);
  margin:52px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); font-weight:600;
}
h3{font-size:14px; margin:30px 0 10px; font-weight:650; letter-spacing:-.01em}
p.note{color:var(--muted); font-size:13.5px; margin:8px 0}
.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0 6px}
.card{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:15px 16px; box-shadow:var(--shadow)}
.card .lab{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600}
.card .val{font-size:23px; margin-top:7px; letter-spacing:-.02em; font-variant-numeric:tabular-nums}
.card .nte{font-size:12.5px; color:var(--muted); margin-top:3px}
dl.kv{display:grid; grid-template-columns:auto 1fr; gap:7px 22px; margin:16px 0}
dl.kv dt{color:var(--muted); font-size:13.5px}
dl.kv dd{margin:0; font-variant-numeric:tabular-nums; font-size:13.5px}
table{width:100%; border-collapse:collapse; margin:14px 0 6px; font-size:13.5px}
th{
  text-align:left; font-size:11px; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:0 10px 7px 0; border-bottom:1px solid var(--line);
}
td{padding:7px 10px 7px 0; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums}
th.r,td.r{text-align:right}
tr:last-child td{border-bottom:none}
tr.total td{font-weight:650; border-top:2px solid var(--line); border-bottom:none}
td.pos{color:var(--pos)} td.neg{color:var(--neg)}
td.barcell{width:150px; padding-right:0}
.bar{height:7px; border-radius:4px; background:var(--accent); opacity:.85; min-width:2px}
.bar.neg{background:var(--neg)}
.chart{margin:18px 0 6px}
.legend{display:flex; gap:18px; font-size:12.5px; color:var(--muted); margin:2px 0 10px}
.legend i{display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px}
.ok{color:var(--pos); font-weight:600} .bad{color:var(--neg); font-weight:600}
footer{margin-top:64px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px}
@media print{ body{background:#fff} .card{box-shadow:none} .wrap{padding:0} }
"""


def html_num_class(cell: str) -> str:
    s = ANSI_RE.sub("", str(cell)).strip()
    if s.startswith("-$") or s.startswith("-"):
        return " neg"
    if s.startswith("+$") or s.startswith("+"):
        return " pos"
    return ""


def svg_chart(labels: list[str], series: list[tuple[str, list[float]]]) -> str:
    if not labels:
        return ""
    colors = ["var(--accent)", "var(--accent2)"]
    w, h = 940, 230
    pad_l, pad_b, pad_t = 8, 26, 10
    peak = max((max(v) if v else 0) for _, v in series) or 1
    n = len(labels)
    group = (w - pad_l * 2) / n
    bw = min(14.0, (group - 4) / max(1, len(series)))
    bars = []
    for gi, lab in enumerate(labels):
        gx = pad_l + gi * group + (group - bw * len(series)) / 2
        for si, (_, vals) in enumerate(series):
            v = max(0.0, vals[gi])
            bh = (v / peak) * (h - pad_b - pad_t)
            x = gx + si * bw
            y = h - pad_b - bh
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1.5:.1f}" height="{max(bh, 1):.1f}" '
                f'rx="2" fill="{colors[si % len(colors)]}" opacity="0.88"></rect>')
        if n <= 14 or gi % 2 == 0:
            bars.append(
                f'<text x="{gx + bw * len(series) / 2:.1f}" y="{h - 8}" font-size="9.5" '
                f'text-anchor="middle" fill="var(--muted)">{escape(lab)}</text>')
    bars.append(f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_l}" y2="{h - pad_b}" '
                f'stroke="var(--line)" stroke-width="1"></line>')
    legend = " ".join(
        f'<span><i style="background:{colors[i % len(colors)]}"></i>{escape(name)}</span>'
        for i, (name, _) in enumerate(series))
    return (f'<div class="legend">{legend}</div>'
            f'<div class="chart"><svg viewBox="0 0 {w} {h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{"".join(bars)}</svg></div>')


def render_html(doc: Doc, generated: str) -> str:
    parts: list[str] = []
    for b in doc.blocks:
        p = b.payload
        if b.kind == "section":
            parts.append(f"<h2>{escape(p['title'])}</h2>")
            if p["note"]:
                parts.append(f"<p class=note>{escape(p['note'])}</p>")
        elif b.kind == "sub":
            parts.append(f"<h3>{escape(p['title'])}</h3>")
            if p["note"]:
                parts.append(f"<p class=note>{escape(p['note'])}</p>")
        elif b.kind == "note":
            parts.append(f"<p class=note>{escape(p['text'])}</p>")
        elif b.kind == "stats":
            cards = "".join(
                f'<div class=card><div class=lab>{escape(l)}</div>'
                f'<div class="val{html_num_class(v)}">{escape(v)}</div>'
                + (f'<div class=nte>{escape(nt)}</div>' if nt else "")
                + "</div>"
                for l, v, nt in p["items"])
            parts.append(f'<div class=cards>{cards}</div>')
        elif b.kind == "kv":
            cells = []
            for k, v in p["pairs"]:
                cls = " class=ok" if str(v).startswith("✓") else (
                    " class=bad" if str(v).startswith("✗") else "")
                cells.append(f"<dt>{escape(k)}</dt><dd><span{cls}>{escape(str(v))}</span></dd>")
            parts.append(f'<dl class=kv>{"".join(cells)}</dl>')
        elif b.kind == "table":
            headers, rows, aligns, bars = p["headers"], p["rows"], p["aligns"], p["bars"]
            th = "".join(
                f'<th class="{"r" if aligns[i] == "r" else ""}">{escape(h)}</th>'
                for i, h in enumerate(headers))
            if bars:
                th += "<th></th>"
            body = []
            for n, row in enumerate(rows):
                cls = " class=total" if p["emphasise_last"] and n == len(rows) - 1 else ""
                tds = "".join(
                    f'<td class="{"r" if aligns[i] == "r" else ""}{html_num_class(c)}">'
                    f'{escape(str(c))}</td>' for i, c in enumerate(row))
                if bars:
                    frac = max(0.0, min(1.0, bars[n] or 0.0))
                    tds += (f'<td class=barcell><div class=bar '
                            f'style="width:{frac * 100:.1f}%"></div></td>')
                body.append(f"<tr{cls}>{tds}</tr>")
            parts.append(f"<table><thead><tr>{th}</tr></thead>"
                         f"<tbody>{''.join(body)}</tbody></table>")
        elif b.kind == "chart":
            parts.append(svg_chart(p["labels"], p["series"]))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(doc.title)}</title>
<style>{CSS}</style></head>
<body><div class=wrap>
<header class=top><h1>{escape(doc.title)}</h1>
<div class=sub>{escape(doc.subtitle)}</div></header>
{''.join(parts)}
<footer>Generated {escape(generated)} by checking-history-analyzer. Figures are derived
solely from the Capital One 360 Checking export; spending charged to credit cards appears
only as the payment that settled it.</footer>
</div></body></html>"""


# ------------------------------------------------------------- analysis


DAYS_PER_MONTH = 365.25 / 12


def build_doc(df: pd.DataFrame, args: argparse.Namespace, label: str | None = None,
              period: str | None = None,
              window: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> Doc:
    live = df[~df["is_reversal"]]
    inc = live[live["bucket"] == INCOME]
    spend = live[live["bucket"] == SPENDING]
    xfer = live[live["bucket"] == TRANSFER]
    pas = live[live["bucket"] == PASSTHROUGH]

    total_in = inc["amount"].sum()
    total_out = spend["amount"].sum()
    start, end = df["date"].min(), df["date"].max()
    # Length of the reporting window, not the number of calendar months it touches:
    # an anchored 11 Nov -> 10 Nov year is 12 months, not 13.
    if window:
        months = max(1.0, (window[1] - window[0]).days / DAYS_PER_MONTH)
    else:
        months = max(1.0, ((end - start).days + 1) / DAYS_PER_MONTH)
    months_label = (f"{months:.0f}" if abs(months - round(months)) < 0.15 else f"{months:.1f}")
    net = total_in - total_out

    out_x = xfer[~xfer["is_credit"]]["amount"].sum()
    in_x = xfer[xfer["is_credit"]]["amount"].sum()
    out_p = pas[~pas["is_credit"]]["amount"].sum()
    in_p = pas[pas["is_credit"]]["amount"].sum()

    title = "Checking account analysis" + (f" — {label}" if label else "")
    span = period or f"{start:%d %b %Y} – {end:%d %b %Y}"
    doc = Doc(title, f"{span} · {months_label} months")

    # ---------------------------------------------------------- headline
    doc.stats([
        ("Income", money(total_in), f"{money(total_in / months)} / month"),
        ("Spending", money(total_out), f"{money(total_out / months)} / month"),
        ("Net saved", money(net), f"{money(net / months)} / month"),
        ("Savings rate", pct(net / total_in * 100) if total_in else "—",
         "of take-home pay"),
    ])

    doc.section("Overview")
    breaks = reconcile(df)
    pairs = [
        ("Period", f"{span} ({months_label} months)"),
        ("First and last activity", f"{start:%d %b %Y} – {end:%d %b %Y}"),
        ("Transactions", f"{len(df):,} · {int(df['is_credit'].sum())} in, "
                         f"{int((~df['is_credit']).sum())} out"),
        ("Opening balance", money(df["balance"].iloc[0] - df["signed"].iloc[0])),
        ("Closing balance", money(df["balance"].iloc[-1])),
        ("Net change", money(df["signed"].sum(), sign=True)),
        ("Reconciliation",
         "✓ every row ties to the running balance" if not breaks
         else f"✗ {len(breaks)} break(s)"),
    ]
    if "account_number" in df.columns:
        pairs.insert(0, ("Account", f"····{df['account_number'].iloc[0]}"))
    rev = df[df["is_reversal"]]
    if len(rev):
        pairs.append(("Reversed holds",
                      f"{len(rev) // 2} pair(s) netted out "
                      f"({money(rev[rev['is_credit']]['amount'].sum())})"))
    doc.kv(pairs)
    for b in breaks[:5]:
        doc.note(b)

    # --------------------------------------------------------- cash flow
    doc.section("Cash flow",
                "Only the paycheck counts as income. Money moved between your own accounts, "
                "and money that passed through on someone else's behalf, are excluded from "
                "both totals.")

    def flow_table(frame: pd.DataFrame, t_out: float, t_in: float) -> None:
        rows = []
        for cat, grp in frame.groupby("category"):
            o = grp[~grp["is_credit"]]["amount"].sum()
            i = grp[grp["is_credit"]]["amount"].sum()
            rows.append([cat, money(o), money(i), money(i - o, sign=True)])
        rows.append(["Total", money(t_out), money(t_in), money(t_in - t_out, sign=True)])
        doc.table(["Category", "Out", "In", "Net"], rows, "lrrr", emphasise_last=True)

    doc.sub("Internal transfers", "Money you still own — brokerage, savings, crypto.")
    flow_table(xfer, out_x, in_x)
    doc.sub("Pass-through", "Money that was never really yours, in either direction. "
                            "Over a long enough window this should net to about zero.")
    flow_table(pas, out_p, in_p)

    computed = total_in - total_out - (out_x - in_x) + (in_p - out_p)
    actual = df["signed"].sum()
    doc.sub("Identity check",
            "The four buckets partition every row, so they must add up to the statement's "
            "own net balance change.")
    doc.table(["Component", "Amount"], [
        ["Income", money(total_in, sign=True)],
        ["Spending", money(-total_out, sign=True)],
        ["Net internal transfers", money(-(out_x - in_x), sign=True)],
        ["Net pass-through", money(in_p - out_p, sign=True)],
        ["Computed balance change", money(computed, sign=True)],
    ], "lr", emphasise_last=True)
    doc.note(("✓ matches the statement's " if abs(computed - actual) < 0.01
              else "✗ does NOT match the statement's ") + money(actual, sign=True))

    # ----------------------------------------------------------- monthly
    doc.section("Month by month")
    monthly = (live.groupby(["month", "bucket"])["amount"].sum()
               .unstack(fill_value=0.0)
               .reindex(columns=[INCOME, SPENDING, TRANSFER, PASSTHROUGH], fill_value=0.0))
    monthly["net"] = monthly[INCOME] - monthly[SPENDING]
    monthly["rate"] = np.where(monthly[INCOME] > 0, monthly["net"] / monthly[INCOME] * 100, np.nan)
    monthly["end_balance"] = df.groupby("month")["balance"].last()

    doc.chart([f"{m:%b}" for m in monthly.index.to_timestamp()],
              [("Income", monthly[INCOME].tolist()), ("Spending", monthly[SPENDING].tolist())])

    rows = []
    for m, row in monthly.iterrows():
        rows.append([str(m), money(row[INCOME]), money(row[SPENDING]),
                     money(row["net"], sign=True),
                     "—" if pd.isna(row["rate"]) else f"{row['rate']:.0f}%",
                     money(row["end_balance"])])
    doc.table(["Month", "Income", "Spending", "Net", "Save %", "End balance"], rows, "lrrrrr")
    best, worst = monthly["net"].idxmax(), monthly["net"].idxmin()
    doc.note(f"Best month {best} at {money(monthly.loc[best, 'net'])}; "
             f"worst {worst} at {money(monthly.loc[worst, 'net'])}. "
             f"{int((monthly['net'] < 0).sum())} of {len(monthly)} months in the red. "
             f"Median month: {money(monthly[INCOME].median())} in, "
             f"{money(monthly[SPENDING].median())} out.")

    if len(pas):
        rows = []
        for m, grp in pas.groupby("month"):
            i = grp[grp["is_credit"]]["amount"].sum()
            o = grp[~grp["is_credit"]]["amount"].sum()
            if i + o >= 50:
                rows.append([str(m), money(i), money(o), money(i - o, sign=True)])
        if rows:
            rows.append(["Total", money(in_p), money(out_p), money(in_p - out_p, sign=True)])
            doc.sub("Pass-through by month", "Months under $50 of activity omitted.")
            doc.table(["Month", "In", "Out", "Net"], rows, "lrrr", emphasise_last=True)

    # ------------------------------------------------------------ income
    doc.section("Income")
    inc_cat = (inc.groupby("category")["amount"].agg(["sum", "count"])
               .sort_values("sum", ascending=False))
    peak = inc_cat["sum"].max() if len(inc_cat) else 1
    doc.table(["Category", "Total", "Share", "n", "Per month"],
              [[c, money(r["sum"]), pct(r["sum"] / total_in * 100), f"{int(r['count'])}",
                money(r["sum"] / months)] for c, r in inc_cat.iterrows()],
              "lrrrr", bars=[r["sum"] / peak for _, r in inc_cat.iterrows()])

    pay = inc[inc["category"] == "Payroll"].sort_values("date")
    if len(pay) > 1:
        gaps = pay["date"].diff().dt.days.dropna()
        per_year = len(pay) / (months / 12)
        cadence = ("biweekly, 26/yr" if per_year >= 25 else
                   "semi-monthly, 24/yr" if per_year >= 21 else
                   "monthly" if per_year >= 11 else "irregular")
        doc.sub("Paycheck pattern",
                f"{len(pay)} deposits, median gap {gaps.median():.0f} days, "
                f"{per_year:.1f} per year ({cadence}).")
        changed = pay["amount"].diff().abs().fillna(RAISE_THRESHOLD + 1) > RAISE_THRESHOLD
        rows, prev = [], None
        for _, row in pay[changed].iterrows():
            rows.append([f"{row['date']:%d %b %Y}", money(row["amount"]),
                         money(row["amount"] - prev, sign=True) if prev is not None else "—"])
            prev = row["amount"]
        doc.table(["Effective", "Net per paycheck", "Change"], rows, "lrr")
        first, last = pay["amount"].iloc[0], pay["amount"].iloc[-1]
        doc.note(f"Over the period the paycheck moved {money(last - first, sign=True)} "
                 f"({(last / first - 1) * 100:+.1f}%).")

    # ---------------------------------------------------------- spending
    doc.section("Spending")
    sp_cat = (spend.groupby("category")["amount"].agg(["sum", "count"])
              .sort_values("sum", ascending=False))
    peak = sp_cat["sum"].max() if len(sp_cat) else 1
    rows = [[c, money(r["sum"]), pct(r["sum"] / total_out * 100), f"{int(r['count'])}",
             money(r["sum"] / months)] for c, r in sp_cat.iterrows()]
    bars = [r["sum"] / peak for _, r in sp_cat.iterrows()]
    rows.append(["Total", money(total_out), "100.0%", f"{len(spend)}", money(total_out / months)])
    bars.append(0.0)
    doc.table(["Category", "Total", "Share", "n", "Per month"], rows, "lrrrr",
              bars=bars, emphasise_last=True)

    cards = spend[spend["category"] == "Card purchases"]
    if len(cards):
        sub = (cards.groupby("subcategory")["amount"].agg(["sum", "count"])
               .sort_values("sum", ascending=False))
        doc.sub("Card purchases by type", "Debit card used directly, not via a credit card.")
        doc.table(["Type", "Total", "n", "Average"],
                  [[s, money(r["sum"]), f"{int(r['count'])}", money(r["sum"] / r["count"])]
                   for s, r in sub.iterrows()], "lrrr",
                  bars=[r["sum"] / sub["sum"].max() for _, r in sub.iterrows()])

    # ---------------------------------------------------- counterparties
    doc.section("Payees and sources")
    live2 = live.assign(merchant=live["description"].map(clean_merchant))
    for heading, mask in (("Money out", live2["bucket"] == SPENDING),
                          ("Money in", live2["bucket"].isin([INCOME, PASSTHROUGH]))):
        top = (live2[mask].groupby("merchant")["amount"].agg(["sum", "count"])
               .sort_values("sum", ascending=False).head(args.top))
        if not len(top):
            continue
        doc.sub(heading)
        doc.table(["Payee / source", "Total", "n"],
                  [[m[:46], money(r["sum"]), f"{int(r['count'])}"] for m, r in top.iterrows()],
                  "lrr", bars=[r["sum"] / top["sum"].max() for _, r in top.iterrows()])

    # ------------------------------------------------------------- zelle
    z = live[live["counterparty"] != ""]
    if len(z):
        doc.section("Zelle / peer-to-peer")
        rows_ = []
        for name, grp in z.groupby("counterparty"):
            rows_.append({"person": name,
                          "received": grp[grp["is_credit"]]["amount"].sum(),
                          "sent": grp[~grp["is_credit"]]["amount"].sum(),
                          "n": len(grp)})
        g = pd.DataFrame(rows_).set_index("person")
        g["net"] = (g["received"] - g["sent"]).round(2)
        g = g.sort_values("net", ascending=False)
        rows = [[p, money(r["received"]), money(r["sent"]), money(r["net"], sign=True),
                 f"{int(r['n'])}"] for p, r in g.iterrows()]
        rows.append(["Total", money(g["received"].sum()), money(g["sent"].sum()),
                     money(g["net"].sum(), sign=True), f"{int(g['n'].sum())}"])
        doc.table(["Person", "Received", "Sent", "Net", "n"], rows, "lrrrr", emphasise_last=True)

    # ---------------------------------------------------------- recurring
    doc.section("Recurring", "Payees seen in four or more distinct months.")
    rec = (live.groupby(live["description"].map(clean_merchant))
           .agg(n=("amount", "size"), months=("month", "nunique"), total=("amount", "sum"),
                median=("amount", "median"), last=("date", "max")))
    rec = rec[rec["months"] >= 4].sort_values("total", ascending=False)
    if len(rec):
        doc.table(["Payee / source", "Typical", "Months", "Total", "Last seen"],
                  [[m[:44], money(r["median"]), f"{int(r['months'])}", money(r["total"]),
                    f"{r['last']:%b %Y}"] for m, r in rec.iterrows()], "lrrrr")
        dormant = rec[rec["last"] < end - pd.Timedelta(days=90)]
        if len(dormant):
            doc.sub("Stopped or dormant", "Recurring, but nothing in the last 90 days.")
            doc.table(["Payee / source", "Last seen", "Months", "Total"],
                      [[m[:44], f"{r['last']:%b %Y}", f"{int(r['months'])}", money(r["total"])]
                       for m, r in dormant.sort_values("total", ascending=False).iterrows()],
                      "llrr")

    # ------------------------------------------------------------ trends
    half = start + (end - start) / 2
    h1, h2 = live[live["date"] < half], live[live["date"] >= half]
    doc.section("Trends", f"First half versus second half, split at {half:%d %b %Y}.")
    rows = []
    for lab, bkt in (("Income", INCOME), ("Spending", SPENDING)):
        a = h1[h1["bucket"] == bkt]["amount"].sum()
        b = h2[h2["bucket"] == bkt]["amount"].sum()
        rows.append([lab, money(a), money(b), f"{(b / a - 1) * 100:+.1f}%" if a else "—"])
    a_net = (h1[h1["bucket"] == INCOME]["amount"].sum()
             - h1[h1["bucket"] == SPENDING]["amount"].sum())
    b_net = (h2[h2["bucket"] == INCOME]["amount"].sum()
             - h2[h2["bucket"] == SPENDING]["amount"].sum())
    rows.append(["Net saved", money(a_net), money(b_net), money(b_net - a_net, sign=True)])
    doc.table(["", "1st half", "2nd half", "Change"], rows, "lrrr", emphasise_last=True)

    piv = (live[live["bucket"] == SPENDING]
           .assign(half=lambda d: np.where(d["date"] >= half, "h2", "h1"))
           .pivot_table(index="category", columns="half", values="amount",
                        aggfunc="sum", fill_value=0.0))
    for col in ("h1", "h2"):
        if col not in piv.columns:
            piv[col] = 0.0
    piv["delta"] = piv["h2"] - piv["h1"]
    doc.sub("Spending by category")
    doc.table(["Category", "1st half", "2nd half", "Change"],
              [[c, money(r["h1"]), money(r["h2"]), money(r["delta"], sign=True)]
               for c, r in piv.sort_values("delta", ascending=False).iterrows()], "lrrr")

    roll = monthly[[INCOME, SPENDING, "net"]].rolling(3).mean().dropna()
    if len(roll):
        doc.sub("Three-month rolling average")
        doc.table(["Month", "Income", "Spending", "Net"],
                  [[str(m), money(r[INCOME]), money(r[SPENDING]), money(r["net"], sign=True)]
                   for m, r in roll.iterrows()], "lrrr")

    # ---------------------------------------------------------- patterns
    doc.section("Patterns")
    daily = df.set_index("date")["balance"].resample("D").last().ffill()
    below = int((daily < args.low_balance).sum())
    doc.kv([
        ("Minimum balance", f"{money(daily.min())} on {daily.idxmin():%d %b %Y}"),
        ("Maximum balance", f"{money(daily.max())} on {daily.idxmax():%d %b %Y}"),
        ("Median balance", money(daily.median())),
        (f"Days under {money(args.low_balance)}",
         f"{below} of {len(daily)} ({below / len(daily) * 100:.0f}% of the period)"),
        ("Closing buffer",
         f"{daily.iloc[-1] / (total_out / months):.1f} months of average spending"
         if total_out else "—"),
    ])

    dom = spend.groupby(spend["date"].dt.day)["amount"].sum()
    if len(dom):
        doc.sub("Spending by day of month")
        rows, bars = [], []
        for d in range(1, 32):
            v = float(dom.get(d, 0.0))
            if v:
                rows.append([str(d), money(v)])
                bars.append(v / dom.max())
        doc.table(["Day", "Total"], rows, "lr", bars=bars)

    wd = spend.groupby(spend["date"].dt.day_name())["amount"].agg(["sum", "count"])
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    present = [d for d in order if d in wd.index]
    if present:
        doc.sub("Spending by weekday")
        doc.table(["Weekday", "Total", "n"],
                  [[d, money(wd.loc[d, "sum"]), f"{int(wd.loc[d, 'count'])}"] for d in present],
                  "lrr", bars=[wd.loc[d, "sum"] / wd["sum"].max() for d in present])

    # ------------------------------------------------------- big and odd
    doc.section("Largest transactions",
                f"Up to {args.top} each way, at most three per payee so one recurring "
                f"item cannot fill the table.")
    for heading, bkts in (("Money out — real spending only", [SPENDING]),
                          ("Money in — payroll and pass-through", [INCOME, PASSTHROUGH])):
        seen: dict[str, int] = {}
        rows = []
        for _, row in (live[live["bucket"].isin(bkts)]
                       .sort_values("amount", ascending=False).iterrows()):
            key = clean_merchant(row["description"])
            if seen.get(key, 0) >= 3:
                continue
            seen[key] = seen.get(key, 0) + 1
            rows.append([f"{row['date']:%Y-%m-%d}", money(row["amount"]),
                         row["category"], row["description"][:46]])
            if len(rows) >= args.top:
                break
        if rows:
            doc.sub(heading)
            doc.table(["Date", "Amount", "Category", "Description"], rows, "lrll")

    cm = live[live["bucket"] == SPENDING].groupby(["category", "month"])["amount"].sum()
    rows = []
    for cat in cm.index.get_level_values(0).unique():
        s = cm.loc[cat]
        total, med = s.sum(), s.median()
        for m, v in s.items():
            if v <= 300:
                continue
            spike = med > 0 and v >= 3 * med
            concentrated = len(s) > 1 and v >= 0.6 * total
            if spike or concentrated:
                why = (f"3× the typical {money(med)} month" if spike
                       else f"{v / total * 100:.0f}% of all '{cat}' in the period")
                rows.append([str(m), cat, money(v), why])
    if rows:
        doc.section("Unusual months",
                    "Spending months 3× a typical month, or that dominate their category.")
        doc.table(["Month", "Category", "Amount", "Why"], rows, "llrl")

    unc = live[live["category"].str.startswith("Uncategorised")]
    if len(unc):
        doc.section("Uncategorised",
                    "Matched no rule — add patterns to RULES in analyze.py.")
        doc.table(["Date", "Direction", "Amount", "Description"],
                  [[f"{r['date']:%Y-%m-%d}", "in" if r["is_credit"] else "out",
                    money(r["amount"]), r["description"][:56]] for _, r in unc.iterrows()],
                  "llrl")
    return doc


# ---------------------------------------------------------------- output


def export(df: pd.DataFrame, doc: Doc, outdir: Path, args: argparse.Namespace) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    live = df[~df["is_reversal"]]
    written = []

    stamp = pd.Timestamp.now().strftime("%d %b %Y")
    html_path = outdir / "report.html"
    html_path.write_text(render_html(doc, stamp), encoding="utf-8")
    written.append(html_path)

    if args.text:
        txt = outdir / "report.txt"
        txt.write_text(render_text(doc), encoding="utf-8")
        written.append(txt)

    cols = ["date", "description", "transaction_type", "amount", "signed", "balance",
            "bucket", "category", "subcategory", "counterparty", "is_reversal"]
    df[cols].to_csv(outdir / "transactions_categorized.csv", index=False, float_format="%.2f")

    monthly = (live.groupby(["month", "bucket"])["amount"].sum()
               .unstack(fill_value=0.0)
               .reindex(columns=[INCOME, SPENDING, TRANSFER, PASSTHROUGH], fill_value=0.0))
    monthly["net"] = monthly[INCOME] - monthly[SPENDING]
    monthly["end_balance"] = df.groupby("month")["balance"].last()
    monthly.to_csv(outdir / "monthly_summary.csv", float_format="%.2f")

    (live.groupby(["bucket", "category"])["amount"].agg(["sum", "count", "median"])
     .sort_values(["bucket", "sum"], ascending=[True, False])
     .to_csv(outdir / "category_summary.csv", float_format="%.2f"))
    return written


def periods(df: pd.DataFrame, calendar: bool) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """12-month windows anchored on the first transaction, or calendar years."""
    start, end = df["date"].min().normalize(), df["date"].max().normalize()
    out = []
    if calendar:
        for y in range(start.year, end.year + 1):
            lo = pd.Timestamp(year=y, month=1, day=1)
            hi = pd.Timestamp(year=y + 1, month=1, day=1)
            out.append((str(y), "", lo, hi))
        return out
    cur, i = start, 1
    while cur <= end:
        nxt = cur + pd.DateOffset(years=1)
        last = min(nxt - pd.Timedelta(days=1), end)
        out.append((f"Year {i}", f"{cur:%d %b %Y} – {last:%d %b %Y}", cur, nxt))
        cur, i = nxt, i + 1
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Full analysis of a Capital One 360 Checking CSV export.")
    ap.add_argument("csv", help="path to the Capital One transaction export")
    ap.add_argument("--outdir", default="output", help="where to write reports (default: output)")
    ap.add_argument("--top", type=int, default=15, help="rows in 'top N' tables (default: 15)")
    ap.add_argument("--low-balance", type=float, default=500.0,
                    help="low-balance threshold for the patterns section (default: 500)")
    ap.add_argument("--by-year", action="store_true",
                    help="a separate report per 12-month window from the first transaction")
    ap.add_argument("--calendar-years", action="store_true",
                    help="with --by-year, split on Jan-Dec instead of the anchored window")
    ap.add_argument("--text", action="store_true", help="also write a plain-text report.txt")
    ap.add_argument("--quiet", action="store_true", help="do not print the report to the console")
    ap.add_argument("--no-export", action="store_true", help="print only, write nothing")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"No such file: {path}")

    full = flag_reversals(categorize(load(path)))
    out = Path(args.outdir)
    written: list[Path] = []

    jobs = [(None, None, None, full, out)]
    if args.by_year:
        jobs = []
        for label, span, lo, hi in periods(full, args.calendar_years):
            part = full[(full["date"] >= lo) & (full["date"] < hi)].reset_index(drop=True)
            if part.empty:
                continue
            jobs.append((label, span, (lo, hi), part, out / label.lower().replace(" ", "-")))

    for label, span, window, part, outdir in jobs:
        doc = build_doc(part, args, label, span, window)
        if not args.quiet:
            print(render_text(doc))
        if not args.no_export:
            written += export(part, doc, outdir, args)

    for p in written:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
