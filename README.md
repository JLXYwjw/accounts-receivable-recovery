# Accounts Receivable Recovery — Reconciliation & Collections Priority

**The business problem.** An e-commerce / B2B finance team doesn't get paid on
time, and nobody knows *which* unpaid invoices to chase first. Before you can
recover money you have to (a) quantify the gap between "should have been paid"
and "was actually paid", and (b) rank the open invoices so a collections team
can spend its hours where they return the most. This repo does exactly that on
**real** SAP accounts-receivable data.

Runs against the real SAP receivables dataset (`datasets/receivables/dataset.csv`,
50,000 invoice lines, 2018–2020) — not synthetic data.

## What it does

- Cleans a messy real export: drops a fully-empty column, normalizes mixed
  date formats (strings vs numeric `20200210.0`), and proves the label is sound —
  `clear_date` missing ⇔ `isOpen = 1` on **all 50,000 rows**.
- Quantifies the reconciliation gap: **10,000 open invoices (20%) = ~$334M**,
  mostly invoiced in the last 4 months (snapshot), plus **41.9% of paid invoices
  were late** (~$571.7M).
- Predicts which invoices will remain open (`isOpen`) with **GBM AUC 0.912**
  (vs logistic baseline 0.67), and ranks them by **priority = open amount ×
  probability** → a collections queue (top-50 ≈ $9.2M in the holdout set).
- Pinpoints the structural root cause: **payment term NAX2 is late 99.8% of the
  time, 42.6 days** — a negotiation target, not an accident.

## Honest notes (what I did NOT overclaim)

- `isOpen` is a **snapshot** label: ~92% of the model's signal is `posting_month`
  (recent invoices simply haven't hit their settlement cycle). AUC 0.912 is real,
  but this is not a "bad-debt over time" model. The value is the **priority queue**,
  not a long-horizon default forecast.
- **No deep learning used**: 50k rows, plain tabular, no sequence. Gradient
  boosting already hits 0.91 AUC and stays *explainable* (feature importance).
  DL gains little here and hurts the collections narrative — same rule as my
  other projects: only reach for DL when the data is big enough to justify it.
- The reconciliation here is at the **data layer** (billed-vs-paid gap + ranking);
  line-level invoice↔PO↔payment matching needs bank statements, listed as next
  step — not claimed as done.

## Layout

```
scripts/
  clean_data.py        # cleaning + derived features + train/val split (stratified)
  analyze.py           # key metrics -> data/output/metrics.json
  ml_analysis.py       # logistic / RF / GBM, priority queue, feature importance
tests/
  test_cleaning.py     # shape, date validity, isOpen⇔clear_date consistency
  test_report_numbers.py  # recompute metrics from parquet, assert HTML matches
报告/
  应收对账分析报告.html   # Chinese report, every chart has Insight + Action
```

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/clean_data.py
python scripts/analyze.py
python scripts/ml_analysis.py
pytest tests/ -q
```

## Key verified numbers

- 50,000 lines / 48,833 invoices / ~$1.62B AR; open rate 20% ($334M); DSO 18.1d
- Paid-but-late rate 41.9% ($571.7M); worst term NAX2 99.8% late, 42.6d
- Models (holdout): logistic AUC 0.669 / RF 0.908 / **GBM 0.912**
- Collections queue top-50 ≈ $9.2M in holdout

— Jiawei Wang
