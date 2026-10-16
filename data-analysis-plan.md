# Accounts Receivable Recovery — Analysis Plan (Project Outline)

> Data: `datasets/receivables/dataset.csv` (real SAP accounts receivable, 50,000 rows, 2019–2020)
> Target role: reconciliation / collections (reconcile claims vs invoices vs payment, revenue recovery)

---

## 1. Business Background & Objectives

**Business owner**: Finance / AR (accounts receivable) team.
**Core pain**: of 50,000 receivable invoices, **10,000 (20%) remain unpaid past due**, ~ **$334M** in open amount. The team has no resources to chase every customer one by one, and does not know who to chase first — collection effort is spread thin, DSO (days sales outstanding) stretches, and bad-debt risk accumulates.

**Objective of this project (business-landing)**: predict "which invoices are most likely to remain unpaid past due", and from that build a **collection priority queue**, concentrating collection effort on invoices with "large amount × high probability of remaining open", so as to:
1. Increase recovery rate (chase what can be collected first);
2. Shorten DSO (turn receivable into received sooner);
3. Inform payment-term negotiation (which terms are worst for past-due).

**Reconciliation angle**: `clear_date` (settled) vs `isOpen` (open) is exactly the "receivable vs received" reconciliation gap. This project quantifies the gap ($334M) and gives the recovery order — the core action of a "revenue recovery" role.

---

## 2. Data Understanding (19 columns)

| Field | Meaning | Note |
|---|---|---|
| `invoice_id` / `doc_id` | Invoice / document no. | doc_id has duplicates (needs dedup judgment) |
| `business_code` | Business unit (6) | U001 dominant |
| `cust_number` / `name_customer` | Customer no. / name (1425 / 4197) | real company names |
| `posting_date` / `baseline_create_date` | Posting date / baseline create date | |
| `due_in_date` | Due date | |
| `clear_date` | Actual settlement date | **20% missing = open** |
| `total_open_amount` | Open amount (USD) | target weighting |
| `invoice_currency` | Currency (USD 46081 / CAD 3919) | |
| `cust_payment_terms` | SAP payment terms (74 kinds) | NAA8/NAH4… |
| `isOpen` | **Target**: 0 settled / 1 open | 10000 / 40000, 20% positive |
| `area_business` | Empty column | **drop** |
| `document type` | RV receivable / X2 | RV dominant |

---

## 3. Cleaning / Preprocessing Plan

- Drop fully empty column `area_business`.
- Dates: parse `posting_date`/`clear_date`; convert `due_in_date`/`baseline_create_date` (numeric `20200210.0`) to datetime.
- Missing: missing `clear_date` means open (matches isOpen; keep as feature "has settlement date"); `invoice_id` 0.01% missing → blank.
- Duplicates: `doc_id` shared across rows (one doc, many lines); judge by invoice whether it is the same receivable; state the definition clearly.
- Derived features: posting→due days, posting→settlement days (if settled), due→settlement (past-due days), log amount, business-unit/currency/terms codes.
- Time split: train on first 80% by posting date, validate on last 20% (prevent leakage; never use future info to predict).

---

## 4. Metrics

| Metric | Definition | Use |
|---|---|---|
| Open rate | share isOpen=1 (20%) | gap size |
| Open amount | Σ total_open_amount (isOpen=1, ~$334M) | recoverable pool |
| DSO | avg posting→settlement days | collection efficiency |
| Past-due amount / days | due date vs settlement date | collection urgency |
| Collection priority score | open amount × open probability | rank for collection |
| ML metrics | AUC / PR-AUC / recall | catch all past-due |

---

## 5. Modeling (business-driven; explain principle/why/how)

### 5.1 isOpen binary classification (core)
- **Why**: rules only separate "past due or not"; they cannot **rank the 10,000 open invoices by recovery risk**. ML gives each invoice a "probability of remaining unpaid past due", × amount = collection priority.
- **Principle**: learn the mapping features→open probability. Baseline logistic regression (interpretable coefficients) → random forest / gradient boosting (tree models capture interactions and non-linearity, e.g. "high amount × high past-due × specific terms").
- **Features**: log amount, posting→due days, business unit, currency, payment terms, customer historical settlement rate (computed within CV to prevent leakage), posting month.
- **Evaluation**: AUC (ranking), PR-AUC (imbalance), recall@high-amount segment (recovery view).
- **Business output**: collection priority queue = open probability × amount; top N handed directly to the collections team.

### 5.2 Whether to use deep learning — judgment (honest)
- This dataset is **50,000 rows, tabular, non-sequential**: gradient boosting / random forest already perform very well and are **interpretable** (feature importance, coefficients), matching the "collections narrative".
- DL (e.g. MLP) adds little at this scale and is hard to explain — **not used as the main model**; only as an ablation comparison, not the protagonist. → Same standard as the procurement project: DL needs large data to pay off; here it is insufficient.

---

## 6. Visualization Plan (each chart Insight + Action; sky-blue + gold palette)

1. **Open distribution**: isOpen 0/1 donut + amount comparison ($334M gap).
2. **Business-unit view**: open rate / amount by business_code.
3. **Aging distribution**: posting→due days histogram (reflecting terms).
4. **Payment terms vs past-due**: top past-due among the 74 terms.
5. **Past-due amount trend**: by posting month.
6. **ML feature importance**: horizontal bars (interpretability).
7. **Collection priority Top N**: amount × probability ranking (directly executable).

---

## 7. Recommendations (landing)

1. **Collection priority queue**: ML-ranked top open invoices, chase "large amount × high probability" first.
2. **Terms negotiation**: bring the worst past-due payment terms into contract revision.
3. **Reconciliation gap check**: line-by-line settled vs open, recover omissions.
4. **Automated reminders**: trigger collection tickets before due, shortening DSO.

---

## 8. Deliverables

- Clean table `data/output/ar_clean.parquet` / CSV.
- ML results (probabilities, priority queue).
- Bilingual HTML reports (every chart Insight/Action).
- Tests (cleaning definitions, number consistency, reproducibility).
- Local git committed in small steps.
