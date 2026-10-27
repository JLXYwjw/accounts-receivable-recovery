# -*- coding: utf-8 -*-
"""应收关键指标分析 —— 基础可读版（analyze_simple.py）
=====================================================

  1. 读入清洗好的 ar_clean.parquet
  2. 算整体规模：总行数、发票数、总金额
  3. 算未结清口径：有多少笔未结、占比多少、金额多少
  4. 算"真·逾期应收"：已经过了到期日、却还没结清的部分
  5. 算已结清部分的账期 DSO、逾期率
  6. 按业务代码、币种、付款条款、过账月、大客户分别汇总
  7. 把结果写成 metrics.json，并在屏幕上打印一行摘要

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么。

运行方式（不带参数）：
    python scripts/analyze_simple.py

English:
Accounts-receivable key-metric analysis (plain, readable version).

Steps:
  1. Read the cleaned ar_clean.parquet
  2. Overall size: rows, invoices, total amount
  3. Open-amount view: how many invoices are open, their share and amount
  4. "True overdue" AR: past the due date and still open
  5. Cleared part: settlement age DSO and overdue rate
  6. Summaries by business code, currency, payment terms, posting month,
     top customers
  7. Write metrics.json and print a one-line summary

The code uses the plainest style on purpose (for-loops, if-statements,
ordinary functions); every step states what it does and why.

Run (no arguments):
    python scripts/analyze_simple.py
"""
import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "output"


# ------------------------------------------------------------------
# 1. 小组件：给一个分组小表，算出"未结清"的几个数
# ------------------------------------------------------------------
def open_stats(g):
    """输入一个分组 DataFrame，返回 (总行数, 未结清笔数, 未结清金额)。"""
    rows = len(g)
    open_mask = (g["isOpen"] == 1)                 # 哪几行是未结清
    open_rows = int(open_mask.sum())
    open_amt = float(g.loc[open_mask, "total_open_amount"].sum())
    return rows, open_rows, open_amt


def main():
    df = pd.read_parquet(OUT_DIR / "ar_clean.parquet")

    m = {}

    # ---- 2.1 整体规模 ----
    m["total_rows"] = len(df)
    m["total_invoices"] = df["invoice_id"].nunique()
    m["total_amount"] = round(float(df["total_open_amount"].sum()), 2)
    # 注意：total_open_amount 这个列名是沿袭原始命名，实际它是"发票金额"，
    # 已结清的发票在这一列也有值，并不是"只有未结清的金额"。
    m["total_open_amount_note"] = "列名沿袭原始命名，实际为该发票金额，非仅未结清金额"

    # ---- 2.2 未结清整体情况 ----
    sub = df[df["isOpen"] == 1]
    m["open_count"] = int(len(sub))
    m["open_rate_pct"] = round(len(sub) / len(df) * 100, 1)
    m["open_amount"] = round(float(sub["total_open_amount"].sum()), 2)

    # ---- 2.3 真·逾期应收：已到期(due<快照时点) 且 未结清 ----
    # 快照时点 = 数据里最后一次结清的日期；在此之后还没结清的才算真逾期。
    snapshot = df["clear_date"].max()
    m["snapshot_date"] = str(snapshot.date())
    open_df = sub.copy()
    open_df["due_passed"] = open_df["due_in_date"] < snapshot
    m["open_overdue_count"] = int(open_df["due_passed"].sum())
    m["open_overdue_amount"] = round(float(open_df.loc[open_df["due_passed"], "total_open_amount"].sum()), 2)
    m["open_overdue_rate_pct"] = round(float(open_df["due_passed"].mean() * 100), 1)

    # ---- 2.4 已结清部分：账期 DSO 与逾期 ----
    paid = df[df["isOpen"] == 0]
    m["paid_dso_days"] = round(float(paid["days_to_clear"].mean()), 1)
    m["paid_overdue_rate_pct"] = round(float((paid["days_overdue"] > 0).mean() * 100), 1)
    m["paid_overdue_amount"] = round(float(paid.loc[paid["days_overdue"] > 0, "total_open_amount"].sum()), 2)

    # ---- 2.5 平均账期（过账→到期） ----
    m["avg_days_to_due"] = round(float(df["days_to_due"].mean()), 1)

    # ---- 2.6 按业务代码汇总 ----
    # 遍历每个 business_code 的分组，算出：行数、未结清率、未结清金额。
    by_business = []
    for code, g in df.groupby("business_code"):
        rows, open_rows, open_amt = open_stats(g)
        by_business.append({
            "business_code": code,
            "rows": float(rows),
            "open_rate_pct": round(open_rows / rows * 100, 1),
            "open_amount": round(open_amt, 2),
        })
    m["by_business"] = by_business

    # ---- 2.7 按币种汇总 ----
    by_currency = []
    for cur, g in df.groupby("invoice_currency"):
        rows, open_rows, open_amt = open_stats(g)
        by_currency.append({
            "invoice_currency": cur,
            "open_rate_pct": round(open_rows / rows * 100, 1),
            "open_amount": round(open_amt, 2),
        })
    m["by_currency"] = by_currency

    # ---- 2.8 付款条款逾期排行（只用已结清的行） ----
    # 每个条款算：笔数、逾期率、平均逾期天数（负数的天数按 0 算）。
    # 只保留笔数 >= 50 的条款，避免只有一两笔的小样本干扰排行。
    terms_rows = []
    for t, g in paid.groupby("cust_payment_terms"):
        count = len(g)
        overdue_pct = round(float((g["days_overdue"] > 0).mean() * 100), 1)
        # 把逾期天数小于 0 的部分截成 0 再求平均
        clip_vals = []
        for v in g["days_overdue"]:
            if pd.isna(v):
                clip_vals.append(0.0)
            elif v < 0:
                clip_vals.append(0.0)
            else:
                clip_vals.append(float(v))
        if len(clip_vals) > 0:
            avg_over = sum(clip_vals) / len(clip_vals)
        else:
            avg_over = 0.0
        terms_rows.append({
            "cust_payment_terms": t,
            "count": float(count),
            "overdue_pct": overdue_pct,
            "avg_overdue_days": round(avg_over, 1),
        })
    # 过滤小样本
    filtered = []
    for r in terms_rows:
        if r["count"] >= 50:
            filtered.append(r)

    # 按逾期率从高到低排序（用普通函数当排序键，不用 lambda）
    def overdue_key(r):
        return r["overdue_pct"]

    filtered.sort(key=overdue_key, reverse=True)
    m["top_overdue_terms"] = filtered[:10]
    m["overdue_terms_sample_size"] = "min_count=50"

    # ---- 2.9 按月汇总：每个过账月的未结金额 / 总金额 ----
    # 先把过账日期转成 "YYYY-MM" 字符串，再按月分组。
    df["posting_month"] = df["posting_date"].dt.to_period("M").astype(str)
    monthly = []
    for mon, g in df.groupby("posting_month"):
        open_amt = float(g.loc[g["isOpen"] == 1, "total_open_amount"].sum())
        total_amt = float(g["total_open_amount"].sum())
        monthly.append({
            "posting_month": mon,
            "open_amount": round(open_amt, 2),
            "total_amount": round(total_amt, 2),
        })
    m["monthly"] = monthly

    # ---- 2.10 未结清金额最大的 Top10 客户 ----
    cust_rows = []
    for name, g in sub.groupby("name_customer"):
        cust_rows.append({
            "name_customer": name,
            "amount": float(g["total_open_amount"].sum()),
            "n_invoices": int(len(g)),
        })
    # 按金额从大到小排序
    def amount_key(r):
        return r["amount"]

    cust_rows.sort(key=amount_key, reverse=True)
    # 取前 10 名，金额保留两位小数
    top_customers = []
    for r in cust_rows[:10]:
        top_customers.append({
            "name_customer": r["name_customer"],
            "amount": round(r["amount"], 2),
            "n_invoices": r["n_invoices"],
        })
    m["top_open_customers"] = top_customers

    # ---- 2.11 写出 metrics.json ----
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    # 打印一行摘要（打印格式一致）
    print(json.dumps({"ok": True, "metrics_keys": list(m.keys()),
                      "open_amount": m["open_amount"], "dso": m["paid_dso_days"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
