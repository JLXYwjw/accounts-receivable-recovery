# -*- coding: utf-8 -*-
"""拷打诊断 —— 基础可读版（diag_qa_simple.py）
=================================================

  1. 读入 ar_clean.parquet，打印过账日期/结清日期的时间范围
  2. 看未结清(isOpen=1)的发票都分布在哪些过账月份
  3. 用最后一次结清日期当"评估时点"，看未结清发票里有多少已经过了到期日
  4. 验证 total_open_amount 这一列的真实语义（已结清的行这一列也有值）
  5. 看未结清发票的 days_to_due 分布，判断它们是不是大多还没到期
  6. 算"真逾期应收"= 已到期且未结清 的笔数和金额

它只是个诊断脚本，不写文件，只往屏幕打印。
运行方式（不带参数）：
    python scripts/diag_qa_simple.py

English:
Stress-test diagnostics (plain, readable version).

Steps:
  1. Read ar_clean.parquet, print the posting/cleared date range
  2. See which posting months the open (isOpen = 1) invoices fall into
  3. Use the last cleared date as the evaluation point and count how many
     open invoices are already past their due date
  4. Verify the real semantics of the total_open_amount column (cleared
     rows also carry a value there)
  5. Look at the days_to_due distribution of open invoices to judge
     whether most of them are not due yet
  6. Compute "true overdue AR" = past due AND open, in count and amount

Diagnostics only: prints to screen, writes nothing.

Run (no arguments):
    python scripts/diag_qa_simple.py
"""
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"


def main():
    df = pd.read_parquet(OUT / "ar_clean.parquet")

    # ---- 1. 时间范围 ----
    print("时间范围: posting", df["posting_date"].min(), "->", df["posting_date"].max())
    print("clear 最大:", df["clear_date"].max(), "最小:", df["clear_date"].min())
    print("rows:", len(df), "isOpen1:", int((df["isOpen"] == 1).sum()))

    # ---- 2. isOpen=1 的过账月份分布 ----
    print("\n=== isOpen=1 的过账月份分布 ===")
    open_rows = df[df["isOpen"] == 1]
    print(open_rows.groupby(open_rows["posting_date"].dt.to_period("M")).size().to_string())

    # ---- 3. 评估时点：用最后一张有结清记录的日期当参考 ----
    # 定义"已到期未结清"= due_in_date < 评估时点
    snapshot = df["clear_date"].max()
    print("\n评估时点(最后结清日):", snapshot)
    open_df = df[df["isOpen"] == 1].copy()
    open_df["due_passed"] = open_df["due_in_date"] < snapshot
    print("isOpen=1 中已到期(due<评估时点)的:", int(open_df["due_passed"].sum()), "/", len(open_df))
    print("isOpen=1 中未到期(due>=评估时点)的:", int((~open_df["due_passed"]).sum()))
    not_passed = open_df[~open_df["due_passed"]]
    passed = open_df[open_df["due_passed"]]
    print("未到期未结清的金额:", round(float(not_passed["total_open_amount"].sum()), 2))
    print("已到期未结清的金额:", round(float(passed["total_open_amount"].sum()), 2))

    # ---- 4. total_open_amount 语义验证 ----
    # 看已结清(isOpen=0)的行，这一列是不是也有值（说明它其实是发票金额，不是"未结金额"）
    paid = df[df["isOpen"] == 0]
    print("\n=== total_open_amount 语义 ===")
    print("isOpen=0 行数:", len(paid), "其中 total_open_amount>0:", int((paid["total_open_amount"] > 0).sum()))
    print("isOpen=0 的 total_open_amount mean:", round(float(paid["total_open_amount"].mean()), 2))

    # ---- 5. isOpen=1 的 days_to_due 分布 ----
    # days_to_due = 到期日 - 过账日；小于 0 表示已经过了到期日
    print("\n=== isOpen=1 的 days_to_due 分布 ===")
    print(open_df["days_to_due"].describe().to_string())
    print("isOpen=1 中 days_to_due<0(已到期):", int((open_df["days_to_due"] < 0).sum()))
    print("isOpen=1 中 days_to_due>=0(未到期):", int((open_df["days_to_due"] >= 0).sum()))

    # ---- 6. 真逾期应收：已到期 且 未结清 ----
    print("\n=== 真逾期应收(已到期且未结清) ===")
    true_overdue = open_df[open_df["due_passed"]]
    print("数量:", len(true_overdue), "金额:", round(float(true_overdue["total_open_amount"].sum()), 2))
    # （末尾的注释：若按 due passed 重新定义目标，posting_month 还有那么重要吗？）


if __name__ == "__main__":
    main()
