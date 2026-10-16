# -*- coding: utf-8 -*-
"""拷打诊断：验证 isOpen 是否纯由时间(快照)决定，以及 total_open_amount 语义。"""
import pandas as pd
from pathlib import Path

OUT = Path(r"D:\B编程\powerBI\github\accounts-receivable-recovery\data\output")
df = pd.read_parquet(OUT / "ar_clean.parquet")

print("时间范围: posting", df["posting_date"].min(), "->", df["posting_date"].max())
print("clear 最大:", df["clear_date"].max(), "最小:", df["clear_date"].min())
print("rows:", len(df), "isOpen1:", int((df["isOpen"]==1).sum()))

# 1) isOpen=1 的 posting 月份分布
print("\n=== isOpen=1 的过账月份分布 ===")
print(df[df["isOpen"]==1].groupby(df["posting_date"].dt.to_period("M")).size().to_string())

# 2) 评估时点: 用最后一张有结清记录的 clear 时间作参考; isOpen=1 中有多少已过 due
#    定义"已到期未结清": due_in_date < 评估时点(取 max clear_date 或数据末尾)
snapshot = df["clear_date"].max()
print("\n评估时点(最后结清日):", snapshot)
open_df = df[df["isOpen"]==1].copy()
open_df["due_passed"] = open_df["due_in_date"] < snapshot
print("isOpen=1 中已到期(due<评估时点)的:", int(open_df["due_passed"].sum()), "/", len(open_df))
print("isOpen=1 中未到期(due>=评估时点)的:", int((~open_df["due_passed"]).sum()))
print("未到期未结清的金额:", round(float(open_df[~open_df["due_passed"]]["total_open_amount"].sum()),2))
print("已到期未结清的金额:", round(float(open_df[open_df["due_passed"]]["total_open_amount"].sum()),2))

# 3) total_open_amount 语义: isOpen=0(已结清) 的行也有 total_open_amount 吗?
paid = df[df["isOpen"]==0]
print("\n=== total_open_amount 语义 ===")
print("isOpen=0 行数:", len(paid), "其中 total_open_amount>0:", int((paid["total_open_amount"]>0).sum()))
print("isOpen=0 的 total_open_amount mean:", round(float(paid["total_open_amount"].mean()),2))

# 4) isOpen=1 中 days_to_due(过账->到期) 分布: 是否大部分还没到期
print("\n=== isOpen=1 的 days_to_due 分布 ===")
print(open_df["days_to_due"].describe().to_string())
print("isOpen=1 中 days_to_due<0(已到期):", int((open_df["days_to_due"]<0).sum()))
print("isOpen=1 中 days_to_due>=0(未到期):", int((open_df["days_to_due"]>=0).sum()))

# 5) 若定义"真逾期应收"= 已到期(due passed) 且 未结清, 数量与金额
print("\n=== 真逾期应收(已到期且未结清) ===")
true_overdue = open_df[open_df["due_passed"]]
print("数量:", len(true_overdue), "金额:", round(float(true_overdue["total_open_amount"].sum()),2))
# 若按 due passed 重新定义目标, posting_month 还有那么重要吗?
