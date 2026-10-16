# -*- coding: utf-8 -*-
"""探查 receivables/dataset.csv —— 真实 SAP 应收账款数据（账期/逾期/催收）。
输出：列、类型、缺失、唯一值、关键分类字段分布、数值/日期统计、目标列 isOpen。
"""
import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "data" / "raw" / "dataset.csv"
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 40)

df = pd.read_csv(SRC)
print("shape:", df.shape)
print("columns:", list(df.columns))

print("\n=== 每列: dtype / null% / 唯一值 / 示例 ===")
for c in df.columns:
    s = df[c]
    nuniq = s.nunique(dropna=True)
    nullp = round(s.isna().mean()*100, 2)
    ex = s.dropna().head(3).tolist()
    print(f"- {c}  | {s.dtype} | null {nullp}% | uniq {nuniq} | ex {ex}")

print("\n=== 分类/关键字段分布 ===")
for c in ["isOpen","invoice_currency","document type","business_code","area_business","cust_payment_terms"]:
    if c in df.columns:
        print(f"\n[{c}]")
        print(df[c].value_counts(dropna=False).to_string())

print("\n=== 数值字段统计 ===")
num_cols = ["total_open_amount","buisness_year","doc_id","posting_id","invoice_id"]
for c in num_cols:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"- {c}: min={s.min()} max={s.max()} mean={round(s.mean(),2) if s.notna().any() else None} null%={round(s.isna().mean()*100,2)}")

print("\n=== 日期字段原始示例 ===")
for c in ["clear_date","posting_date","document_create_date","due_in_date","baseline_create_date"]:
    if c in df.columns:
        print(f"- {c}: ex {df[c].dropna().head(5).tolist()}")

# isOpen 与金额关系
if "isOpen" in df.columns and "total_open_amount" in df.columns:
    print("\n=== isOpen x total_open_amount ===")
    print(df.groupby("isOpen")["total_open_amount"].agg(["count","sum","mean"]).to_string())
