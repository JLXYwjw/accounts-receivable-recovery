# -*- coding: utf-8 -*-
"""探查原始应收数据 —— 基础可读版（explore_data_simple.py）
=========================================================
这个文件做的事，和原版 explore_data.py 完全一样：
  1. 读入 data/raw/dataset.csv（真实 SAP 应收账款，约 5 万行）
  2. 打印整体形状、所有列名
  3. 逐列打印：类型 / 缺失百分比 / 唯一值个数 / 前 3 个示例
  4. 打印几个关键字段（是否未结清、币种、业务代码等）的取值分布
  5. 打印几个数值列的 min/max/mean/缺失率
  6. 打印几个日期列的原始示例
  7. 按是否未结清分组，看未结清金额的笔数/合计/平均

它只是个"看完心里有数"的探查脚本，不写任何文件，只往屏幕上打印。
运行方式（和原版一样，不带参数）：
    python scripts/explore_data_simple.py
"""
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置与打印选项
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "dataset.csv"

# 让 pandas 打印的时候不要把行/列/内容截断，方便看清完整结果
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 40)


def main():
    # ---- 1. 读入原始 CSV，打印整体形状与列名 ----
    df = pd.read_csv(SRC)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    # ---- 2. 逐列体检：类型、缺失率、唯一值、示例 ----
    print("\n=== 每列: dtype / null% / 唯一值 / 示例 ===")
    for c in df.columns:
        s = df[c]
        nuniq = s.nunique(dropna=True)              # 这列有多少种不同取值
        nullp = round(s.isna().mean() * 100, 2)     # 缺失行占百分之几
        ex = s.dropna().head(3).tolist()            # 取前 3 个非空值当示例
        print(f"- {c}  | {s.dtype} | null {nullp}% | uniq {nuniq} | ex {ex}")

    # ---- 3. 几个关键字段的取值分布 ----
    # 这些字段是分类/业务维度，看看每个值各有多少行。
    print("\n=== 分类/关键字段分布 ===")
    key_cols = ["isOpen", "invoice_currency", "document type", "business_code",
                "area_business", "cust_payment_terms"]
    for c in key_cols:
        if c in df.columns:
            print(f"\n[{c}]")
            print(df[c].value_counts(dropna=False).to_string())

    # ---- 4. 数值字段统计：最小、最大、平均、缺失率 ----
    print("\n=== 数值字段统计 ===")
    num_cols = ["total_open_amount", "buisness_year", "doc_id", "posting_id", "invoice_id"]
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            # 平均这一列：如果整列都没有有效值，就输出 None，避免报错
            if s.notna().any():
                mean_text = round(s.mean(), 2)
            else:
                mean_text = None
            nullp = round(s.isna().mean() * 100, 2)
            print(f"- {c}: min={s.min()} max={s.max()} mean={mean_text} null%={nullp}")

    # ---- 5. 日期列原始示例 ----
    # 直接看原文本长什么样，确认后面日期解析的格式对不对。
    print("\n=== 日期字段原始示例 ===")
    date_cols = ["clear_date", "posting_date", "document_create_date",
                 "due_in_date", "baseline_create_date"]
    for c in date_cols:
        if c in df.columns:
            print(f"- {c}: ex {df[c].dropna().head(5).tolist()}")

    # ---- 6. isOpen 与金额的关系 ----
    # 按"是否未结清"分组，看每组的笔数、金额合计、平均金额。
    if "isOpen" in df.columns and "total_open_amount" in df.columns:
        print("\n=== isOpen x total_open_amount ===")
        print(df.groupby("isOpen")["total_open_amount"].agg(["count", "sum", "mean"]).to_string())


if __name__ == "__main__":
    main()
