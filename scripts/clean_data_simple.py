# -*- coding: utf-8 -*-
"""应收数据清洗与预处理 —— 基础可读版（clean_data_simple.py）
=============================================================

  1. 读入真实 SAP 应收账款原始 CSV（data/raw/dataset.csv，约 5 万行）
  2. 删掉全空的列（原始数据里有一整列 area_business 全是空）
  3. 把几列脏日期解析成真正的日期类型
  4. 派生几个有用的特征：距到期天数、距结清天数、逾期天数、金额对数
  5. 按过账时间把数据切成 训练集(前80%) / 验证集(后20%)，防止模型偷看未来
  6. 检查 isOpen 标记 和 clear_date 是否一致，最后写成 parquet + 清洗日志

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么，方便你从头读懂。

运行方式（不带参数即可）：
    python scripts/clean_data_simple.py

English:
AR data cleaning and preprocessing (plain, readable version).

Steps:
  1. Read the raw SAP accounts-receivable CSV (data/raw/dataset.csv,
     ~50k rows)
  2. Drop fully-empty columns (one column, area_business, is all empty)
  3. Parse dirty date columns into real dates
  4. Derive useful features: days to due, days to cleared, overdue days,
     log amount
  5. Split chronologically into train (first 80%) / validation (last 20%)
     so models cannot peek at the future
  6. Check consistency between the isOpen flag and clear_date, then write
     parquet + a cleaning log

The code uses the plainest style on purpose; every step states what it
does and why.

Run (no arguments needed):
    python scripts/clean_data_simple.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
# 本文件在 scripts/ 下，上一级就是项目根目录
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "dataset.csv"        # 原始 CSV 默认位置
OUT_DIR = ROOT / "data" / "output"                  # 清洗结果放这里
OUT_DIR.mkdir(parents=True, exist_ok=True)          # 输出目录不存在就建一个
OUT_PARQUET = OUT_DIR / "ar_clean.parquet"          # 清洗后的主表
OUT_LOG = OUT_DIR / "cleaning_report.json"          # 清洗统计日志


# ------------------------------------------------------------------
# 1. 日期解析的两个小工具
# ------------------------------------------------------------------
def parse_yyyymmdd_col(series):
    """把数值形式的 YYYYMMDD（比如 20200522）整列解析成日期。

    原始数据里 due_in_date / baseline_create_date 是整数 8 位数字，
    先转成可空整数、再转成 "20200522" 这种字符串，最后按 %Y%m%d 解析。
    解析不出来（脏数据或缺失）的统一变成空日期 NaT。
    """
    as_int = series.astype("Int64")     # 先转可空整数，去掉小数点后面的 .0
    as_text = as_int.astype(str)        # 再转字符串，例如 "20200522"
    return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")


def parse_str_date_col(series):
    """把本来就是字符串、但格式是 YYYYMMDD 的列解析成日期（document_create_date 系列）。"""
    as_text = series.astype(str)
    return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")


# ------------------------------------------------------------------
# 2. 主流程
# ------------------------------------------------------------------
def main():
    # ---- 2.1 读入原始 CSV ----
    df = pd.read_csv(SRC)
    report = {"input_rows": len(df), "input_cols": df.shape[1]}

    # ---- 2.2 删除全空列 ----
    # 原始数据里有一整列完全没有信息（area_business），留着只会碍事。
    empty_cols = []
    for c in df.columns:
        if df[c].isna().all():          # 这一列是不是全为空
            empty_cols.append(c)
    for c in empty_cols:
        df = df.drop(columns=[c])
    report["dropped_empty_cols"] = empty_cols

    # ---- 2.3 日期解析 ----
    # posting_date / clear_date 本来就是正常日期文本，直接解析。
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df["clear_date"] = pd.to_datetime(df["clear_date"], errors="coerce")

    # 这几列是 8 位数字/字符串，要用上面的小工具按 YYYYMMDD 解析。
    df["due_in_date"] = parse_yyyymmdd_col(df["due_in_date"])
    df["baseline_create_date"] = parse_yyyymmdd_col(df["baseline_create_date"])
    df["document_create_date"] = parse_str_date_col(df["document_create_date"])
    df["document_create_date.1"] = parse_str_date_col(df["document_create_date.1"])

    # 日期合理性：posting_date 解析不出来的行数（真实数据范围是 2018-2021）
    bad = df["posting_date"].isna()
    report["posting_date_invalid"] = int(bad.sum())

    # ---- 2.4 派生特征 ----
    # 两个日期相减得到的天数。只要有一边是空，结果也是空（NaN）。
    df["days_to_due"] = (df["due_in_date"] - df["posting_date"]).dt.days      # 过账→到期
    df["days_to_clear"] = (df["clear_date"] - df["posting_date"]).dt.days     # 过账→结清
    df["days_overdue"] = (df["clear_date"] - df["due_in_date"]).dt.days      # 到期→结清；NaN=还没结清
    # 金额取对数：大额发票量纲差异太大，取 log1p 让后面的模型更好处理
    df["amount_log"] = np.log1p(df["total_open_amount"])

    # ---- 2.5 重复凭证统计 ----
    # doc_id 可以重复（同一张凭证下有多条明细），这里只记录数量，明细都保留。
    dup_doc = df["doc_id"].duplicated().sum()
    report["duplicate_doc_id_rows"] = int(dup_doc)

    # ---- 2.6 时间切分：训练/验证 ----
    # 按过账日期排序，前 80% 当训练集，后 20% 当验证集。
    # 这样模型在训练时看不到后面时间段的数据，避免"偷看未来"。
    df = df.sort_values("posting_date").reset_index(drop=True)
    n = len(df)
    cut = int(n * 0.8)
    split_list = []
    for i in range(n):
        if i < cut:
            split_list.append("train")
        else:
            split_list.append("val")
    df["split"] = split_list
    report["train_rows"] = int((df["split"] == "train").sum())
    report["val_rows"] = int((df["split"] == "val").sum())

    # ---- 2.7 isOpen 与 clear_date 一致性检查 ----
    # 业务上：clear_date 为空 = 还没结清，应该正好等于 isOpen=1。
    # 我们逐行核对这个关系对不对，记下一致的行数。
    consistent = 0
    for i in range(len(df)):
        clear_missing = pd.isna(df["clear_date"].iloc[i])   # 这行有没有结清日期
        is_open = df["isOpen"].iloc[i] == 1                  # 这行有没有被标成未结清
        if clear_missing == is_open:                         # 两者一致
            consistent += 1
    report["isOpen_clear_consistent_rows"] = consistent
    report["isOpen_dist"] = df["isOpen"].value_counts().to_dict()

    report["output_rows"] = len(df)
    report["output_cols"] = df.shape[1]

    # ---- 2.8 写出 parquet 和清洗日志 ----
    df.to_parquet(OUT_PARQUET, index=False)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 打印清洗统计（打印内容一致）
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("WROTE:", OUT_PARQUET)


if __name__ == "__main__":
    main()
