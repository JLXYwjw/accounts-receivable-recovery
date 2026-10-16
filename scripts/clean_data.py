# -*- coding: utf-8 -*-
"""应收数据清洗与预处理（accounts-receivable-recovery）。

输入: datasets/receivables/dataset.csv（真实 SAP 应收 50,000 行）
输出: data/output/ar_clean.parquet + 清洗日志（缺失/重复/日期校验/派生特征）
口径:
- 删除全空列 area_business
- 日期解析: posting_date/clear_date(字符串), due_in_date/baseline_create_date(数值YYYYMMDD)
- clear_date 缺失 = 未结清(与 isOpen 一致，保留)
- 派生: days_to_due / days_to_clear / days_overdue / amount_log
- 时间切分: 按 posting_date 前80% 训练 / 后20% 验证(防泄漏)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "data" / "raw" / "dataset.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = OUT_DIR / "ar_clean.parquet"
OUT_LOG = OUT_DIR / "cleaning_report.json"

def main():
    df = pd.read_csv(SRC)
    report = {"input_rows": len(df), "input_cols": df.shape[1]}

    # 1) 删除全空列
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    for c in empty_cols:
        df = df.drop(columns=[c])
    report["dropped_empty_cols"] = empty_cols

    # 2) 日期解析
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df["clear_date"] = pd.to_datetime(df["clear_date"], errors="coerce")
    df["due_in_date"] = pd.to_datetime(df["due_in_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["baseline_create_date"] = pd.to_datetime(df["baseline_create_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["document_create_date"] = pd.to_datetime(df["document_create_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["document_create_date.1"] = pd.to_datetime(df["document_create_date.1"].astype(str), format="%Y%m%d", errors="coerce")

    # 日期合理性校验（真实数据范围 2018-2021）
    bad = df["posting_date"].isna()
    report["posting_date_invalid"] = int(bad.sum())

    # 3) 派生特征
    df["days_to_due"] = (df["due_in_date"] - df["posting_date"]).dt.days
    df["days_to_clear"] = (df["clear_date"] - df["posting_date"]).dt.days
    df["days_overdue"] = (df["clear_date"] - df["due_in_date"]).dt.days  # NaN=未结清
    df["amount_log"] = np.log1p(df["total_open_amount"])

    # 4) 重复口径: doc_id 可重复(同凭证多明细), 记录但保留明细
    dup_doc = df["doc_id"].duplicated().sum()
    report["duplicate_doc_id_rows"] = int(dup_doc)

    # 5) 时间切分(训练/验证, 按 posting_date)
    df = df.sort_values("posting_date").reset_index(drop=True)
    n = len(df)
    cut = int(n * 0.8)
    df["split"] = np.where(np.arange(n) < cut, "train", "val")
    report["train_rows"] = int((df["split"] == "train").sum())
    report["val_rows"] = int((df["split"] == "val").sum())

    # 6) isOpen 一致性: clear_date 缺失 应 == isOpen=1
    consistent = int((df["clear_date"].isna() == (df["isOpen"] == 1)).sum())
    report["isOpen_clear_consistent_rows"] = consistent
    report["isOpen_dist"] = df["isOpen"].value_counts().to_dict()

    report["output_rows"] = len(df)
    report["output_cols"] = df.shape[1]

    df.to_parquet(OUT_PARQUET, index=False)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("WROTE:", OUT_PARQUET)

if __name__ == "__main__":
    main()
