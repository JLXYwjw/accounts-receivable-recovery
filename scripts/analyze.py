# -*- coding: utf-8 -*-
"""应收关键指标分析。读 ar_clean.parquet，输出 metrics JSON（口径与报告同源）。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"
df = pd.read_parquet(OUT_DIR / "ar_clean.parquet")

m = {}
m["total_rows"] = len(df)
m["total_invoices"] = df["invoice_id"].nunique()
m["total_amount"] = round(float(df["total_open_amount"].sum()), 2)

# 未结清
sub = df[df["isOpen"] == 1]
m["open_count"] = int(len(sub))
m["open_rate_pct"] = round(len(sub) / len(df) * 100, 1)
m["open_amount"] = round(float(sub["total_open_amount"].sum()), 2)

# 已结清账期 DSO
paid = df[df["isOpen"] == 0]
m["paid_dso_days"] = round(float(paid["days_to_clear"].mean()), 1)
m["paid_overdue_rate_pct"] = round(float((paid["days_overdue"] > 0).mean() * 100), 1)
m["paid_overdue_amount"] = round(float(paid.loc[paid["days_overdue"] > 0, "total_open_amount"].sum()), 2)

# 平均账期（过账→到期）
m["avg_days_to_due"] = round(float(df["days_to_due"].mean()), 1)

# 业务单元未结清
bu = df.groupby("business_code").apply(
    lambda g: pd.Series({"rows": len(g), "open_rate_pct": round((g["isOpen"] == 1).mean()*100, 1),
                         "open_amount": round(float(g.loc[g["isOpen"]==1, "total_open_amount"].sum()), 2)}, index=["rows","open_rate_pct","open_amount"]),
    include_groups=False).reset_index()
m["by_business"] = bu.to_dict("records")

# 币种
cur = df.groupby("invoice_currency").apply(
    lambda g: pd.Series({"open_rate_pct": round((g["isOpen"]==1).mean()*100, 1),
                         "open_amount": round(float(g.loc[g["isOpen"]==1, "total_open_amount"].sum()), 2)}, index=["open_rate_pct","open_amount"]),
    include_groups=False).reset_index()
m["by_currency"] = cur.to_dict("records")

# 付款条款逾期（已结清中 days_overdue>0 的比例）
terms = df[df["isOpen"] == 0].groupby("cust_payment_terms").apply(
    lambda g: pd.Series({"count": len(g), "overdue_pct": round(float((g["days_overdue"] > 0).mean()*100), 1),
                         "avg_overdue_days": round(float(g["days_overdue"].clip(lower=0).mean()), 1)}, index=["count","overdue_pct","avg_overdue_days"]),
    include_groups=False).reset_index()
# 只保留有统计意义的条款(样本>=50), 避免 count=1 的小样本噪声
terms = terms[terms["count"] >= 50].sort_values("overdue_pct", ascending=False)
m["top_overdue_terms"] = terms.head(10).to_dict("records")
m["overdue_terms_sample_size"] = "min_count=50"

# 月度未结清金额（按过账月）
df["posting_month"] = df["posting_date"].dt.to_period("M").astype(str)
mon = df.groupby("posting_month").apply(
    lambda g: pd.Series({"open_amount": round(float(g.loc[g["isOpen"]==1, "total_open_amount"].sum()), 2),
                         "total_amount": round(float(g["total_open_amount"].sum()), 2)}, index=["open_amount","total_amount"]),
    include_groups=False).reset_index()
m["monthly"] = mon.to_dict("records")

# top 未结清客户
cust = df[df["isOpen"]==1].groupby("name_customer")["total_open_amount"].agg(["sum","count"]).reset_index()
cust = cust.sort_values("sum", ascending=False).head(10)
m["top_open_customers"] = cust.rename(columns={"sum":"amount","count":"n_invoices"}).to_dict("records")

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=2)

print(json.dumps({"ok": True, "metrics_keys": list(m.keys()), "open_amount": m["open_amount"], "dso": m["paid_dso_days"]}, ensure_ascii=False))
