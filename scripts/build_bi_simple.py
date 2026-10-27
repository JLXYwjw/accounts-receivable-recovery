# -*- coding: utf-8 -*-
"""应收账款 → Tableau + PowerBI 双份报表 —— 基础可读版（build_bi_simple.py）
=============================================================================

  1. 读入 ar_clean.parquet，挑出未结清(isOpen=1)的发票
  2. 给每张未结发票打三个分桶：账龄桶、金额桶、客户分组（Top10 客户 + 其他）
  3. 算月度挂账/未结/逾期金额表
  4. 导出两张 CSV：data/bi/ar_open.csv、data/bi/ar_month.csv
  5. 生成 Tableau 工作簿 AccountsReceivable_Tableau105.twb
  6. 生成 PowerBI 工程 AccountsReceivable.pbip

口径说明（重要，别改成 name_customer）：
  数据里 1,425 个 cust_number 对应 4,197 个 name_customer —— 同一客户有最多 11 种写法。
  按客户名聚合，Top10 会全是 "WAL-MAR xxx" 的变体，结论直接失真。
  所以客户维度一律用 cust_number，客户名只取最高频写法做展示。

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么。

运行方式（不带参数）：
    python scripts/build_bi_simple.py

English:
AR -> Tableau + Power BI twin reports (plain, readable version).

Steps:
  1. Read ar_clean.parquet, keep open invoices (isOpen = 1)
  2. Bucket each open invoice: aging bucket, amount bucket, customer
     group (Top-10 customers + others)
  3. Build monthly open / outstanding / overdue amount table
  4. Export two CSVs: data/bi/ar_open.csv, data/bi/ar_month.csv
  5. Generate Tableau workbook AccountsReceivable_Tableau105.twb
  6. Generate Power BI project AccountsReceivable.pbip

Caliber note (important, do not switch to name_customer):
  In the data, 1,425 cust_numbers map to 4,197 name_customer values - the
  same customer can appear in up to 11 spellings. Aggregating by customer
  name would make the Top-10 all "WAL-MAR ..." variants and distort the
  conclusion. So the customer dimension always uses cust_number, and the
  customer name only takes the most frequent spelling for display.

The code uses the plainest style on purpose; every step states what it
does and why.

Run (no arguments):
    python scripts/build_bi_simple.py
"""
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 把公共报表工具库 bi_kit 加进 import 路径（它在 github/tools/ 下）
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "tools", "bi_kit"))
import csvkit                      # noqa: E402
import pbipkit as K                # noqa: E402
import twbkit                      # noqa: E402

SRC = os.path.join(ROOT, "data", "output", "ar_clean.parquet")
BI = os.path.join(ROOT, "data", "bi")
SNAP = pd.Timestamp("2020-05-22")   # 快照日：metrics.json 里记录的时点

# 账龄桶标签（按"快照日 - 到期日"的天数分）
AGE_LAB = ["未到期", "1-30天", "31-60天", "61-90天"]
# 金额桶标签（单位：百万美元）
AMT_LAB = ["<1万", "1-5万", "5-10万", "10-50万", "50-100万", ">100万"]


# ------------------------------------------------------------------
# 1. 数据准备
# ------------------------------------------------------------------
def prep():
    d = pd.read_parquet(SRC)
    # 确保日期列是真正的日期类型
    d["posting_date"] = pd.to_datetime(d["posting_date"])
    d["due_in_date"] = pd.to_datetime(d["due_in_date"])
    # 过账月份格式化成 "YYYY-MM"
    d["posting_month"] = d["posting_date"].dt.strftime("%Y-%m")
    # 金额单位换成"百万美元"
    d["open_m"] = d["total_open_amount"] / 1e6

    # 只看未结清的发票
    op = d[d.isOpen == 1].copy()
    total = op.open_m.sum()

    # ---- 1.1 账龄桶 ----
    # 未结发票没有 days_overdue（全空），必须用 快照日 - 到期日 自己算逾期天数。
    op["ovd"] = (SNAP - op["due_in_date"]).dt.days
    op["age_bucket"] = pd.cut(
        op["ovd"], [-10 ** 6, 0, 30, 60, 10 ** 6],
        labels=AGE_LAB, right=True).astype(str)
    # 给每个账龄桶一个排序序号（"未到期"=0, "1-30天"=1 ...），方便图表按顺序排
    age_order_map = {}
    for i, v in enumerate(AGE_LAB):
        age_order_map[v] = i
    op["age_order"] = op["age_bucket"].map(age_order_map)

    # ---- 1.2 金额桶 ----
    op["amount_bucket"] = pd.cut(
        op["open_m"], [0, 0.01, 0.05, 0.1, 0.5, 1.0, 10 ** 9],
        labels=AMT_LAB, right=False).astype(str)
    amount_order_map = {}
    for i, v in enumerate(AMT_LAB):
        amount_order_map[v] = i
    op["amount_order"] = op["amount_bucket"].map(amount_order_map)

    # ---- 1.3 客户分组：按 cust_number 聚合，Top10 客户保留，其余合并成"其他 N 家" ----
    c = op.groupby("cust_number").open_m.sum().sort_values(ascending=False)
    n_other = max(len(c) - 10, 0)
    other = f"其他 {n_other} 家"
    top10 = list(c.head(10).index)
    top10_set = set(top10)
    cust_group_col = []
    for cn in op["cust_number"]:
        if cn in top10_set:
            cust_group_col.append(cn)        # Top10 客户：保留原编号
        else:
            cust_group_col.append(other)      # 其余客户：合并成"其他 N 家"
    op["cust_group"] = cust_group_col

    # 每个客户编号对应几种不同写法的客户名（最多 11 种），留个展示用的数
    name_variant_map = d.groupby("cust_number").name_customer.nunique()
    op["name_variants"] = op["cust_number"].map(name_variant_map)

    # ---- 1.4 付款条款分组：43 个值太多，Top6 + 其他条款 ----
    t = op.groupby("cust_payment_terms").open_m.sum().sort_values(ascending=False)
    t6 = list(t.head(6).index)
    t6_set = set(t6)
    terms_group_col = []
    for terms in op["cust_payment_terms"]:
        if terms in t6_set:
            terms_group_col.append(terms)
        else:
            terms_group_col.append("其他条款")
    op["terms_group"] = terms_group_col

    # ---- 1.5 挑出要进报表的列 ----
    keep = ["cust_number", "cust_group", "business_code", "terms_group",
            "age_bucket", "age_order", "amount_bucket", "amount_order",
            "open_m", "name_variants"]
    open_df = op[keep].copy()
    open_df["invoice_count"] = 1          # 每行就是一张发票，计数=1

    # ---- 1.6 月度汇总：全部发票的挂账额 + 未结额 + 逾期额 ----
    m = d.groupby("posting_month").agg(billed_m=("open_m", "sum")).reset_index()
    mo = op.groupby(op["posting_month"]).open_m.sum().reset_index()
    mo.columns = ["posting_month", "open_m"]
    m = m.merge(mo, on="posting_month", how="left").fillna(0)

    # 逾期月表：只取"账龄不是未到期"的行，按月汇总，再对齐到 m 的月份顺序
    overdue_op = op[op.age_bucket != AGE_LAB[0]]
    overdue_by_month = overdue_op.groupby("posting_month").open_m.sum()
    overdue_m_list = []
    for mon in m["posting_month"]:
        if mon in overdue_by_month.index:
            overdue_m_list.append(float(overdue_by_month[mon]))
        else:
            overdue_m_list.append(0.0)
    m["overdue_m"] = overdue_m_list

    # ---- 1.7 算一组关键事实数字，给报表标题/卡片用 ----
    top = c.iloc[0]
    facts = dict(
        total=total,
        rows=len(op),
        cust=op.cust_number.nunique(),
        overdue_pct=(op[op.age_bucket != AGE_LAB[0]].open_m.sum() / total * 100),
        top1_pct=top / total * 100,
        top1_val=top,
        top10_pct=c.head(10).sum() / total * 100,
        terms_top=t.index[0],
        terms_top_pct=t.iloc[0] / total * 100,
        biz_top=op.groupby("business_code").open_m.sum().idxmax(),
        biz_top_pct=op.groupby("business_code").open_m.sum().max() / total * 100,
        amt_top=op.groupby("amount_bucket").open_m.sum().idxmax(),
        amt_top_pct=op.groupby("amount_bucket").open_m.sum().max() / total * 100,
        peak_m=m.loc[m.billed_m.idxmax(), "posting_month"],
        peak_val=m.billed_m.max(),
        months=len(m),
        names=d.name_customer.nunique(),
        max_var=int(op.name_variants.max()),
    )
    return open_df, m, facts


# ------------------------------------------------------------------
# 2. 生成 Tableau 工作簿
# ------------------------------------------------------------------
def build_tableau(open_df, m, f):
    OPEN_DS, OPEN_TS, OPEN_CAP = "federated.0aropen", "textscan.0aropen", "未结发票"
    MON_DS, MON_TS, MON_CAP = "federated.0armonth", "textscan.0armonth", "月度挂账"

    # ar_open.csv 的字段定义：(字段名, 类型, 角色, 数据种类, 宽度, 中文标题)
    open_fields = [
        ("cust_number",   "string",  "dimension", "nominal",      129, "客户编号"),
        ("cust_group",    "string",  "dimension", "nominal",      129, "客户分组"),
        ("business_code", "string",  "dimension", "nominal",      129, "业务线"),
        ("terms_group",   "string",  "dimension", "nominal",      129, "付款条款"),
        ("age_bucket",    "string",  "dimension", "nominal",      129, "账龄"),
        ("age_order",      "integer", "measure",   "quantitative",  20, "账龄序"),
        ("amount_bucket", "string",  "dimension", "nominal",      129, "金额区间"),
        ("amount_order",   "integer", "measure",   "quantitative",  20, "金额序"),
        ("open_m",         "real",    "measure",   "quantitative",   5, "未结(百万)"),
        ("invoice_count",  "integer", "measure",   "quantitative",  20, "发票数"),
        ("name_variants",  "integer", "measure",   "quantitative",  20, "名字变体数"),
    ]
    # ar_month.csv 的字段定义
    mon_fields = [
        ("posting_month", "string", "dimension", "nominal",      129, "挂账月份"),
        ("billed_m",      "real",   "measure",   "quantitative",   5, "挂账(百万)"),
        ("open_m",        "real",   "measure",   "quantitative",   5, "未结(百万)"),
        ("overdue_m",     "real",   "measure",   "quantitative",   5, "逾期(百万)"),
    ]

    O, M = OPEN_DS, MON_DS
    S_OPEN = f"[{O}].[sum:open_m:qk]"

    # 6 张工作表配置：账龄结构 / 付款条款 / 客户集中度 / 业务线 / 月度挂账 / 金额区间
    sheets = [
        dict(name="1 账龄结构", ds=O,
             deps=[("string", "age_bucket", "dimension", "nominal", "none:age_bucket:nk", "None", "nominal"),
                   ("integer", "age_order", "measure", "quantitative", "sum:age_order:qk", "Sum", "quantitative"),
                   ("real", "open_m", "measure", "quantitative", "sum:open_m:qk", "Sum", "quantitative")],
             rows=f"[{O}].[none:age_bucket:nk]", cols=S_OPEN, mark="Bar",
             encodings=[("text", S_OPEN)],
             sort=(f"[{O}].[none:age_bucket:nk]", f"[{O}].[sum:age_order:qk]", "ASC"),
             labels=True, num_format=(S_OPEN, "*0.0"),
             axis_titles={"cols": "百万美元"},
             title=f"{f['overdue_pct']:.1f}% 未结金额已逾期"),

        dict(name="2 付款条款", ds=O,
             deps=[("string", "terms_group", "dimension", "nominal", "none:terms_group:nk", "None", "nominal"),
                   ("real", "open_m", "measure", "quantitative", "sum:open_m:qk", "Sum", "quantitative")],
             rows=f"[{O}].[none:terms_group:nk]", cols=S_OPEN, mark="Bar",
             encodings=[("text", S_OPEN)], labels=True,
             sort=(f"[{O}].[none:terms_group:nk]", S_OPEN, "DESC"),
             num_format=(S_OPEN, "*0.0"),
             title=f"{f['terms_top']} 一条款占 {f['terms_top_pct']:.1f}%"),

        dict(name="3 客户集中度", ds=O,
             deps=[("string", "cust_group", "dimension", "nominal", "none:cust_group:nk", "None", "nominal"),
                   ("real", "open_m", "measure", "quantitative", "sum:open_m:qk", "Sum", "quantitative")],
             rows=f"[{O}].[none:cust_group:nk]", cols=S_OPEN, mark="Bar",
             encodings=[("text", S_OPEN)], labels=True,
             sort=(f"[{O}].[none:cust_group:nk]", S_OPEN, "DESC"),
             num_format=(S_OPEN, "*0.0"),
             title=f"最大客户一家占 {f['top1_pct']:.1f}%"),

        dict(name="4 业务线", ds=O,
             deps=[("string", "business_code", "dimension", "nominal", "none:business_code:nk", "None", "nominal"),
                   ("real", "open_m", "measure", "quantitative", "sum:open_m:qk", "Sum", "quantitative")],
             rows=f"[{O}].[none:business_code:nk]", cols=S_OPEN, mark="Bar",
             encodings=[("text", S_OPEN)], labels=True,
             sort=(f"[{O}].[none:business_code:nk]", S_OPEN, "DESC"),
             num_format=(S_OPEN, "*0.0"),
             title=f"{f['biz_top']} 占 {f['biz_top_pct']:.1f}%"),

        dict(name="5 月度挂账", ds=M,
             deps=[("string", "posting_month", "dimension", "nominal", "none:posting_month:nk", "None", "nominal"),
                   ("real", "billed_m", "measure", "quantitative", "sum:billed_m:qk", "Sum", "quantitative")],
             rows=f"[{M}].[sum:billed_m:qk]", cols=f"[{M}].[none:posting_month:nk]",
             mark="Line", num_format=(f"[{M}].[sum:billed_m:qk]", "*0.0"),
             axis_titles={"rows": "百万美元"},
             title=f"{f['peak_m']} 单月挂账 {f['peak_val']:.0f}M"),

        dict(name="6 金额区间", ds=O,
             deps=[("string", "amount_bucket", "dimension", "nominal", "none:amount_bucket:nk", "None", "nominal"),
                   ("integer", "amount_order", "measure", "quantitative", "sum:amount_order:qk", "Sum", "quantitative"),
                   ("real", "open_m", "measure", "quantitative", "sum:open_m:qk", "Sum", "quantitative")],
             rows=f"[{O}].[none:amount_bucket:nk]", cols=S_OPEN, mark="Bar",
             encodings=[("text", S_OPEN)], labels=True,
             sort=(f"[{O}].[none:amount_bucket:nk]", f"[{O}].[sum:amount_order:qk]", "ASC"),
             num_format=(S_OPEN, "*0.0"),
             title=f"{f['amt_top']} 段占 {f['amt_top_pct']:.1f}%"),
    ]

    twbkit.configure(os.path.join(ROOT, "data", "bi"),
                      os.path.join(ROOT, "AccountsReceivable_Tableau105.twb"),
                      {OPEN_DS: OPEN_CAP, MON_DS: MON_CAP})
    out = twbkit.build(
        datasources=[
            dict(ds=OPEN_DS, ts=OPEN_TS, cap=OPEN_CAP, csv="ar_open.csv", fields=open_fields),
            dict(ds=MON_DS, ts=MON_TS, cap=MON_CAP, csv="ar_month.csv", fields=mon_fields),
        ],
        sheets=sheets,
        dash=dict(name="应收账款分析总览", title="应收账款 · 未结 334.2M", ncols=3, nrows=2),
    )
    twbkit.validate(out)


# ------------------------------------------------------------------
# 3. 生成 PowerBI 工程
# ------------------------------------------------------------------
# DAX 度量值定义：未结金额、未结发票数、逾期金额、逾期占比、最大客户占比
MEASURES = """
	measure '未结金额' = SUM('ar_open'[open_m])
		formatString: $#.0

	measure '未结发票' = COUNTROWS('ar_open')
		formatString: #,0

	measure '逾期金额' =
			CALCULATE([未结金额], 'ar_open'[age_bucket] <> "未到期")
		formatString: $#.0

	measure '逾期占比' = DIVIDE([逾期金额], [未结金额])
		formatString: 0.0%

	measure '最大客户占比' =
			DIVIDE(MAXX(ALL('ar_open'[cust_number]), [未结金额]), [未结金额])
		formatString: 0.0%
"""


def build_pbip(f):
    K.configure(proj="AccountsReceivable", logical_id="a1b2c3d4-1111-4a2b-9c3d-000000000001",
                root=ROOT, csv_dir=BI)
    K.build_semantic_model(tables=[
        dict(name="ar_open", csv="ar_open.csv", measures=MEASURES,
             desc="未结发票：10,000 笔，含账龄/条款/客户分组/金额分桶",
             money_cols=["open_m"], pct_cols=[]),
        # ar_month 不进 Power BI 模型：它和 ar_open 是不同聚合粒度（月汇总 vs 明细），
        # 建不上关系，又没有视觉引用它 → 模型视图里就是一张没连线的空表。
        # CSV 保留，Tableau 侧仍在用。要放进报表得先加一张月度趋势图。
    ])

    E = "ar_open"
    # 报表页上的 6 个视觉对象：标题 + 4 张卡片 + 账龄柱图 + 客户条形图
    vs = [
        K.textbox("v100", 16, 12, 1248, 36, "应收账款 · 账龄与催收"),
        K.visual("v101", "card", {"Values": [K.meas_ref(E, "未结金额")]},
                 16, 64, 303, 96, "未结金额（百万）", accent="primary", panel=True),
        K.visual("v102", "card", {"Values": [K.meas_ref(E, "未结发票")]},
                 331, 64, 303, 96, "未结发票", accent="primary", panel=True),
        K.visual("v103", "card", {"Values": [K.meas_ref(E, "逾期占比")]},
                 646, 64, 303, 96, "逾期占比", accent="alert", panel=True),
        K.visual("v104", "card", {"Values": [K.meas_ref(E, "最大客户占比")]},
                 961, 64, 303, 96, "最大客户占比", accent="alert", panel=True),
        K.visual("v105", "clusteredColumnChart",
                 {"Category": [K.col_ref(E, "age_bucket")],
                  "Y": [K.agg_ref(E, "open_m", 0)]},
                 16, 172, 764, 532, "逾期占 96.6%，61–90 天仍有 11.8%",
                 accent="primary", panel=True, labels=True),
        K.visual("v106", "clusteredBarChart",
                 {"Category": [K.col_ref(E, "cust_group")],
                  "Y": [K.agg_ref(E, "open_m", 0)]},
                 792, 172, 472, 532, "最大客户一家 16.7%",
                 accent="primary", panel=True, labels=True,
                 sort=(K.agg_ref(E, "open_m", 0), "Descending")),
    ]
    K.build_report(pages=[("ReportSection1", "1 账龄与催收", vs)])
    K.validate()


# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------
def main():
    print("[1/4] 数据准备 ...")
    open_df, m, f = prep()
    print(f"  未结 {f['rows']:,} 笔 / ${f['total']:.1f}M / {f['cust']:,} 个客户")
    print(f"  逾期占比 {f['overdue_pct']:.1f}% | 最大客户 {f['top1_pct']:.1f}% "
          f"| Top10 {f['top10_pct']:.1f}%")
    print(f"  客户名 {f['names']:,} 个 vs cust_number {f['cust']:,} 个（最多 {f['max_var']} 种写法）")

    print("[2/4] 导出 CSV ...")
    p1 = csvkit.export(open_df, os.path.join(BI, "ar_open.csv"))
    p2 = csvkit.export(m, os.path.join(BI, "ar_month.csv"))
    csvkit.check(p1)
    csvkit.check(p2)

    print("\n[3/4] Tableau 工作簿 ...")
    build_tableau(open_df, m, f)

    print("\n[4/4] PowerBI 工程 ...")
    build_pbip(f)
    print("\n完成。")


if __name__ == "__main__":
    main()
