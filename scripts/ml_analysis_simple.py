# -*- coding: utf-8 -*-
"""应收 ML —— 基础可读版（ml_analysis_simple.py）
=================================================

  1. 读入清洗好的 ar_clean.parquet
  2. 做特征工程：把业务代码/币种/付款条款转成字符串，把出现次数最多的
     8 种付款条款保留、其余归成 "OTHER"，再加一个"过账月份"特征
  3. 用分层随机切分把数据分成训练集(80%) / 验证集(20%)
  4. 训练 4 个模型：逻辑回归(基线)、随机森林、梯度提升(GBM)、MLP(神经网络消融)
  5. 在验证集上评估 AUC / PR-AUC / 高金额段召回率
  6. 用主模型 GBM 给验证集每张发票算"未结清概率"，按 金额×概率 排出
     Top50 催收优先清单
  7. 写出 ml_result.json 和 ar_val_with_prob.parquet

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么。

运行方式（不带参数）：
    python scripts/ml_analysis_simple.py

English:
AR machine learning (plain, readable version).

Steps:
  1. Read the cleaned ar_clean.parquet
  2. Feature engineering: turn business code / currency / payment terms
     into strings; keep the 8 most frequent payment terms and fold the
     rest into "OTHER"; add a posting-month feature
  3. Stratified random split: train (80%) / validation (20%)
  4. Train 4 models: logistic regression (baseline), random forest,
     gradient boosting (GBM), MLP (neural-net ablation)
  5. Evaluate on validation: AUC / PR-AUC / recall on high-amount segment
  6. Use GBM (main model) to score each validation invoice's
     "open probability", rank the Top-50 collection-priority list by
     amount x probability
  7. Write ml_result.json and ar_val_with_prob.parquet

The code uses the plainest style on purpose; every step states what it
does and why.

Run (no arguments):
    python scripts/ml_analysis_simple.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "output"


def main():
    # ---- 1. 读入清洗好的数据 ----
    df = pd.read_parquet(OUT_DIR / "ar_clean.parquet")
    df = df.reset_index(drop=True)

    # ---- 2. 特征工程（注意不能用全量信息，避免泄漏）----
    # 业务代码、币种先统一转成字符串，后面好做 one-hot
    df["business_code"] = df["business_code"].astype(str)
    df["invoice_currency"] = df["invoice_currency"].astype(str)

    # 付款条款种类太多，只保留出现次数最多的 8 种，其余统统归成 "OTHER"
    term_counts = df["cust_payment_terms"].value_counts()
    top_terms = term_counts.head(8).index.tolist()
    top_terms_set = set(top_terms)     # 转成集合，后面判断"在不在里面"更快

    term_col = []
    for v in df["cust_payment_terms"]:
        if pd.isna(v):
            term_col.append("OTHER")           # 缺失条款：归到 OTHER
        elif v in top_terms_set:
            term_col.append(v)                 # 高频条款：原样保留
        else:
            term_col.append("OTHER")            # 低频条款：归到 OTHER
    df["term"] = term_col

    # 过账月份：从日期里取出 1~12 的数字
    df["posting_month"] = df["posting_date"].dt.month

    # ---- 3. 组装特征矩阵 X 和标签 y ----
    # 对业务代码/币种/term 这三个类别变量做 one-hot（drop_first 去掉第一列防共线）
    X = pd.get_dummies(
        df[["amount_log", "days_to_due", "business_code", "invoice_currency",
            "term", "posting_month"]],
        columns=["business_code", "invoice_currency", "term"],
        drop_first=True,
    )
    y = df["isOpen"].values

    # ---- 4. 分层随机切分训练/验证集 ----
    # 用分层切分而不是按时间切分：因为"发票最终是否未结清"这个标签本身
    # 就和过账时间强相关，按时间切会让验证集大多还没到结清时点，评估失真。
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ---- 5. 加一个客户出现频次特征（只用训练集统计，防泄漏）----
    # 训练集里每个客户出现了多少次；验证集里没见过的客户用训练集的平均频次兜底。
    freq_tr = df.loc[Xtr.index, "cust_number"].value_counts()

    train_freq = Xtr.index.map(freq_tr)       # 训练集客户频次
    train_freq = train_freq.fillna(0)          # 训练集里没匹配上的（不会出现，兜底 0）
    Xtr["cust_freq"] = train_freq.astype(float)

    mean_freq = Xtr["cust_freq"].mean()        # 训练集频次的平均值，给验证集兜底用
    val_freq = Xva.index.map(freq_tr)
    val_freq = val_freq.fillna(mean_freq)      # 验证集里训练集没见过的客户，用平均值
    Xva["cust_freq"] = val_freq.astype(float)

    # ---- 6. 定义 4 个模型 ----
    models = {
        # 逻辑回归：当基线
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
        # 随机森林：300 棵树，最大深度 8，叶子至少 20 个样本
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42),
        # 梯度提升 GBM：主模型
        "gbm": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42),
        # MLP 消融：2 个隐层(64→32)，ReLU，Adam，标准化，L2 正则，早停。
        # StandardScaler 放在 Pipeline 里，保证只用训练集统计量，不泄漏。
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", alpha=1e-3, batch_size=256,
                learning_rate_init=1e-3, max_iter=200,
                early_stopping=True, n_iter_no_change=10,
                random_state=42)),
        ]),
    }

    # ---- 7. 逐个模型：3 折交叉验证 + 验证集评估 ----
    res = {}
    for name, mdl in models.items():
        cv = cross_val_score(mdl, Xtr, ytr, cv=3, scoring="roc_auc")
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xva)[:, 1]                 # 预测成 1 的概率
        auc = roc_auc_score(yva, p)
        prauc = average_precision_score(yva, p)
        # 高金额段召回：只看验证集里金额排前 30% 的大额发票，预测概率>=0.5 的召回率
        thr_amt = np.quantile(df.loc[Xva.index, "total_open_amount"], 0.7)
        hi = df.loc[Xva.index, "total_open_amount"].values >= thr_amt
        hi_pred = p >= 0.5
        if hi.sum() > 0:
            rec_hi = recall_score(yva[hi], hi_pred[hi])
        else:
            rec_hi = None
        if rec_hi is not None:
            rec_hi_out = round(float(rec_hi), 4)
        else:
            rec_hi_out = None
        res[name] = {
            "cv_auc_mean": round(float(cv.mean()), 4),
            "test_auc": round(float(auc), 4),
            "test_pr_auc": round(float(prauc), 4),
            "recall_hi_amount": rec_hi_out,
        }
        print(name, res[name])

    # ---- 8. 用主模型 GBM 输出催收优先队列 ----
    # 催收对象 = 已到期(due<快照) 且 未结清(isOpen=1) 的发票；
    # 在它们内部按 金额×概率 排序，而不是对整个验证集排序，
    # 否则会把刚过账、还没到期的发票也排进"先催清单"。
    gbm = models["gbm"]
    p = gbm.predict_proba(Xva)[:, 1]
    df_val = df.loc[Xva.index].copy()
    df_val["open_prob"] = p
    df_val["priority"] = df_val["total_open_amount"] * df_val["open_prob"]

    snapshot = df["clear_date"].max()
    df_val["due_passed"] = df_val["due_in_date"] < snapshot

    # 特征重要性（GBM 自带），取前 12
    imp = pd.Series(gbm.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    top_imp = imp.head(12).reset_index()
    top_imp.columns = ["feature", "importance"]

    # 未结清发票按优先级从高到低排，取 Top50
    collections = df_val[df_val["isOpen"] == 1].sort_values("priority", ascending=False)
    queue = collections[
        ["invoice_id", "name_customer", "cust_payment_terms", "posting_date", "due_in_date",
         "total_open_amount", "open_prob", "priority", "isOpen"]
    ].head(50)

    # ---- 9. 写出结果 ----
    out = {
        "models": res,
        "top_features": top_imp.to_dict("records"),
        "n_train": int(Xtr.shape[0]),
        "n_val": int(Xva.shape[0]),
        "val_open_amount_total": round(float(df_val[df_val["isOpen"] == 1]["total_open_amount"].sum()), 2),
        "collections_total": round(float(collections["total_open_amount"].sum()), 2),
        "collections_n": int(len(collections)),
        "queue_top50_sum": round(float(queue["total_open_amount"].sum()), 2),
        "queue_top50_open": int((queue["isOpen"] == 1).sum()),
    }
    with open(OUT_DIR / "ml_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    df_val.to_parquet(OUT_DIR / "ar_val_with_prob.parquet", index=False)

    print(json.dumps({"ok": True, "models": res,
                      "queue_top50_sum": out["queue_top50_sum"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
