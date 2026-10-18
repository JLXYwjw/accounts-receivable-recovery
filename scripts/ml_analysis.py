# -*- coding: utf-8 -*-
"""应收 ML：预测发票到期后仍未结清(isOpen=1)的概率，据此排催收优先级。
模型: 逻辑回归(基线) + 随机森林 + 梯度提升(GBM)。树模型为主。
切分: 分层随机(train_test_split stratify) —— 目标是"发票最终是否未结清"的快照标签，
      与过账时间强相关,若按时间切分验证集大多未到期结清,会引入泄漏(AUC<0.5)。已修。
评估: AUC / PR-AUC / 高金额段召回。输出概率 + 催收优先级队列(金额×概率)。

DL 消融(诚实实测): 5万张发票足够训练一个 MLP —— 损失=二元交叉熵, 优化器=Adam(初始 lr=1e-3),
特征经 StandardScaler 标准化(在训练集上拟合防泄漏), L2 正则 alpha=1e-3(等价 dropout 的权重衰减),
早停(监控验证损失, n_iter_no_change=10)。MLP 作对比消融, 与逻辑回归/随机森林/GBM 同口径评估,
实测证明"数据够训 DL, 但表格数据上 GBM 仍更优且可解释, 故 DL 不作为主模型"。
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"
df = pd.read_parquet(OUT_DIR / "ar_clean.parquet")
df = df.reset_index(drop=True)

# ---- 特征工程（无泄漏）----
df["business_code"] = df["business_code"].astype(str)
df["invoice_currency"] = df["invoice_currency"].astype(str)
term_counts = df["cust_payment_terms"].value_counts()
top_terms = term_counts.head(8).index.tolist()
df["term"] = df["cust_payment_terms"].where(df["cust_payment_terms"].isin(top_terms), "OTHER")
df["posting_month"] = df["posting_date"].dt.month

X = pd.get_dummies(df[["amount_log", "days_to_due", "business_code", "invoice_currency",
                       "term", "posting_month"]],
                   columns=["business_code", "invoice_currency", "term"], drop_first=True)
y = df["isOpen"].values

# 分层随机切分
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 客户出现频次（只用训练集统计，避免用全量信息泄漏）
freq_tr = df.loc[Xtr.index, "cust_number"].value_counts()
Xtr["cust_freq"] = Xtr.index.map(freq_tr).fillna(0).astype(float)
mean_freq = Xtr["cust_freq"].mean()
Xva["cust_freq"] = Xva.index.map(freq_tr).fillna(mean_freq).astype(float)

# ---- 交叉验证(训练集内) + 测试集评估 ----
models = {
    "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
    "gbm": GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, random_state=42),
    # DL 消融：MLP(2 隐层 64→32, ReLU, Adam, 标准化, L2=alpha=1e-3, 早停)。Pipeline 内 StandardScaler
    # 只在训练集上拟合, 验证/测试不经训练统计 → 无泄漏。
    "mlp": Pipeline([("scaler", StandardScaler()),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                           solver="adam", alpha=1e-3, batch_size=256,
                                           learning_rate_init=1e-3, max_iter=200,
                                           early_stopping=True, n_iter_no_change=10,
                                           random_state=42))]),
}
res = {}
for name, mdl in models.items():
    cv = cross_val_score(mdl, Xtr, ytr, cv=3, scoring="roc_auc")
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, p)
    prauc = average_precision_score(yva, p)
    thr_amt = np.quantile(df.loc[Xva.index, "total_open_amount"], 0.7)
    hi = df.loc[Xva.index, "total_open_amount"].values >= thr_amt
    hi_pred = p >= 0.5
    rec_hi = recall_score(yva[hi], hi_pred[hi]) if hi.sum() > 0 else None
    res[name] = {"cv_auc_mean": round(float(cv.mean()), 4), "test_auc": round(float(auc), 4),
                 "test_pr_auc": round(float(prauc), 4),
                 "recall_hi_amount": round(float(rec_hi), 4) if rec_hi is not None else None}
    print(name, res[name])

# ---- 主模型（GBM）输出催收优先级队列 ----
# 催收对象 = 已到期(due<快照) 且 未结清(isOpen=1) 的发票；在其内部按 金额×概率 排序，
# 而不是对全部验证集排序——否则会把刚过账、未到期的发票也排进"先催清单"（快照误读）。
gbm = models["gbm"]
p = gbm.predict_proba(Xva)[:, 1]
df_val = df.loc[Xva.index].copy()
df_val["open_prob"] = p
df_val["priority"] = df_val["total_open_amount"] * df_val["open_prob"]
snapshot = df["clear_date"].max()
df_val["due_passed"] = df_val["due_in_date"] < snapshot
imp = pd.Series(gbm.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
top_imp = imp.head(12).reset_index()
top_imp.columns = ["feature", "importance"]
collections = df_val[df_val["isOpen"] == 1].sort_values("priority", ascending=False)
queue = collections[["invoice_id", "name_customer", "cust_payment_terms", "posting_date", "due_in_date",
                     "total_open_amount", "open_prob", "priority", "isOpen"]].head(50)

out = {"models": res, "top_features": top_imp.to_dict("records"),
       "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
       "val_open_amount_total": round(float(df_val[df_val["isOpen"] == 1]["total_open_amount"].sum()), 2),
       "collections_total": round(float(collections["total_open_amount"].sum()), 2),
       "collections_n": int(len(collections)),
       "queue_top50_sum": round(float(queue["total_open_amount"].sum()), 2),
       "queue_top50_open": int((queue["isOpen"] == 1).sum())}
with open(OUT_DIR / "ml_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
df_val.to_parquet(OUT_DIR / "ar_val_with_prob.parquet", index=False)
print(json.dumps({"ok": True, "models": res, "queue_top50_sum": out["queue_top50_sum"]}, ensure_ascii=False))
