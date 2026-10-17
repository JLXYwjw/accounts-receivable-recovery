# -*- coding: utf-8 -*-
"""报告数字一致性测试：从 parquet/metrics/ml 重算关键数字，断言 HTML 含同数字。"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
HTML = ROOT / "report" / "应收对账分析报告.html"


def _html():
    return HTML.read_text(encoding="utf-8")


def _df():
    return pd.read_parquet(OUT / "ar_clean.parquet")


def test_open_rate_matches_html():
    df = _df()
    rate = (df["isOpen"] == 1).mean() * 100
    assert abs(rate - 20) < 0.1, rate
    assert "20%" in _html()


def test_open_amount_matches_html():
    df = _df()
    amt = df[df["isOpen"] == 1]["total_open_amount"].sum() / 1e8  # 亿
    assert abs(amt - 3.34) < 0.02, amt
    assert "$334M" in _html()  # 全未结清口径（报告用 $334M；卡片主推已到期未结清 $3.23亿）


def test_dso_matches_html():
    df = _df()
    paid = df[df["isOpen"] == 0]
    dso = paid["days_to_clear"].mean()
    assert abs(dso - 18.1) < 0.05, dso
    assert "18.1" in _html()


def test_overdue_rate_matches_html():
    df = _df()
    paid = df[df["isOpen"] == 0]
    rate = (paid["days_overdue"] > 0).mean() * 100
    assert abs(rate - 41.9) < 0.1, rate
    assert "41.9%" in _html()


def test_model_auc_matches_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    gbm_auc = ml["models"]["gbm"]["test_auc"]
    assert abs(gbm_auc - 0.912) < 0.01, gbm_auc
    assert "0.912" in _html()
    assert "0.920" in _html() or "92%" in _html()  # posting_month 主导的诚实声明


def test_top_term_matches_html():
    m = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    top = m["top_overdue_terms"][0]
    assert top["cust_payment_terms"] == "NAX2"
    assert abs(top["overdue_pct"] - 99.8) < 0.1
    assert "NAX2" in _html() and "99.8%" in _html()


def test_open_overdue_matches_html():
    # 已到期未结清（快照误读修正的核心口径）
    df = _df()
    snapshot = df["clear_date"].max()
    open_df = df[df["isOpen"] == 1]
    due_rate = (open_df["due_in_date"] < snapshot).mean() * 100
    assert abs(due_rate - 94.8) < 0.1, due_rate
    amt = open_df[open_df["due_in_date"] < snapshot]["total_open_amount"].sum() / 1e8
    assert abs(amt - 3.23) < 0.02, amt
    assert "$3.23亿" in _html() and "94.8%" in _html()


def test_queue_top50_matches_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert abs(ml["queue_top50_sum"] / 1e6 - 8.0) < 0.1
    assert ml["queue_top50_open"] == 50  # 催收队列全部为当前未结清
    assert "$8.0M" in _html()
