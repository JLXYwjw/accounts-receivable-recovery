# -*- coding: utf-8 -*-
"""清洗口径测试：ar_clean.parquet 的结构与一致性。"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"


def _df():
    return pd.read_parquet(OUT / "ar_clean.parquet")


def test_parquet_exists_and_shape():
    df = _df()
    assert len(df) == 50000, f"expect 50000 rows, got {len(df)}"
    assert df.shape[1] == 23, f"expect 23 cols, got {df.shape[1]}"


def test_area_business_dropped():
    df = _df()
    assert "area_business" not in df.columns


def test_posting_date_valid():
    df = _df()
    assert df["posting_date"].notna().all()


def test_isopen_clear_consistency():
    # clear_date 缺失 == isOpen=1（全部一致）
    df = _df()
    assert (df["clear_date"].isna() == (df["isOpen"] == 1)).all()


def test_split_ratio():
    df = _df()
    assert (df["split"] == "train").sum() == 40000
    assert (df["split"] == "val").sum() == 10000


def test_isopen_dist():
    df = _df()
    assert df["isOpen"].value_counts().to_dict() == {0: 40000, 1: 10000}
