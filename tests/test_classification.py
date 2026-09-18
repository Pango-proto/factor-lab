from datetime import UTC, date, datetime

import polars as pl
import pytest

from factor_matrix.classification import (
    industry_as_of,
    normalize_industry_classification,
    normalize_industry_membership,
)
from factor_matrix.storage import DataLake


def test_industry_membership_uses_inclusive_out_date(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    raw = pl.DataFrame(
        {
            "l1_code": ["OLD", "NEW"], "l1_name": ["旧行业", "新行业"],
            "l2_code": ["O2", "N2"], "l2_name": ["旧二级", "新二级"],
            "l3_code": ["O3", "N3"], "l3_name": ["旧三级", "新三级"],
            "ts_code": ["000001.SZ", "000001.SZ"], "name": ["甲", "甲"],
            "in_date": ["20220101", "20230701"], "out_date": ["20230630", None],
            "is_new": ["N", "Y"],
        }
    )
    normalized = normalize_industry_membership(raw, datetime(2026, 8, 7, tzinfo=UTC), "SW2021")
    assert normalized.get_column("available_at").unique().to_list() == [datetime(2026, 8, 7, tzinfo=UTC)]
    lake.replace("industry_membership_history", normalized)
    before = industry_as_of(lake, date(2023, 6, 30))
    boundary = industry_as_of(lake, date(2023, 7, 1))
    assert before.get_column("l1_name").to_list() == ["旧行业"]
    assert boundary.get_column("l1_name").to_list() == ["新行业"]


def test_industry_as_of_never_returns_future_membership(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    raw = pl.DataFrame(
        {
            "l1_code": ["NEW"], "l1_name": ["新行业"], "l2_code": ["N2"],
            "l2_name": ["新二级"], "l3_code": ["N3"], "l3_name": ["新三级"],
            "ts_code": ["000001.SZ"], "name": ["甲"], "in_date": ["20240101"],
            "out_date": [None], "is_new": ["Y"],
        }
    )
    lake.replace(
        "industry_membership_history",
        normalize_industry_membership(raw, datetime(2026, 8, 7, tzinfo=UTC), "SW2021"),
    )
    assert industry_as_of(lake, date(2023, 12, 31)).is_empty()


def test_industry_as_of_requires_requested_level(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    raw = pl.DataFrame(
        {
            "l1_code": ["L1", "L1"], "l1_name": ["一级", "一级"],
            "l2_code": [None, "L2"], "l2_name": [None, "二级"],
            "l3_code": [None, "L3"], "l3_name": [None, "三级"],
            "ts_code": ["000001.SZ", "000002.SZ"], "name": ["甲", "乙"],
            "in_date": ["20220101", "20220101"], "out_date": [None, None],
            "is_new": ["Y", "Y"],
        }
    )
    lake.replace(
        "industry_membership_history",
        normalize_industry_membership(raw, datetime(2026, 8, 7, tzinfo=UTC), "SW2021"),
    )
    assert industry_as_of(lake, date(2023, 7, 3), level="L1").height == 2
    l2 = industry_as_of(lake, date(2023, 7, 3), level="L2")
    assert l2.get_column("asset_id").to_list() == ["000002.SZ"]


def test_classification_tree_keeps_all_sw2021_levels() -> None:
    raw = pl.DataFrame(
        {
            "index_code": ["801000.SI", "801010.SI", "850111.SI"],
            "industry_name": ["一级", "二级", "三级"],
            "level": ["L1", "L2", "L3"],
            "industry_code": ["01", "0101", "010101"],
            "is_pub": ["1", "1", "1"],
            "parent_code": [None, "801000.SI", "801010.SI"],
            "src": ["SW2021", "SW2021", "SW2021"],
        }
    )
    normalized = normalize_industry_classification(
        raw, datetime(2026, 8, 7, tzinfo=UTC), "SW2021"
    )
    assert normalized.get_column("level").to_list() == ["L1", "L2", "L3"]
    assert normalized.get_column("parent_code").to_list() == [None, "801000.SI", "801010.SI"]


def test_industry_as_of_blocks_ambiguous_active_memberships(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.replace(
        "industry_membership_history",
        pl.DataFrame(
            {
                "asset_id": ["000001.SZ", "000001.SZ"],
                "l1_code": ["L1A", "L1B"], "l1_name": ["一级A", "一级B"],
                "l2_code": ["L2A", "L2B"], "l2_name": ["二级A", "二级B"],
                "l3_code": ["L3A", "L3B"], "l3_name": ["三级A", "三级B"],
                "in_date": [date(2020, 1, 1), date(2021, 1, 1)],
                "out_date": [None, None],
                "classification_standard": ["SW2021", "SW2021"],
            },
            schema_overrides={"out_date": pl.Date},
        ),
    )
    with pytest.raises(RuntimeError, match="PIT_INDUSTRY_AMBIGUOUS"):
        industry_as_of(lake, date(2026, 8, 6), level="L2")
