from datetime import date, datetime, UTC

import polars as pl

from factor_matrix.industry_observation import (
    GAP_SAME_L1_FILLED, MEMBERSHIP_GAP, OBSERVED, _interval_rows,
)


def test_industry_gaps_fill_only_matching_adjacent_l1() -> None:
    history = pl.DataFrame({
        "classification_standard": ["SW2021"] * 4,
        "asset_id": ["A", "A", "B", "B"],
        "in_date": [date(2020, 1, 1), date(2020, 1, 10)] * 2,
        "out_date": [date(2020, 1, 5), None] * 2,
        "l1_code": ["L1", "L1", "OLD", "NEW"],
        "l1_name": ["one", "one", "old", "new"],
        "l2_code": ["L2A", "L2B", "OLD2", "NEW2"],
        "l2_name": ["two-a", "two-b", "old-2", "new-2"],
        "ingested_at": [datetime(2026, 8, 16, tzinfo=UTC)] * 4,
    })
    intervals = _interval_rows(history, "SW2021")
    a_gap = intervals.filter(
        (pl.col("asset_id") == "A")
        & (pl.col("industry_observation_state") == GAP_SAME_L1_FILLED)
    ).row(0, named=True)
    assert a_gap["effective_from"] == date(2020, 1, 6)
    assert a_gap["effective_to"] == date(2020, 1, 9)
    assert a_gap["l1_code"] == "L1"
    assert a_gap["l2_code"] is None
    assert a_gap["filled_from_adjacent"] is True
    b_gap = intervals.filter(
        (pl.col("asset_id") == "B")
        & (pl.col("industry_observation_state") == MEMBERSHIP_GAP)
    ).row(0, named=True)
    assert b_gap["l1_code"] is None
    assert b_gap["filled_from_adjacent"] is False
    assert intervals.filter(pl.col("industry_observation_state") == OBSERVED).height == 4
