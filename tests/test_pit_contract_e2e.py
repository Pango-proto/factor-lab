from datetime import date, timedelta

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.pit_contract_e2e import (
    build_alignment_rows, summarize_alignment,
)


def test_alignment_canaries_and_placebo() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    rng = np.random.default_rng(11)
    values = rng.normal(size=(len(dates), 20))
    rows = []
    for index, trade_date in enumerate(dates):
        for asset in range(20):
            rows.append({
                "trade_date": trade_date,
                "asset_id": f"A{asset:02d}",
                "total_return": float(values[index, asset]),
            })
    source = pl.DataFrame(rows)
    aligned = build_alignment_rows(source, sample_days=10, minimum_cross_section=10)
    summary, placebo = summarize_alignment(aligned, permutations=1000, seed=7)
    assert summary["t_plus_1_factor_min_daily_ic"] == 1.0
    assert abs(summary["t_plus_5_factor_abs_mean_daily_ic"]) < 0.15
    assert abs(summary["placebo_mean_ic"]) < 0.03
    assert placebo.height == 1000


def test_alignment_requires_five_future_dates() -> None:
    source = pl.DataFrame({
        "trade_date": [date(2026, 1, i) for i in range(1, 8)],
        "asset_id": ["A"] * 7,
        "total_return": [0.01] * 7,
    })
    try:
        build_alignment_rows(source, sample_days=5, minimum_cross_section=1)
    except ValueError as exc:
        assert str(exc) == "PIT_E2E_INSUFFICIENT_DATE_COVERAGE"
    else:
        raise AssertionError("expected insufficient date coverage")
