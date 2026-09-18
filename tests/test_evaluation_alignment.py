from datetime import date
from pathlib import Path

import polars as pl

from factor_matrix.calculation.l2.estimation_panel import (
    TRADABLE_UNIVERSE_PATH,
    _global_exposure_date_mapping,
    attach_next_day_execution_eligibility,
)


def test_exposure_shift_uses_one_global_trading_calendar() -> None:
    dates = [
        date(2024, 9, 27),
        date(2024, 9, 30),
        date(2024, 10, 8),
        date(2024, 10, 9),
        date(2024, 10, 10),
    ]

    previous = _global_exposure_date_mapping(
        dates, start=date(2024, 10, 8), end=date(2024, 10, 9), offset=-1
    )
    following = _global_exposure_date_mapping(
        dates, start=date(2024, 10, 8), end=date(2024, 10, 9), offset=1
    )

    assert previous.to_dicts() == [
        {"trade_date": date(2024, 10, 8), "exposure_date_m1": date(2024, 9, 30)},
        {"trade_date": date(2024, 10, 9), "exposure_date_m1": date(2024, 10, 8)},
    ]
    assert following.to_dicts() == [
        {"trade_date": date(2024, 10, 8), "exposure_date_p1": date(2024, 10, 9)},
        {"trade_date": date(2024, 10, 9), "exposure_date_p1": date(2024, 10, 10)},
    ]


def test_execution_filter_uses_next_day_facts_and_signal_day_liquidity(
    tmp_path: Path,
) -> None:
    universe_path = tmp_path / TRADABLE_UNIVERSE_PATH
    universe_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2024, 1, 3)] * 2,
        "asset_id": ["A", "B"],
        "listed_as_of": [True, True],
        "is_st": [False, False],
        "is_suspended": [False, False],
        "board_id": ["MAIN", "MAIN"],
        "days_since_exchange_list": [80, 80],
        "base_tradable_without_listing_age": [True, True],
    }).write_parquet(universe_path)
    prices_path = tmp_path / "silver/prices_daily/data.parquet"
    prices_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2024, 1, 3)] * 2,
        "asset_id": ["A", "B"],
        "limit_up": [11.0, 22.0],
        "limit_down": [9.0, 18.0],
        "raw_open": [11.0, 20.0],
        "raw_high": [11.0, 21.0],
        "raw_low": [11.0, 19.0],
        "raw_close": [11.0, 20.0],
    }).write_parquet(prices_path)
    panel = pl.DataFrame({
        "trade_date": [date(2024, 1, 2)] * 2,
        "asset_id": ["A", "B"],
        "label_start_date_5": [date(2024, 1, 3)] * 2,
        "amount": [10.0, 100.0],
    })

    result = attach_next_day_execution_eligibility(
        panel, tmp_path, primary_horizon=5, liquidity_floor_quantile=0.20
    ).sort("asset_id")

    assert result["execution_can_buy"].to_list() == [False, True]
    assert result["execution_eligible"].to_list() == [False, True]
    assert result["execution_trade_date"].to_list() == [
        date(2024, 1, 3), date(2024, 1, 3)
    ]
