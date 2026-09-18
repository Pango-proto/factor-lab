from datetime import UTC, date, datetime

import polars as pl

from factor_matrix.normalize import build_returns_daily


def test_suspension_zero_and_resumption_recognizes_accumulated_return() -> None:
    prices = pl.DataFrame(
        {
            "trade_date": [date(2026, 8, 3), date(2026, 8, 6)],
            "asset_id": ["600000.SH", "600000.SH"],
            "close": [10.0, 11.0],
            "adj_factor": [1.0, 1.0],
        }
    )
    suspensions = pl.DataFrame(
        {
            "trade_date": [date(2026, 8, 4), date(2026, 8, 5), date(2026, 8, 6)],
            "asset_id": ["600000.SH"] * 3,
            "suspend_type": ["S", "S", "R"],
        }
    )
    result = build_returns_daily(prices, suspensions, datetime.now(UTC))
    values = result.get_column("total_return").to_list()
    assert values[:3] == [None, 0.0, 0.0]
    assert abs(values[3] - 0.1) < 1e-12
    assert result.get_column("return_source").to_list() == [
        "price",
        "suspension",
        "suspension",
        "resumption",
    ]
