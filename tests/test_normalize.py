from datetime import UTC, datetime

import polars as pl

from factor_matrix.normalize import normalize_market, recompute_price_series


def test_total_return_uses_raw_price_and_adjustment_factor() -> None:
    daily = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20250102", "20250103"],
            "open": [10.0, 9.0], "high": [10.0, 9.0], "low": [10.0, 9.0],
            "close": [10.0, 9.0], "pre_close": [10.0, 10.0],
            "change": [0.0, -1.0], "pct_chg": [0.0, -10.0], "vol": [1.0, 1.0], "amount": [1.0, 1.0],
        }
    )
    adj = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20250102", "20250103"], "adj_factor": [1.0, 1.1]}
    )
    valuation = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"], "trade_date": ["20250103"], "close": [9.0],
            "turnover_rate": [1.0], "turnover_rate_f": [1.0], "volume_ratio": [1.0],
            "pe": [1.0], "pe_ttm": [1.0], "pb": [1.0], "ps": [1.0], "ps_ttm": [1.0],
            "dv_ratio": [1.0], "dv_ttm": [1.0], "total_share": [2.0], "float_share": [1.5],
            "free_share": [1.0], "total_mv": [18.0], "circ_mv": [13.5],
        }
    )
    limits = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20250102", "20250103"],
            "up_limit": [11.0, 11.0],
            "down_limit": [9.0, 9.0],
        }
    )
    prices, values = normalize_market(
        daily, adj, valuation, datetime.now(UTC), "20250103", limits
    )
    assert abs(prices.get_column("total_return")[1] - (-0.01)) < 1e-12
    assert prices.get_column("raw_close").to_list() == [10.0, 9.0]
    assert prices.get_column("gross_close_index").to_list() == [10.0, 9.9]
    assert prices.get_column("limit_up").to_list() == [11.0, 11.0]
    assert prices.get_column("limit_down").to_list() == [9.0, 9.0]
    assert values.get_column("total_mkt_cap")[0] == 180_000
    assert values.get_column("float_mkt_cap")[0] == 135_000


def test_qfq_anchor_is_fixed_to_end_date() -> None:
    daily = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20250102", "20250103"],
            "open": [10.0, 9.0], "high": [10.0, 9.0], "low": [10.0, 9.0], "close": [10.0, 9.0],
            "pre_close": [10.0, 10.0], "change": [0.0, -1.0], "pct_chg": [0.0, -10.0],
            "vol": [1.0, 1.0], "amount": [1.0, 1.0],
        }
    )
    adj = pl.DataFrame(
        {"ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20250102", "20250103"], "adj_factor": [1.0, 1.1]}
    )
    valuation = pl.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20250103"], "total_share": [1.0], "float_share": [1.0], "free_share": [1.0], "total_mv": [1.0], "circ_mv": [1.0]}
    )
    prices, _ = normalize_market(daily, adj, valuation, datetime.now(UTC), "20250103")
    assert prices.get_column("qfq_close").to_list() == [10.0 / 1.1, 9.0]


def test_recompute_fills_incremental_batch_boundary_and_reanchors() -> None:
    history = pl.DataFrame(
        {
            "asset_id": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": [
                __import__("datetime").date(2025, 1, 2),
                __import__("datetime").date(2025, 1, 3),
                __import__("datetime").date(2025, 1, 6),
            ],
            "close": [10.0, 11.0, 12.0],
            "adj_factor": [1.0, 1.0, 1.2],
            "qfq_close": [10.0, 11.0, 12.0],
            "qfq_anchor_date": [
                __import__("datetime").date(2025, 1, 3),
                __import__("datetime").date(2025, 1, 3),
                __import__("datetime").date(2025, 1, 6),
            ],
            "total_return": [None, 0.1, None],
        }
    )
    open_dates = pl.Series(
        "trade_date",
        [
            __import__("datetime").date(2025, 1, 2),
            __import__("datetime").date(2025, 1, 3),
            __import__("datetime").date(2025, 1, 6),
        ],
    )
    result = recompute_price_series(history, open_dates)
    assert result.get_column("qfq_anchor_date").n_unique() == 1
    assert result.get_column("total_return").null_count() == 1
    assert abs(result.get_column("total_return")[2] - (12.0 * 1.2 / 11.0 - 1)) < 1e-12


def test_recompute_does_not_bridge_unloaded_market_dates() -> None:
    history = pl.DataFrame(
        {
            "asset_id": ["000001.SZ", "000001.SZ"],
            "trade_date": [
                __import__("datetime").date(2025, 1, 2),
                __import__("datetime").date(2025, 1, 6),
            ],
            "close": [10.0, 12.0],
            "adj_factor": [1.0, 1.0],
        }
    )
    open_dates = pl.Series(
        "trade_date",
        [
            __import__("datetime").date(2025, 1, 2),
            __import__("datetime").date(2025, 1, 3),
            __import__("datetime").date(2025, 1, 6),
        ],
    )
    result = recompute_price_series(history, open_dates)
    assert result.get_column("total_return").to_list() == [None, None]
