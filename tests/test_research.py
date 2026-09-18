from datetime import UTC, date, datetime
import json

import polars as pl
import pytest

from factor_matrix.research import (
    UNIVERSE_CHUNK_LOOKBACK_DAYS,
    MatrixConfig,
    build_feature_values,
    build_market_universe,
    build_tradable_universe,
    resolve_book_features,
)
from factor_matrix.storage import DataLake
from factor_matrix.revisioned_silver import SilverVersionLedger


def test_market_universe_keeps_bse_as_an_independent_board_by_default(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    as_of = date(2026, 8, 6)
    lake.replace(
        "security_master",
        pl.DataFrame(
            {
                "asset_id": ["600000.SH", "920001.BJ"],
                "name": ["沪市样本", "北交所样本"],
                "exchange": ["SSE", "BSE"],
                "list_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "delist_date": [None, None],
            },
            schema_overrides={"delist_date": pl.Date},
        ),
    )
    lake.replace(
        "trade_calendar",
        pl.DataFrame({"cal_date": [as_of], "is_open": [1]}),
    )
    lake.replace(
        "prices_daily",
        pl.DataFrame(
            {
                "trade_date": [as_of, as_of],
                "asset_id": ["600000.SH", "920001.BJ"],
                "raw_open": [10.0, 20.0],
                "raw_high": [10.5, 20.5],
                "raw_low": [9.5, 19.5],
                "raw_close": [10.0, 20.0],
                "gross_open_index": [10.0, 20.0],
                "gross_close_index": [10.0, 20.0],
                "adj_factor": [1.0, 1.0],
                "limit_up": [11.0, 22.0],
                "limit_down": [9.0, 18.0],
                "total_return": [0.01, 0.02],
            }
        ),
    )
    lake.replace(
        "valuation_daily",
        pl.DataFrame(
            {
                "trade_date": [as_of, as_of],
                "asset_id": ["600000.SH", "920001.BJ"],
                "total_mkt_cap": [100.0, 200.0],
                "float_mkt_cap": [80.0, 150.0],
                "pb": [1.0, 2.0],
                "pe_ttm": [8.0, 12.0],
                "turnover_rate": [0.5, 1.0],
            }
        ),
    )
    lake.replace(
        "industry_membership_history",
        pl.DataFrame(
            {
                "asset_id": ["600000.SH", "920001.BJ"],
                "l1_code": ["801780.SI", "801780.SI"],
                "l1_name": ["银行", "银行"],
                "l2_code": ["851911.SI", "851911.SI"],
                "l2_name": ["国有大型银行Ⅱ", "国有大型银行Ⅱ"],
                "l3_code": ["85191101.SI", "85191101.SI"],
                "l3_name": ["国有大型银行Ⅲ", "国有大型银行Ⅲ"],
                "in_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "out_date": [None, None],
                "classification_standard": ["SW2021", "SW2021"],
            },
            schema_overrides={"out_date": pl.Date},
        ),
    )

    universe = build_market_universe(
        lake, as_of, MatrixConfig(min_listing_trading_days=1)
    )

    assert universe.height == 2
    assert universe.filter(pl.col("eligible_market_matrix")).get_column("asset_id").to_list() == [
        "600000.SH", "920001.BJ"
    ]
    assert universe.select("asset_id", "board_id").to_dicts() == [
        {"asset_id": "600000.SH", "board_id": "MAIN"},
        {"asset_id": "920001.BJ", "board_id": "BSE"},
    ]
    assert universe.filter(pl.col("reason_bse")).is_empty()
    assert universe.filter(pl.col("asset_id") == "600000.SH").get_column("industry").item() == "国有大型银行Ⅱ"


def test_feature_values_are_point_in_time_tagged(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    as_of = date(2026, 8, 6)
    # Reuse the full integration fixture through the first test's public contract is
    # intentionally avoided; this test focuses on the feature-store invariant.
    universe = pl.DataFrame(
        {
            "asset_id": ["600000.SH"],
            "eligible_market_matrix": [True],
            "raw_open": [10.0],
            "raw_high": [10.5],
            "raw_low": [9.5],
            "raw_close": [10.0],
            "gross_open_index": [10.0],
            "gross_close_index": [10.0],
            "adj_factor": [1.0],
            "limit_up": [11.0],
            "limit_down": [9.0],
            "total_return": [0.01],
            "total_mkt_cap": [100.0],
            "float_mkt_cap": [80.0],
            "log_float_mkt_cap": [4.382],
            "pb": [1.0],
            "pe_ttm": [8.0],
            "turnover_rate": [0.5],
            "book_equity": [50.0],
            "financial_available_at": [as_of],
        }
    )

    features = build_feature_values(universe, as_of)

    assert features.height == 17
    assert features.get_column("available_at").max() == as_of
    assert features.get_column("feature_id").n_unique() == 17


def test_book_feature_resolver_rejects_future_revision(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    as_of = date(2026, 8, 6)
    lake.replace(
        "financial_pit",
        pl.DataFrame(
            {
                "asset_id": ["600000.SH", "600000.SH"],
                "report_period": [date(2025, 12, 31), date(2025, 12, 31)],
                "available_at": [date(2026, 4, 21), date(2026, 8, 7)],
                "first_seen_at": [
                    date(2026, 4, 20),
                    date(2026, 8, 6),
                ],
                "book_equity": [100.0, 999.0],
            }
        ),
    )

    result = resolve_book_features(lake, as_of, MatrixConfig())

    assert result.height == 1
    assert result.get_column("book_equity").item() == 100.0
    assert result.get_column("financial_available_at").item() == date(2026, 4, 21)


def test_tradable_universe_keeps_signal_date_asset_that_does_not_survive_endpoint(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    first = date(2026, 8, 5)
    endpoint = date(2026, 8, 6)
    ingested = datetime(2026, 8, 7, tzinfo=UTC)
    lake.replace(
        "security_master",
        pl.DataFrame(
            {
                "asset_id": ["DELIST.SH", "LIVE.SH"],
                "symbol": ["600001", "600002"],
                "exchange": ["SSE", "SSE"],
                "list_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "delist_date": [endpoint, None],
                "ingested_at": [ingested, ingested],
            },
            schema_overrides={"delist_date": pl.Date},
        ),
    )
    lake.replace(
        "trade_calendar",
        pl.DataFrame({
            "exchange": ["SSE", "SSE", "SSE"],
            "cal_date": [date(2026, 8, 4), first, endpoint],
            "is_open": [1, 1, 1],
        }),
    )
    lake.replace(
        "prices_daily",
        pl.DataFrame(
            {
                "trade_date": [first, first, endpoint],
                "asset_id": ["DELIST.SH", "LIVE.SH", "LIVE.SH"],
                "raw_close": [10.0, 20.0, 21.0],
                "raw_open": [10.0, 20.0, 21.0],
                "raw_high": [10.5, 20.5, 21.5],
                "raw_low": [9.5, 19.5, 20.5],
                "limit_up": [11.0, 22.0, 23.1],
                "limit_down": [9.0, 18.0, 18.9],
            }
        ),
    )
    lake.replace(
        "valuation_daily",
        pl.DataFrame(
            {
                "trade_date": [first, first, endpoint],
                "asset_id": ["DELIST.SH", "LIVE.SH", "LIVE.SH"],
                "float_mkt_cap": [100.0, 200.0, 210.0],
            }
        ),
    )
    lake.replace(
        "industry_membership_history",
        pl.DataFrame(
            {
                "asset_id": ["DELIST.SH", "LIVE.SH"],
                "l1_code": ["L1", "L1"],
                "l1_name": ["一级", "一级"],
                "l2_code": ["L2", "L2"],
                    "l2_name": ["二级", "二级"],
                    "l3_code": ["L3", "L3"],
                    "l3_name": ["三级", "三级"],
                "in_date": [date(2020, 1, 1), date(2020, 1, 1)],
                "out_date": [None, None],
                "classification_standard": ["SW2021", "SW2021"],
            },
            schema_overrides={"out_date": pl.Date},
        ),
    )
    lake.replace(
        "suspensions_daily",
        pl.DataFrame(
            {
                "trade_date": [endpoint],
                "asset_id": ["LIVE.SH"],
                "suspend_type": ["R"],
            }
        ),
    )
    lake.replace(
        "stock_st_daily",
        pl.DataFrame({"trade_date": [endpoint], "asset_id": ["LIVE.SH"]}),
    )
    lake.replace(
        "price_limits_daily",
        pl.DataFrame(
            {
                "trade_date": [first, first, endpoint],
                "asset_id": ["DELIST.SH", "LIVE.SH", "LIVE.SH"],
                "limit_up": [11.0, 22.0, 23.1],
                "limit_down": [9.0, 18.0, 18.9],
            }
        ),
    )
    lake.replace(
        "security_daily_state",
        pl.DataFrame({
            "trade_date": [first, first, endpoint],
            "asset_id": ["DELIST.SH", "LIVE.SH", "LIVE.SH"],
            "in_a_share_scope": [True, True, True],
            "listed_as_of": [True, True, True],
            "is_st": [False, False, True],
            "is_suspended": [False, False, False],
            "board_id": ["MAIN", "MAIN", "MAIN"],
            "venue": ["SSE", "SSE", "SSE"],
            "exchange_list_date": [date(2020, 1, 1)] * 3,
            "days_since_exchange_list": [100, 100, 101],
            "observation_state": ["TRADED", "TRADED", "TRADED"],
        }),
    )
    artifacts = {
        table: {"path": f"silver/{table}/data.parquet"}
        for table in (
            "security_daily_state", "prices_daily", "valuation_daily",
            "industry_membership_history", "trade_calendar",
        )
    }
    base_dir = lake.metadata / "base_manifests"
    base_dir.mkdir(parents=True)
    (base_dir / "legacy_base_v1.json").write_text(json.dumps({
        "snapshot_id": "base", "frozen_at": "2026-08-07T00:00:00+00:00",
        "artifacts": artifacts,
    }))
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=ingested, changes=[], quality_gate={"status": "passed"},
    )

    metadata = build_tradable_universe(
        lake, first, endpoint, MatrixConfig(min_listing_trading_days=1)
    )
    universe = pl.read_parquet(lake.root / metadata["artifact"])

    delisted = universe.filter(pl.col("asset_id") == "DELIST.SH")
    assert delisted.get_column("trade_date").to_list() == [first]
    assert delisted.get_column("is_tradable").item()
    assert metadata["counts"]["future_conditioned_rows"] == 0
    assert universe.get_column("source_snapshot_hash").n_unique() == 1
    assert UNIVERSE_CHUNK_LOOKBACK_DAYS == 0
    assert metadata["policy"]["chunking_contract"] == {
        "boundary": "calendar_month",
        "field_semantics": "row_local_point_in_time_only",
        "lookback_days": 0,
        "rolling_fields": "forbidden_use_separate_upstream_artifact",
    }
