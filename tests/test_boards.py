from datetime import UTC, date, datetime
import json

import polars as pl

from factor_matrix.boards import board_quality, normalize_board_membership, sync_board_membership
from factor_matrix.board_benchmarks import build_board_benchmarks
from factor_matrix.storage import DataLake
from factor_matrix.revisioned_silver import SilverVersionLedger


def security_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["600001.SH", "300001.SZ", "688001.SH", "920001.BJ"],
            "symbol": ["600001", "300001", "688001", "920001"],
            "market": ["主板", "创业板", "科创板", "北交所"],
            "exchange": ["SSE", "SZSE", "SSE", "BSE"],
            "list_date": [date(2020, 1, 1)] * 4,
            "delist_date": [None] * 4,
            "ingested_at": [datetime(2026, 8, 13, tzinfo=UTC)] * 4,
        },
        schema_overrides={"delist_date": pl.Date},
    )


def test_board_contract_classifies_all_four_markets_once() -> None:
    result = normalize_board_membership(security_fixture())
    assert dict(result.select("asset_id", "board_id").iter_rows()) == {
        "600001.SH": "MAIN", "300001.SZ": "CHINEXT",
        "688001.SH": "STAR", "920001.BJ": "BSE",
    }
    assert result.unique("asset_id").height == result.height


def test_board_sync_writes_pit_table_and_quality_report(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.replace("security_master", security_fixture())
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 13, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    manifest = sync_board_membership(lake, date(2026, 8, 13))
    report = board_quality(lake, date(2026, 8, 13))
    assert manifest["status"] == "passed"
    assert report["coverage"] == 1.0
    assert report["counts"] == {"MAIN": 1, "CHINEXT": 1, "STAR": 1, "BSE": 1}


def test_full_sample_board_benchmarks_use_prior_market_cap(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    first, second = date(2026, 8, 12), date(2026, 8, 13)
    assets = ["600001.SH", "300001.SZ", "688001.SH", "920001.BJ"]
    boards = ["MAIN", "CHINEXT", "STAR", "BSE"]
    lake.replace(
        "board_membership_history",
        pl.DataFrame({
            "asset_id": assets, "board_id": boards,
            "effective_from": [date(2020, 1, 1)] * 4,
            "effective_to": [None] * 4,
        }, schema_overrides={"effective_to": pl.Date}),
    )
    lake.replace(
        "returns_daily",
        pl.DataFrame({
            "trade_date": [first] * 4 + [second] * 4,
            "asset_id": assets * 2,
            "total_return": [0.0] * 4 + [0.01, 0.02, 0.03, 0.04],
        }),
    )
    lake.replace(
        "valuation_daily",
        pl.DataFrame({
            "trade_date": [first] * 4 + [second] * 4,
            "asset_id": assets * 2,
            "float_mkt_cap": [100.0] * 8,
        }),
    )
    lake.replace(
        "security_daily_state",
        pl.DataFrame({
            "trade_date": [first] * 4 + [second] * 4,
            "asset_id": assets * 2,
            "board_id": boards * 2,
            "in_a_share_scope": [True] * 8,
        }),
    )
    base_dir = lake.metadata / "base_manifests"
    base_dir.mkdir(parents=True)
    (base_dir / "legacy_base_v1.json").write_text(json.dumps({
        "snapshot_id": "base", "frozen_at": "2026-08-13T00:00:00+00:00",
        "artifacts": {
            table: {"path": f"silver/{table}/data.parquet"}
            for table in ("returns_daily", "valuation_daily", "security_daily_state")
        },
    }))
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 13, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    summary = build_board_benchmarks(lake, first, second)
    assert summary["quality_gate"]["status"] == "passed"
    returns = {row["board_id"]: row["board_return"] for row in summary["boards"]}
    assert returns == {"MAIN": 0.01, "CHINEXT": 0.02, "STAR": 0.03, "BSE": 0.04}
