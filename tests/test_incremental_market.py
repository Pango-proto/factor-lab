import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from factor_matrix.incremental_market import IncrementalMarketPipeline
from factor_matrix.market_sources import PRICE_LIMIT_FIELDS, SECURITY_FIELDS
from factor_matrix.revisioned_silver import SilverAsOfReader, SilverVersionLedger
from factor_matrix.source import TushareResponse
from factor_matrix.storage import DataLake, open_duckdb


def response(api, params, fields, rows):
    return TushareResponse(
        api, params, fields, rows,
        {"code": 0, "data": {"fields": fields, "items": rows}},
    )


class Client:
    def __init__(self, close=12.0):
        self.calls = []
        self.close = close

    def query(self, api, params, fields=None):
        self.calls.append((api, params))
        day = "20260817"
        if api == "trade_cal":
            names = ["exchange", "cal_date", "is_open", "pretrade_date"]
            return response(api, params, names, [["SSE", day, 1, "20260814"]])
        if api == "stock_basic":
            rows = [[
                "000001.SZ", "000001", "平安银行", "深圳", "银行", "主板", "SZSE",
                "CNY", "L", "19910403", None,
            ]] if params["list_status"] == "L" else []
            return response(api, params, SECURITY_FIELDS, rows)
        if api == "daily":
            names = [
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                "change", "pct_chg", "vol", "amount",
            ]
            return response(api, params, names, [[
                "000001.SZ", day, 11.0, max(12.5, self.close), 10.8, self.close, 10.0,
                self.close - 10.0, (self.close / 10.0 - 1) * 100, 100.0, 1200.0,
            ]])
        if api == "adj_factor":
            names = ["ts_code", "trade_date", "adj_factor"]
            return response(api, params, names, [["000001.SZ", day, 1.0]])
        if api == "daily_basic":
            names = [
                "ts_code", "trade_date", "close", "turnover_rate", "turnover_rate_f",
                "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
                "total_share", "float_share", "free_share", "total_mv", "circ_mv",
            ]
            return response(api, params, names, [[
                "000001.SZ", day, self.close, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 100.0, 80.0, 70.0, 1200.0, 960.0,
            ]])
        if api == "stk_limit":
            return response(api, params, PRICE_LIMIT_FIELDS, [[
                "000001.SZ", day, 10.0, 13.2, 10.8,
            ]])
        if api == "suspend_d":
            names = ["ts_code", "trade_date", "suspend_timing", "suspend_type"]
            return response(api, params, names, [])
        if api == "stock_st":
            names = ["ts_code", "trade_date", "name", "type", "type_name"]
            return response(api, params, names, [])
        raise AssertionError(api)


def initialize_frozen_lake(tmp_path: Path) -> DataLake:
    lake = DataLake(tmp_path)
    prices_path = lake.silver / "prices_daily" / "data.parquet"
    prices_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2026, 8, 14)], "asset_id": ["000001.SZ"],
        "gross_close_index": [10.0], "raw_close": [10.0], "adj_factor": [1.0],
    }).write_parquet(prices_path)
    calendar_path = lake.silver / "trade_calendar" / "data.parquet"
    calendar_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "exchange": ["SSE", "SSE"],
        "cal_date": [date(2026, 8, 14), date(2026, 8, 17)],
        "is_open": [1, 1], "pretrade_date": [date(2026, 8, 13), date(2026, 8, 14)],
    }).write_parquet(calendar_path)
    manifest_dir = lake.metadata / "base_manifests"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "base_id": "legacy_base_v1", "snapshot_id": "base",
        "snapshot_sha256": "base-sha", "frozen_at": "2026-08-15T00:00:00+00:00",
        "artifacts": {
            "prices_daily": {"path": "silver/prices_daily/data.parquet"},
            "trade_calendar": {"path": "silver/trade_calendar/data.parquet"},
        },
    }
    (manifest_dir / "legacy_base_v1.json").write_text(json.dumps(manifest))
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base-sha",
        observed_at=datetime(2026, 8, 15, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    return lake


def test_incremental_market_appends_without_mutating_base(tmp_path):
    lake = initialize_frozen_lake(tmp_path)
    base_path = lake.silver / "prices_daily" / "data.parquet"
    before = hashlib.sha256(base_path.read_bytes()).hexdigest()
    client = Client()
    manifest = IncrementalMarketPipeline(client, lake, "fp").sync(date(2026, 8, 17))
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "passed"
    assert payload["quality_gate"]["status"] == "passed"
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == before
    assert {item["table"] for item in payload["changes"]} >= {
        "prices_daily", "valuation_daily", "returns_daily", "security_daily_state",
    }
    reader = SilverAsOfReader(lake)
    with open_duckdb() as connection:
        returns_sql = reader.relation_sql("returns_daily", datetime.now(UTC))
        value = connection.execute(
            f"SELECT total_return FROM ({returns_sql}) WHERE trade_date=DATE '2026-08-17'"
        ).fetchone()[0]
        assert abs(value - 0.2) < 1e-12
        state_sql = reader.relation_sql("security_daily_state", datetime.now(UTC))
        assert connection.execute(
            f"SELECT observation_state FROM ({state_sql}) WHERE asset_id='000001.SZ'"
        ).fetchone()[0] == "TRADED"
    catalog = lake.refresh_catalog()
    with open_duckdb(catalog, read_only=True) as connection:
        assert connection.execute(
            "SELECT raw_close FROM prices_daily WHERE trade_date=DATE '2026-08-17'"
        ).fetchone()[0] == 12.0


def test_incremental_market_is_idempotent_after_publish(tmp_path):
    lake = initialize_frozen_lake(tmp_path)
    first_client = Client()
    pipeline = IncrementalMarketPipeline(first_client, lake, "fp")
    first = pipeline.sync(date(2026, 8, 17))
    second_client = Client()
    second = IncrementalMarketPipeline(second_client, lake, "fp").sync(date(2026, 8, 17))
    assert second == first
    assert second_client.calls == []


def test_force_run_publishes_correction_without_rewriting_delta(tmp_path):
    lake = initialize_frozen_lake(tmp_path)
    IncrementalMarketPipeline(Client(close=12.0), lake, "fp").sync(date(2026, 8, 17))
    corrected = IncrementalMarketPipeline(Client(close=11.0), lake, "fp").sync(
        date(2026, 8, 17), force=True
    )
    payload = json.loads(corrected.read_text())
    assert payload["status"] == "passed"
    assert {item["layer"] for item in payload["changes"]} == {"corrections"}
    sql = SilverAsOfReader(lake).relation_sql("prices_daily", datetime.now(UTC))
    with open_duckdb() as connection:
        assert connection.execute(
            f"SELECT raw_close FROM ({sql}) WHERE trade_date=DATE '2026-08-17'"
        ).fetchone()[0] == 11.0
