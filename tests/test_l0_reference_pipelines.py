import json
from datetime import UTC, date, datetime

import polars as pl

from factor_matrix.board_benchmarks import INDEX_DAILY_FIELDS, INDEX_NAMES, IndexDailyPipeline
from factor_matrix.revisioned_silver import SilverVersionLedger
from factor_matrix.source import TushareResponse
from factor_matrix.storage import DataLake


class IndexClient:
    def __init__(self):
        self.calls = 0

    def query(self, api_name, params, fields):
        self.calls += 1
        assert api_name == "index_daily"
        row = [
            params["ts_code"], "20260817", 100.0, 99.0, 101.0, 98.0,
            99.0, 1.0, 1.01, 1000.0, 2000.0,
        ]
        raw = {"code": 0, "data": {"fields": INDEX_DAILY_FIELDS, "items": [row]}}
        return TushareResponse(api_name, params, INDEX_DAILY_FIELDS, [row], raw)


def test_index_daily_is_revisioned_and_idempotent(tmp_path):
    lake = DataLake(tmp_path / "data")
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 17, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    client = IndexClient()
    pipeline = IndexDailyPipeline(client, lake, "fp")
    manifest = pipeline.sync(date(2026, 8, 17), date(2026, 8, 17))
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "passed"
    assert payload["silver_version_id"]
    assert payload["changes"][0]["table"] == "index_prices_daily"
    frame = pl.read_parquet(lake.root / payload["changes"][0]["path"])
    assert frame.height == len(INDEX_NAMES)
    assert client.calls == len(INDEX_NAMES)

    same = pipeline.sync(date(2026, 8, 17), date(2026, 8, 17))
    assert same == manifest
    assert client.calls == len(INDEX_NAMES)
