import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from factor_matrix.concepts import (
    ConceptChangeScanner, ConceptSnapshotPipeline, MASTER_FIELDS, MEMBER_FIELDS,
    concept_scan_passed, concept_snapshot_passed,
)
from factor_matrix.revisioned_silver import SilverVersionLedger
from factor_matrix.source import TushareError, TushareResponse
from factor_matrix.storage import DataLake


def response(api_name, params, fields, rows):
    return TushareResponse(
        api_name, params, fields, rows,
        {"code": 0, "data": {"fields": fields, "items": rows}},
    )


class Client:
    def __init__(self, *, fail_members=False, declared_count=1, members=None):
        self.fail_members = fail_members
        self.declared_count = declared_count
        self.members = members or ["000001.SZ"]
        self.calls = []

    def query(self, api_name, params, fields):
        self.calls.append((api_name, params))
        if api_name == "ths_index":
            rows = [["885001.TI", "测试概念", self.declared_count, "A", "20200101", "N"]]
            return response(api_name, params, MASTER_FIELDS, rows)
        if self.fail_members:
            raise TushareError("permission denied")
        rows = [
            ["885001.TI", asset_id, asset_id, None, "20200101", None, "Y"]
            for asset_id in self.members
        ]
        return response(api_name, params, MEMBER_FIELDS, rows)


def initialize_version(lake: DataLake):
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        changes=[], quality_gate={"status": "passed"},
    )


def test_concept_snapshot_is_atomic_append_only_and_dated(tmp_path: Path):
    lake = DataLake(tmp_path)
    manifest = ConceptSnapshotPipeline(Client(), lake, "fp").sync(
        date(2026, 8, 14), request_delay=0, workers=1
    )
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "passed"
    assert concept_snapshot_passed(lake, date(2026, 8, 14))
    output = tmp_path / payload["output"]["path"]
    stored = pl.read_parquet(output)
    assert stored.select("snapshot_date", "concept_id", "asset_id").row(0) == (
        date(2026, 8, 14), "885001.TI", "000001.SZ"
    )
    assert stored.get_column("available_at").null_count() == 0
    assert stored.get_column("first_seen_at").null_count() == 0


def test_member_failure_is_recorded_without_silver_carry_forward(tmp_path: Path):
    lake = DataLake(tmp_path)
    manifest = ConceptSnapshotPipeline(Client(fail_members=True), lake, "fp").sync(
        date(2026, 8, 14), request_delay=0, workers=1
    )
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "source_incomplete"
    assert payload["quality_gate"]["reason"] == "SOURCE_INCOMPLETE"
    assert not (tmp_path / "silver" / "delta").exists()


def test_master_member_count_mismatch_blocks_publication(tmp_path: Path):
    lake = DataLake(tmp_path)
    manifest = ConceptSnapshotPipeline(Client(declared_count=2), lake, "fp").sync(
        date(2026, 8, 14), request_delay=0, workers=1
    )
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "source_incomplete"
    assert payload["quality_gate"]["count_shortfalls"] == [
        {"concept_id": "885001.TI", "expected": 2, "observed": 1}
    ]
    assert not (tmp_path / "silver" / "delta").exists()


def test_daily_concept_scan_uses_only_master_when_unchanged(tmp_path: Path):
    lake = DataLake(tmp_path)
    initialize_version(lake)
    ConceptSnapshotPipeline(Client(), lake, "fp").sync(
        date(2026, 8, 14), request_delay=0, workers=1
    )
    client = Client()
    manifest = ConceptChangeScanner(client, lake, "fp").sync(date(2026, 8, 15))
    payload = json.loads(manifest.read_text())
    assert payload["status"] == "passed"
    assert payload["candidate_concepts"] == []
    assert payload["member_requests"] == 0
    assert payload["event_rows"] == 0
    assert [call[0] for call in client.calls] == ["ths_index"]
    assert concept_scan_passed(lake, date(2026, 8, 15))


def test_daily_concept_scan_emits_delta_once_and_does_not_repeat(tmp_path: Path):
    lake = DataLake(tmp_path)
    initialize_version(lake)
    ConceptSnapshotPipeline(Client(), lake, "fp").sync(
        date(2026, 8, 14), request_delay=0, workers=1
    )
    changed = Client(declared_count=2, members=["000001.SZ", "000002.SZ"])
    first_manifest = ConceptChangeScanner(changed, lake, "fp").sync(date(2026, 8, 15))
    first = json.loads(first_manifest.read_text())
    assert first["event_rows"] == 1
    event = pl.read_parquet(tmp_path / first["changes"][0]["path"]).row(0, named=True)
    assert (event["asset_id"], event["change_type"]) == ("000002.SZ", "ADD")

    repeated = Client(declared_count=2, members=["000001.SZ", "000002.SZ"])
    second_manifest = ConceptChangeScanner(repeated, lake, "fp").sync(date(2026, 8, 16))
    second = json.loads(second_manifest.read_text())
    assert second["candidate_concepts"] == []
    assert second["member_requests"] == 0
    assert second["event_rows"] == 0
    assert [call[0] for call in repeated.calls] == ["ths_index"]
