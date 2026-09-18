from pathlib import Path

import pytest

import json
from datetime import date

import polars as pl

from factor_matrix.storage import DataLake, file_sha256, json_hash


def test_pipeline_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "data")
    with lake.pipeline_lock():
        with pytest.raises(RuntimeError, match="Another factor-matrix writer"):
            with lake.pipeline_lock():
                pass


def test_data_snapshot_hashes_silver_files_and_is_immutable(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "data")
    path = lake.replace("prices_daily", pl.DataFrame({"asset_id": ["A"], "value": [1.0]}))
    first = lake.register_data_snapshot(date(2026, 8, 6), ["prices_daily"])
    second = lake.register_data_snapshot(date(2026, 8, 6), ["prices_daily"])
    assert first == second
    assert first["artifacts"]["prices_daily"]["sha256"] == file_sha256(path)
    assert first["snapshot_hash"] == json_hash(
        {"as_of": "2026-08-06", "artifacts": first["artifacts"]}
    )
    payload = json.loads((lake.manifests / f"{first['run_id']}.json").read_text())
    assert payload["status"] == "registered"


def test_immutable_registry_refuses_same_id_with_changed_content(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "data")
    path = lake.manifests / "same.json"
    lake.write_immutable_json(path, {"value": 1})
    with pytest.raises(RuntimeError, match="IMMUTABLE_RECORD_CONFLICT"):
        lake.write_immutable_json(path, {"value": 2})


def test_calculation_guard_rejects_base_layer_mutation(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.replace("sample", pl.DataFrame({"id": [1]}))
    with pytest.raises(RuntimeError, match="BASE_DATA_MUTATED"):
        with lake.immutable_base_guard():
            lake.replace("sample", pl.DataFrame({"id": [2]}))


def test_upsert_streams_existing_parquet_and_keeps_latest_key(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.upsert("sample", pl.DataFrame({"id": [1, 2], "value": [10, 20]}), primary_key=["id"])
    path = lake.upsert("sample", pl.DataFrame({"id": [2, 3], "value": [99, 30]}), primary_key=["id"])
    result = pl.read_parquet(path).sort("id")
    assert result.to_dicts() == [
        {"id": 1, "value": 10}, {"id": 2, "value": 99}, {"id": 3, "value": 30}
    ]
