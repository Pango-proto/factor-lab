import json

import polars as pl

from factor_matrix.storage import DataLake


def test_legacy_writer_cannot_mutate_frozen_base(tmp_path):
    lake = DataLake(tmp_path)
    manifest = tmp_path / "metadata" / "base_manifests" / "legacy_base_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"artifacts": {"prices_daily": {}}}))
    try:
        lake.upsert(
            "prices_daily", pl.DataFrame({"id": [1]}), primary_key=["id"]
        )
    except RuntimeError as exc:
        assert "FROZEN_BASE_MUTATION_FORBIDDEN" in str(exc)
    else:
        raise AssertionError("frozen base must reject legacy upsert")


def test_new_nonbase_table_can_still_be_created(tmp_path):
    lake = DataLake(tmp_path)
    manifest = tmp_path / "metadata" / "base_manifests" / "legacy_base_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"artifacts": {"prices_daily": {}}}))
    output = lake.upsert("new_table", pl.DataFrame({"id": [1]}), primary_key=["id"])
    assert output.exists()
