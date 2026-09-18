"""PIT industry observation intervals derived without guessing change dates."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .revisioned_silver import SilverVersionLedger, read_as_of_frame
from .storage import DataLake, utc_now


OBSERVED = "OBSERVED"
GAP_SAME_L1_FILLED = "MEMBERSHIP_GAP_SAME_L1_FILLED"
MEMBERSHIP_GAP = "MEMBERSHIP_GAP"
CLASSIFICATION_NOT_YET_EFFECTIVE = "CLASSIFICATION_NOT_YET_EFFECTIVE"


def _interval_rows(history: pl.DataFrame, standard: str) -> pl.DataFrame:
    source = history.filter(pl.col("classification_standard") == standard).sort(
        ["asset_id", "in_date"]
    )
    if source.is_empty():
        raise RuntimeError("INDUSTRY_OBSERVATION_SOURCE_EMPTY")
    rows: list[dict[str, Any]] = []
    for key, group in source.partition_by("asset_id", as_dict=True).items():
        asset_id = key[0] if isinstance(key, tuple) else key
        records = group.sort("in_date").iter_rows(named=True)
        ordered = list(records)
        for index, record in enumerate(ordered):
            rows.append({
                "classification_standard": standard,
                "asset_id": asset_id,
                "effective_from": record["in_date"],
                "effective_to": record["out_date"],
                "l1_code": record["l1_code"], "l1_name": record["l1_name"],
                "l2_code": record["l2_code"], "l2_name": record["l2_name"],
                "industry_observation_state": OBSERVED,
                "filled_from_adjacent": False,
            })
            if index + 1 >= len(ordered) or record["out_date"] is None:
                continue
            following = ordered[index + 1]
            gap_start = record["out_date"] + timedelta(days=1)
            gap_end = following["in_date"] - timedelta(days=1)
            if gap_start > gap_end:
                continue
            same_l1 = record["l1_code"] == following["l1_code"]
            same_l2 = same_l1 and record["l2_code"] == following["l2_code"]
            rows.append({
                "classification_standard": standard,
                "asset_id": asset_id,
                "effective_from": gap_start,
                "effective_to": gap_end,
                "l1_code": record["l1_code"] if same_l1 else None,
                "l1_name": record["l1_name"] if same_l1 else None,
                "l2_code": record["l2_code"] if same_l2 else None,
                "l2_name": record["l2_name"] if same_l2 else None,
                "industry_observation_state": (
                    GAP_SAME_L1_FILLED if same_l1 else MEMBERSHIP_GAP
                ),
                "filled_from_adjacent": same_l1,
            })
    result = pl.DataFrame(rows).sort(["asset_id", "effective_from"])
    duplicates = result.group_by(
        ["classification_standard", "asset_id", "effective_from"]
    ).len().filter(pl.col("len") != 1)
    if not duplicates.is_empty():
        raise RuntimeError("INDUSTRY_OBSERVATION_INTERVAL_DUPLICATE")
    return result


def build_industry_observation_intervals(
    lake: DataLake, *, standard: str = "SW2021",
) -> dict[str, Any]:
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("INDUSTRY_OBSERVATION_REQUIRES_SILVER_VERSION")
    normalized_standard = standard.upper()
    config = {
        "schema_version": 1,
        "classification_standard": normalized_standard,
        "same_l1_gap_policy": "fill_l1_only_from_matching_adjacent_intervals",
        "different_l1_gap_policy": "retain_null_and_mark_membership_gap",
        "l2_gap_policy": "fill_only_when_both_adjacent_l2_codes_match",
    }
    as_of = utc_now()
    run_id, config_hash, code_hash = lake.calculation_run_id(
        "industry_observation_intervals_v1", as_of.date(), config,
        [version["version_id"]],
    )
    run_dir = lake.root / "gold" / "industry_observation_intervals" / f"run_id={run_id}"
    artifact_path = run_dir / "industry_observation_intervals_v1.parquet"
    manifest_path = run_dir / "_MANIFEST.json"
    if artifact_path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    intervals = _interval_rows(
        read_as_of_frame(lake, "industry_membership_history", as_of),
        normalized_standard,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    temporary = artifact_path.with_suffix(".parquet.tmp")
    intervals.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, artifact_path)
    counts = {
        "rows": intervals.height,
        "assets": intervals["asset_id"].n_unique(),
        "observed_intervals": intervals.filter(
            pl.col("industry_observation_state") == OBSERVED
        ).height,
        "same_l1_filled_intervals": intervals.filter(
            pl.col("industry_observation_state") == GAP_SAME_L1_FILLED
        ).height,
        "different_l1_gap_intervals": intervals.filter(
            pl.col("industry_observation_state") == MEMBERSHIP_GAP
        ).height,
    }
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "industry_observation_intervals_v1", "status": "passed",
        "created_at": as_of.isoformat(), "silver_version_id": version["version_id"],
        "config": config, "config_hash": config_hash, "code_hash": code_hash,
        "counts": counts,
        "source_contract": "Tushare index_member_all Bronze equals Silver primary-key set",
        "artifact": lake.artifact_record(artifact_path),
    }
    lake.write_immutable_json(manifest_path, manifest)
    current_path = lake.root / "gold" / "industry_observation_intervals" / "_CURRENT.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_current = current_path.with_suffix(".json.tmp")
    temporary_current.write_text(json.dumps({
        "status": "active", "run_id": run_id,
        "manifest": str(manifest_path.relative_to(lake.root)),
        "artifact": str(artifact_path.relative_to(lake.root)),
        "silver_version_id": version["version_id"], "updated_at": as_of.isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_current, current_path)
    return manifest
