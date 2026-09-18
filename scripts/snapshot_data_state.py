#!/usr/bin/env python3
"""Freeze the current persisted data lake state without copying its payloads.

Silver files are content-addressed by SHA-256. Bronze objects are immutable and
already carry a content hash in their filename, so their complete relative-path
inventory is hashed as a set. The resulting manifest is immutable; `_CURRENT`
is only a pointer to the latest successfully written manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from factor_matrix.storage import DataLake, file_sha256, json_hash


DATE_COLUMNS = (
    "trade_date",
    "cal_date",
    "effective_from",
    "report_period",
    "list_date",
)


def silver_record(data_root: Path, path: Path) -> dict[str, Any]:
    schema = pl.scan_parquet(path).collect_schema()
    expressions = [pl.len().alias("row_count")]
    selected_date = next((column for column in DATE_COLUMNS if column in schema), None)
    if selected_date:
        expressions.extend(
            [
                pl.col(selected_date).min().alias("minimum_date"),
                pl.col(selected_date).max().alias("maximum_date"),
            ]
        )
    stats = pl.scan_parquet(path).select(expressions).collect().row(0, named=True)
    return {
        "path": str(path.relative_to(data_root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "row_count": stats["row_count"],
        "date_column": selected_date,
        "minimum_date": str(stats.get("minimum_date")) if stats.get("minimum_date") else None,
        "maximum_date": str(stats.get("maximum_date")) if stats.get("maximum_date") else None,
    }


def bronze_inventory(data_root: Path, paths: list[Path]) -> dict[str, Any]:
    relative = [str(path.relative_to(data_root)) for path in paths]
    endpoints = Counter(path.relative_to(data_root).parts[2] for path in paths)
    entries = [
        {"path": name, "bytes": path.stat().st_size}
        for name, path in zip(relative, paths, strict=True)
    ]
    return {
        "object_count": len(paths),
        "bytes": sum(entry["bytes"] for entry in entries),
        "endpoint_counts": dict(sorted(endpoints.items())),
        "inventory_sha256": json_hash(entries),
        "integrity_note": "Bronze content hashes are embedded in immutable object filenames.",
    }


def optional_artifact(data_root: Path, relative: str) -> dict[str, Any] | None:
    path = data_root / relative
    if not path.exists():
        return None
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    lake = DataLake(data_root)

    with lake.pipeline_lock():
        catalog = lake.refresh_catalog()
        silver = {
            path.parent.name: silver_record(data_root, path)
            for path in sorted(lake.silver.glob("*/data.parquet"))
        }
        bronze_paths = sorted(lake.bronze.glob("*/ingest_date=*/*.json.gz"))
        manifests = sorted(lake.manifests.glob("*.json"))
        evidence = {
            key: value
            for key, value in {
                "research_window_audit": optional_artifact(
                    data_root, "metadata/research_window_audit.json"
                ),
                "full_history_quality": optional_artifact(
                    data_root, "metadata/quality_reports/full_history.json"
                ),
                "catalog": {
                    "path": str(catalog.relative_to(data_root)),
                    "bytes": catalog.stat().st_size,
                    "sha256": file_sha256(catalog),
                },
            }.items()
            if value is not None
        }
        identity = {
            "schema_version": 1,
            "silver": silver,
            "bronze": bronze_inventory(data_root, bronze_paths),
            "pipeline_manifests": {
                "count": len(manifests),
                "inventory_sha256": json_hash(
                    [
                        {
                            "path": str(path.relative_to(data_root)),
                            "sha256": file_sha256(path),
                        }
                        for path in manifests
                    ]
                ),
            },
            "evidence": evidence,
        }
        snapshot_hash = json_hash(identity)
        snapshot_id = f"data_snapshot_{snapshot_hash[:16]}"
        payload = {
            **identity,
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "frozen",
        }
        snapshot_dir = lake.metadata / "snapshots"
        manifest_path = snapshot_dir / f"{snapshot_id}.json"
        lake.write_immutable_json(manifest_path, payload)

        pointer = snapshot_dir / "_CURRENT"
        pointer_tmp = snapshot_dir / "_CURRENT.tmp"
        pointer_tmp.write_text(manifest_path.name + "\n", encoding="utf-8")
        os.replace(pointer_tmp, pointer)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
