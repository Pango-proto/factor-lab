"""Month-partitioned G2b build and immutable consolidation."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ...research_protocol import ResearchProtocol
from ...storage import DataLake, file_sha256, json_hash, open_duckdb, utc_now
from .config import assert_exposure_publish_allowed
from .runner import l1_calculation_code_hash, run_l1_risk_exposure


def _month_ranges(dates: list[date]) -> list[tuple[date, date]]:
    groups: dict[tuple[int, int], list[date]] = {}
    for value in dates:
        groups.setdefault((value.year, value.month), []).append(value)
    return [(min(values), max(values)) for _, values in sorted(groups.items())]


def _copy_union(paths: list[Path], destination: Path, lake: DataLake) -> None:
    escaped_paths = ",".join(
        "'" + str(path.resolve()).replace("'", "''") + "'" for path in paths
    )
    temporary = destination.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"COPY (SELECT * FROM read_parquet([{escaped_paths}], union_by_name=true)) "
            f"TO '{str(temporary).replace("'", "''")}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
    os.replace(temporary, destination)


def run_l1_risk_exposure_history(
    lake: DataLake,
    *,
    start: date,
    end: date,
    universe_metadata_path: Path,
    board_benchmark_summary_path: Path,
    listing_policy_path: Path = Path("config/new_listing_policy_v1.json"),
    risk_candidates_path: Path = Path("config/risk_factor_set_candidate_v1.json"),
    protocol_path: Path = Path("config/research_protocol_v1.json"),
) -> Path:
    assert_exposure_publish_allowed(
        policy_path=listing_policy_path,
        universe_variant="frozen_d0",
        update_current_pointer=False,
    )
    universe_metadata = json.loads(universe_metadata_path.read_text(encoding="utf-8"))
    universe_path = lake.root / universe_metadata["artifact"]
    dates = (
        pl.scan_parquet(universe_path)
        .filter(pl.col("trade_date").is_between(start, end))
        .select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list()
    )
    if not dates:
        raise RuntimeError("L1_HISTORY_REQUESTED_DATES_EMPTY")
    latest_size = (
        pl.scan_parquet(universe_path)
        .filter(pl.col("trade_date") == dates[-1]).select(pl.len()).collect().item()
    )
    candidates = json.loads(risk_candidates_path.read_text(encoding="utf-8"))
    protocol = ResearchProtocol.load(protocol_path)
    snapshot = protocol.derive(
        factor_count=len(candidates["members"]), maximum_evaluation_horizon_days=1,
        cross_section_size=latest_size, available_estimation_days=len(dates),
    )
    derived_params = dict(snapshot["values"])
    derived_params["new_listing_days_by_regime"] = json.loads(
        listing_policy_path.read_text(encoding="utf-8")
    )["d_star_reference"]["values"]
    derived_params["unresolved_required"] = []

    partition_manifests: list[Path] = []
    partition_rows: list[dict[str, Any]] = []
    for partition_start, partition_end in _month_ranges(dates):
        manifest_path = run_l1_risk_exposure(
            lake,
            start=partition_start,
            end=partition_end,
            universe_metadata_path=universe_metadata_path,
            board_benchmark_summary_path=board_benchmark_summary_path,
            universe_variant="frozen_d0",
            publish_current=False,
            listing_policy_path=listing_policy_path,
            risk_candidates_path=risk_candidates_path,
            protocol_path=protocol_path,
            _derived_params_override=derived_params,
            _allow_latest_invalid=True,
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        partition_manifests.append(manifest_path)
        partition_rows.append({
            "partition_start": partition_start,
            "partition_end": partition_end,
            "run_id": payload["run_id"],
            "rows": payload["counts"]["rows"],
            "invalid_dates": payload["counts"]["invalid_dates"],
            "manifest_path": str(manifest_path.relative_to(lake.root)),
        })

    partition_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in partition_manifests]
    identity = {
        "start": start.isoformat(), "end": end.isoformat(),
        "universe_variant": "frozen_d0",
        "partition_run_ids": [payload["run_id"] for payload in partition_payloads],
        "derived_params": derived_params,
        "code_sha": l1_calculation_code_hash(),
    }
    run_id = f"l1_risk_exposure_history_{end:%Y%m%d}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "gold" / "risk_exposure_matrix" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    exposure_path = run_dir / "risk_exposure_matrix_v1.parquet"
    quality_path = run_dir / "exposure_quality_v1.parquet"
    partition_index_path = run_dir / "history_partition_index_v1.parquet"
    _copy_union([
        lake.root / payload["outputs"]["risk_exposure_matrix_v1"]["path"]
        for payload in partition_payloads
    ], exposure_path, lake)
    _copy_union([
        lake.root / payload["outputs"]["exposure_quality_v1"]["path"]
        for payload in partition_payloads
    ], quality_path, lake)
    pl.DataFrame(partition_rows).write_parquet(partition_index_path)

    validity = pl.scan_parquet(exposure_path).group_by("trade_date").agg(
        pl.col("is_valid").all().alias("is_valid")
    ).collect().sort("trade_date")
    invalid_dates = validity.filter(~pl.col("is_valid"))["trade_date"].to_list()
    invalid_ratio = len(invalid_dates) / len(dates)
    burn_in_days = int(derived_params["burn_in_days"])
    estimation_eligible_dates = dates[burn_in_days:]
    estimation_eligible_set = set(estimation_eligible_dates)
    post_burn_invalid_dates = [
        value for value in invalid_dates if value in estimation_eligible_set
    ]
    post_burn_invalid_ratio = (
        len(post_burn_invalid_dates) / len(estimation_eligible_dates)
        if estimation_eligible_dates else 0.0
    )
    if dates[-1] in invalid_dates:
        raise RuntimeError("L1_HISTORY_LATEST_DATE_INVALID")
    if post_burn_invalid_ratio > float(derived_params["maximum_invalid_date_ratio"]):
        raise RuntimeError("L1_HISTORY_INVALID_DATE_RATIO_FAILED")
    risk_basis_id = f"risk_basis_{json_hash({
        'risk_factor_set_id': partition_payloads[0]['risk_factor_set_id'],
        'risk_metric_id': partition_payloads[0]['risk_metric_id'],
        'exposure_sha256': file_sha256(exposure_path),
    })[:16]}"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "l1_risk_exposure_history",
        "execution_status": "passed_pending_history_diagnostics",
        "publish_mode": "formal_candidate_not_current",
        "start": start.isoformat(), "end": end.isoformat(),
        "created_at": utc_now().isoformat(),
        "universe_variant": "frozen_d0",
        "silver_version_id": partition_payloads[0]["silver_version_id"],
        "tradable_universe_run_id": partition_payloads[0]["tradable_universe_run_id"],
        "board_benchmark_run_id": partition_payloads[0]["board_benchmark_run_id"],
        "registry_snapshot_id": partition_payloads[0]["registry_snapshot_id"],
        "risk_factor_set_id": partition_payloads[0]["risk_factor_set_id"],
        "risk_factor_set_status": "candidate",
        "risk_metric_id": partition_payloads[0]["risk_metric_id"],
        "weight_metric_contract": partition_payloads[0]["weight_metric_contract"],
        "risk_basis_id": risk_basis_id,
        "derived_params": derived_params,
        "code_sha": identity["code_sha"],
        "partition_run_ids": identity["partition_run_ids"],
        "counts": {
            "dates": len(dates),
            "rows": sum(row["rows"] for row in partition_rows),
            "partitions": len(partition_rows),
            "invalid_dates": len(invalid_dates),
            "invalid_date_ratio": invalid_ratio,
            "warmup_dates": min(burn_in_days, len(dates)),
            "post_burn_dates": len(estimation_eligible_dates),
            "post_burn_invalid_dates": len(post_burn_invalid_dates),
            "post_burn_invalid_date_ratio": post_burn_invalid_ratio,
        },
        "invalid_dates": [value.isoformat() for value in invalid_dates],
        "post_burn_invalid_dates": [
            value.isoformat() for value in post_burn_invalid_dates
        ],
        "outputs": {
            "risk_exposure_matrix_v1": lake.artifact_record(exposure_path),
            "exposure_quality_v1": lake.artifact_record(quality_path),
            "history_partition_index_v1": lake.artifact_record(partition_index_path),
        },
        "promotion_gate": {
            "status": "blocked_pending_history_diagnostics",
            "required": [
                "exposure_autocorrelation_v1",
                "board_industry_cross_v1",
                "boundary_diff_v1",
            ],
        },
    }
    return lake.write_immutable_json(manifest_path, manifest)
