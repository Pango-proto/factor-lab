"""Month-partitioned formal G3 risk-only build and immutable consolidation."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, utc_now
from .g3_runner import (
    _load_l1_lineage, _next_return_map, g3_calculation_code_hash,
    run_g3_risk_only,
)
from .g3_gate import G3_GATE_CONFIG_PATH, G3_GATE_VERSION


PRODUCTS = (
    "factor_returns_v1", "specific_returns_v1",
    "factor_regression_quality_v1", "cross_section_stats_v1",
)


def _month_ranges(dates: list[date]) -> list[tuple[date, date]]:
    groups: dict[tuple[int, int], list[date]] = {}
    for value in dates:
        groups.setdefault((value.year, value.month), []).append(value)
    return [(min(values), max(values)) for _, values in sorted(groups.items())]


def _copy_union(paths: list[Path], destination: Path, lake: DataLake) -> None:
    escaped = ",".join("'" + str(path.resolve()).replace("'", "''") + "'" for path in paths)
    temporary = destination.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"COPY (SELECT * FROM read_parquet([{escaped}], union_by_name=true)) "
            f"TO '{str(temporary).replace("'", "''")}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
    os.replace(temporary, destination)


def run_g3_risk_only_history(
    lake: DataLake, *, start: date, end: date, publish_current: bool = False,
    l2_config_path: Path = Path("config/l2_return_decomposition_v1.json"),
    l1_config_path: Path = Path("config/l1_risk_exposure_v1.json"),
    protocol_path: Path = Path("config/research_protocol_v1.json"),
    _base_weight_artifact_path: Path | None = None,
    _base_weight_scheme_id: str = "sqrt_cap",
) -> Path:
    wall_start = perf_counter()
    history, diagnostics, history_path, diagnostics_path = _load_l1_lineage(lake)
    metric_contract = history.get("weight_metric_contract")
    if not metric_contract or not history.get("risk_metric_id"):
        raise RuntimeError("G3_HISTORY_L1_WEIGHT_METRIC_CONTRACT_MISSING")
    expected_weight_path = lake.root / metric_contract["artifact_path"]
    if file_sha256(expected_weight_path) != metric_contract["artifact_sha256"]:
        raise RuntimeError("G3_HISTORY_WEIGHT_METRIC_ARTIFACT_HASH_MISMATCH")
    if _base_weight_artifact_path is None:
        _base_weight_artifact_path = expected_weight_path
    elif file_sha256(_base_weight_artifact_path) != metric_contract["artifact_sha256"]:
        raise RuntimeError("G3_HISTORY_WEIGHT_METRIC_ARTIFACT_LINEAGE_MISMATCH")
    if _base_weight_scheme_id == "sqrt_cap":
        _base_weight_scheme_id = metric_contract["regression_base_scheme_id"]
    if _base_weight_scheme_id != metric_contract["regression_base_scheme_id"]:
        raise RuntimeError("G3_HISTORY_WEIGHT_METRIC_SCHEME_MISMATCH")
    exposure_path = lake.root / history["outputs"]["risk_exposure_matrix_v1"]["path"]
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    prices_path = lake.silver / "prices_daily" / "data.parquet"
    universe_candidates = list((lake.root / "gold" / "tradable_universe").glob(
        f"artifact_version=1/run_id={history['tradable_universe_run_id']}/tradable_universe.parquet"
    ))
    if len(universe_candidates) != 1:
        raise RuntimeError("G3_TRADABLE_UNIVERSE_ARTIFACT_NOT_UNIQUE")
    universe_path = universe_candidates[0]
    exposure_dates = (
        pl.scan_parquet(exposure_path).filter(pl.col("is_valid"))
        .select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list()
    )
    return_dates = (
        pl.scan_parquet(returns_path).select("trade_date").unique().sort("trade_date")
        .collect()["trade_date"].to_list()
    )
    date_map = _next_return_map(exposure_dates, return_dates, start=start, end=end)
    realized_dates = date_map["trade_date"].to_list()
    ranges = _month_ranges(realized_dates)
    if not ranges:
        raise RuntimeError("G3_HISTORY_NO_REALIZED_DATES")

    inputs = {
        "l1_history_manifest": history_path,
        "l1_diagnostics_manifest": diagnostics_path,
        "risk_exposure_matrix_v1": exposure_path,
        "tradable_universe_v1": universe_path,
        "returns_daily": returns_path,
        "return_date_prices_daily": prices_path,
        "valuation_daily": valuation_path,
        "l2_config": l2_config_path,
        "l1_config": l1_config_path,
        "research_protocol": protocol_path,
        "weight_metric_contract": Path("config/weight_metric_contract_v1.json"),
    }
    if _base_weight_artifact_path is not None:
        inputs["base_weight_artifact"] = _base_weight_artifact_path
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    derived = dict(history["derived_params"])
    manifest_paths = [run_g3_risk_only(
        lake, start=partition_start, end=partition_end, publish_current=False,
        l2_config_path=l2_config_path, l1_config_path=l1_config_path,
        protocol_path=protocol_path, _derived_params_override=derived,
        _input_hashes_override=input_hashes,
        _base_weight_artifact_path=_base_weight_artifact_path,
        _base_weight_scheme_id=_base_weight_scheme_id,
    ) for partition_start, partition_end in ranges]
    partition_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths
    ]
    partition_profiling = [
        payload.get("profiling", {}) for payload in partition_payloads
    ]
    partition_rows = [{
        "partition_start": partition_start, "partition_end": partition_end,
        "run_id": payload["run_id"], "dates": payload["counts"]["dates"],
        "invalid_dates": payload["counts"]["invalid_dates"],
        "manifest_path": str(path.relative_to(lake.root)),
    } for (partition_start, partition_end), payload, path in zip(
        ranges, partition_payloads, manifest_paths
    )]

    identity = {
        "start": min(realized_dates).isoformat(), "end": max(realized_dates).isoformat(),
        "l1_run_id": history["run_id"],
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "partition_run_ids": [payload["run_id"] for payload in partition_payloads],
        "input_hashes": input_hashes, "derived_params": derived,
        "code_sha": g3_calculation_code_hash(),
        "base_weight_scheme_id": _base_weight_scheme_id,
    }
    run_id = f"g3_risk_only_history_{max(realized_dates):%Y%m%d}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "gold" / "l2b_risk_only" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    write_start = perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    for product in PRODUCTS:
        destination = run_dir / f"{product}.parquet"
        _copy_union([
            lake.root / payload["outputs"][product]["path"] for payload in partition_payloads
        ], destination, lake)
        output_paths[product] = destination
    index_path = run_dir / "history_partition_index_v1.parquet"
    pl.DataFrame(partition_rows).write_parquet(index_path)

    quality = pl.scan_parquet(output_paths["factor_regression_quality_v1"]).collect()
    invalid = quality.filter(pl.col("status") == "invalid")
    duplicate_dates = quality.group_by("trade_date").len().filter(pl.col("len") != 1)
    if not duplicate_dates.is_empty():
        raise RuntimeError("G3_HISTORY_DUPLICATE_OR_MISSING_QUALITY_DATE")
    expected_dates = set(realized_dates)
    actual_dates = set(quality["trade_date"].to_list())
    if actual_dates != expected_dates:
        raise RuntimeError("G3_HISTORY_DATE_COVERAGE_MISMATCH")
    if not invalid.is_empty():
        raise RuntimeError("G3_HISTORY_REQUIRES_ZERO_INVALID_DATES")
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g3_l2b_risk_only_history",
        "status": "passed", "publish_mode": "formal_current" if publish_current else "formal_candidate",
        "start": min(realized_dates).isoformat(), "end": max(realized_dates).isoformat(),
        "created_at": utc_now().isoformat(), "l1_run_id": history["run_id"],
        "l1_diagnostics_run_id": diagnostics["run_id"],
        "silver_version_id": history["silver_version_id"],
        "risk_factor_set_id": history["risk_factor_set_id"],
        "risk_factor_set_status": "candidate", "regression_mode": "risk_only",
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "base_weight_scheme_id": _base_weight_scheme_id,
        "gate_version": G3_GATE_VERSION,
        "gate_config_sha": file_sha256(G3_GATE_CONFIG_PATH),
        "input_hashes": input_hashes,
        "identity": identity, "partition_run_ids": identity["partition_run_ids"],
        "counts": {
            "dates": quality.height, "invalid_dates": 0, "partitions": len(ranges),
            "factor_return_rows": pl.scan_parquet(output_paths["factor_returns_v1"]).select(pl.len()).collect().item(),
            "specific_return_rows": pl.scan_parquet(output_paths["specific_returns_v1"]).select(pl.len()).collect().item(),
        },
        "profiling": {
            "wall_seconds": perf_counter() - wall_start,
            "t_query_seconds": sum(item.get("t_query_seconds", 0.0) for item in partition_profiling),
            "t_pivot_seconds": sum(item.get("t_pivot_seconds", 0.0) for item in partition_profiling),
            "t_solve_seconds": sum(item.get("t_solve_seconds", 0.0) for item in partition_profiling),
            "t_write_seconds": sum(item.get("t_write_seconds", 0.0) for item in partition_profiling)
            + (perf_counter() - write_start),
            "partition_wall_seconds": sum(item.get("wall_seconds", 0.0) for item in partition_profiling),
        },
        "quality_summary": {
            "maximum_condition_number": float(quality["condition_number"].max()),
            "maximum_constraint_error": float(quality["constraint_error"].max()),
            "maximum_regression_identity_error": float(quality["regression_identity_error"].max()),
            "mean_r_squared": float(quality["r_squared"].mean()),
            "mean_full_label_r_squared": float(quality["full_label_r_squared"].mean()),
            "mean_estimation_domain_r_squared": float(
                quality["estimation_domain_r_squared"].mean()
            ),
            "mean_full_label_unweighted_r_squared": float(
                quality["full_label_unweighted_r_squared"].mean()
            ),
            "mean_estimation_domain_unweighted_r_squared": float(
                quality["estimation_domain_unweighted_r_squared"].mean()
            ),
            "mean_huber_downweight_rate": float(quality["huber_downweight_rate"].mean()),
            "maximum_huber_downweight_rate": float(quality["huber_downweight_rate"].max()),
            "label_sample_rows": int(quality["label_sample_count"].sum()),
            "estimation_sample_rows": int(quality["sample_count"].sum()),
            "estimation_excluded_rows": int(quality["estimation_excluded_count"].sum()),
            "r_squared_expected_range_ratio": float(quality["r_squared_in_expected_range"].mean()),
        },
        "outputs": {
            **{name: lake.artifact_record(path) for name, path in output_paths.items()},
            "history_partition_index_v1": lake.artifact_record(index_path),
        },
    }
    lake.write_immutable_json(manifest_path, manifest)
    if publish_current:
        current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
        temporary = current_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "run_id": run_id, "manifest": str(manifest_path.relative_to(lake.root)),
            "regression_mode": "risk_only", "risk_factor_set_status": "candidate",
            "status": "active", "gate_version": G3_GATE_VERSION,
            "gate_config_sha": file_sha256(G3_GATE_CONFIG_PATH),
            "gate_status": "awaiting_attestation",
            "updated_at": utc_now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, current_path)
    return manifest_path
