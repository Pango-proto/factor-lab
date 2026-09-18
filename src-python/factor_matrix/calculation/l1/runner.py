"""Versioned G2/L1 risk-exposure orchestration and immutable publication."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ...factor_engine.registry import FactorRegistry
from ...factor_engine.store import FactorRegistryStore
from ...research_protocol import ResearchProtocol
from ...revisioned_silver import SilverVersionLedger
from ...storage import (
    DataLake, file_sha256, json_hash, utc_now,
)
from ...weight_metric_contract import load_weight_metric_contract
from .builder import L1DailyBuild, STYLE_ORDER, build_l1_day
from .config import assert_exposure_publish_allowed, load_l1_risk_exposure_config


DEFAULT_L1_CONFIG = Path("config/l1_risk_exposure_v1.json")
DEFAULT_LISTING_POLICY = Path("config/new_listing_policy_v1.json")
DEFAULT_RISK_CANDIDATES = Path("config/risk_factor_set_candidate_v1.json")
DEFAULT_PROTOCOL = Path("config/research_protocol_v1.json")
DEFAULT_WEIGHT_METRIC_CONTRACT = Path("config/weight_metric_contract_v1.json")


def l1_calculation_code_hash() -> str:
    """Hash only code that can change L1 numeric outputs or their lineage.

    History diagnostics and CLI presentation are deliberately excluded so a
    reporting-only edit cannot invalidate 86 immutable calculation partitions.
    """
    package_root = Path(__file__).resolve().parents[2]
    files = [
        path for path in Path(__file__).resolve().parent.glob("*.py")
        if path.name not in {"history_diagnostics.py", "__init__.py"}
    ]
    files.extend([
        package_root / "canonical_definitions.py",
        package_root / "research_protocol.py",
        package_root / "factor_engine" / "contracts.py",
        package_root / "factor_engine" / "registry.py",
    ])
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _registry_snapshot(
    lake: DataLake, registry: FactorRegistry, member_ids: tuple[str, ...],
) -> tuple[str, Path, dict[str, Any]]:
    factors = []
    for factor_id in member_ids:
        spec = registry.get(factor_id).spec
        factors.append({
            "factor_id": spec.factor_id,
            "version": spec.version,
            "family": spec.family.value,
            "role": spec.role.value,
            "feature_key": spec.feature_key,
            "formula_expr": spec.formula_expr,
            "params": dict(spec.params),
            "lookback_days": spec.lookback_days,
            "min_obs": spec.min_obs,
            "orthogonalize_after": list(spec.orthogonalize_after),
            "depends_on": list(spec.depends_on),
            "code_path": spec.code_path,
            "code_sha": spec.code_sha,
            "status": spec.status.value,
        })
    payload = {
        "schema_version": 1,
        "family": "risk",
        "status": "candidate_not_frozen",
        "factors": factors,
    }
    snapshot_id = f"risk_registry_snapshot_{json_hash(payload)[:16]}"
    payload["registry_snapshot_id"] = snapshot_id
    path = lake.metadata / "registry_snapshots" / f"{snapshot_id}.json"
    lake.write_immutable_json(path, payload)
    return snapshot_id, path, payload


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _load_inputs(
    *, lake: DataLake, start: date, end: date,
    universe_metadata: dict[str, Any], board_benchmark_summary_path: Path,
) -> dict[str, pl.DataFrame | Path | dict[str, Any]]:
    universe_path = lake.root / universe_metadata["artifact"]
    board_summary = json.loads(board_benchmark_summary_path.read_text(encoding="utf-8"))
    board_path = board_benchmark_summary_path.parent / "board_benchmark_daily.parquet"
    board = pl.read_parquet(board_path)
    available_dates = board.filter(pl.col("trade_date") <= end)["trade_date"].unique().sort()
    if available_dates.is_empty():
        raise RuntimeError("L1_BOARD_BENCHMARK_HISTORY_EMPTY")
    pre_start_dates = available_dates.filter(available_dates <= start).tail(120).to_list()
    history_start = min(pre_start_dates) if pre_start_dates else available_dates.min()
    universe = pl.scan_parquet(universe_path).filter(
        pl.col("trade_date").is_between(history_start, end)
    ).collect()
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    membership_path = lake.silver / "benchmark_membership_history" / "data.parquet"
    returns = pl.scan_parquet(returns_path).filter(
        pl.col("trade_date").is_between(history_start, end)
    ).collect()
    valuation = pl.scan_parquet(valuation_path).filter(
        pl.col("trade_date").is_between(history_start, end)
    ).collect()
    membership = pl.read_parquet(membership_path)
    return {
        "universe": universe,
        "returns": returns,
        "valuation": valuation,
        "membership": membership,
        "board": board.filter(pl.col("trade_date").is_between(history_start, end)),
        "universe_path": universe_path,
        "board_path": board_path,
        "board_summary": board_summary,
        "returns_path": returns_path,
        "valuation_path": valuation_path,
        "membership_path": membership_path,
    }


def run_l1_risk_exposure(
    lake: DataLake,
    *,
    start: date,
    end: date,
    universe_metadata_path: Path,
    board_benchmark_summary_path: Path,
    universe_variant: str = "frozen_d0",
    publish_current: bool = False,
    l1_config_path: Path = DEFAULT_L1_CONFIG,
    listing_policy_path: Path = DEFAULT_LISTING_POLICY,
    risk_candidates_path: Path = DEFAULT_RISK_CANDIDATES,
    protocol_path: Path = DEFAULT_PROTOCOL,
    weight_metric_contract_path: Path = DEFAULT_WEIGHT_METRIC_CONTRACT,
    _derived_params_override: dict[str, Any] | None = None,
    _allow_latest_invalid: bool = False,
) -> Path:
    if end < start:
        raise ValueError("L1_DATE_RANGE_INVALID")
    if universe_variant != "frozen_d0":
        raise ValueError("L1_FORMAL_RUN_REQUIRES_FROZEN_D0")
    assert_exposure_publish_allowed(
        policy_path=listing_policy_path,
        universe_variant=universe_variant,
        update_current_pointer=publish_current,
    )
    l1_config = load_l1_risk_exposure_config(l1_config_path)
    protocol = ResearchProtocol.load(protocol_path)
    risk_candidates = json.loads(risk_candidates_path.read_text(encoding="utf-8"))
    weight_contract = load_weight_metric_contract(weight_metric_contract_path)
    weight_artifact_path = weight_contract.resolve_artifact(lake.root)
    metric_weights = pl.scan_parquet(weight_artifact_path).filter(
        pl.col("exposure_date").is_between(start, end)
    ).select("exposure_date", "asset_id", "candidate_weight").collect()
    candidate_ids = tuple(item["factor_id"] for item in risk_candidates["members"])
    if tuple(
        item for item in risk_candidates["candidate_selection_order"] if item in STYLE_ORDER
    ) != STYLE_ORDER:
        raise ValueError("L1_STYLE_ORDER_DIFFERS_FROM_PREREGISTERED_ORDER")
    registry = FactorRegistry.discover()
    FactorRegistryStore(lake.metadata / "factor_registry.sqlite").sync_definitions(registry)
    registry_snapshot_id, registry_snapshot_path, registry_payload = _registry_snapshot(
        lake, registry, candidate_ids
    )
    universe_metadata = json.loads(universe_metadata_path.read_text(encoding="utf-8"))
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("L1_REQUIRES_SILVER_VERSION")
    if universe_metadata["silver_version_id"] != version["version_id"]:
        raise RuntimeError("L1_UNIVERSE_SILVER_VERSION_MISMATCH")
    loaded = _load_inputs(
        lake=lake, start=start, end=end,
        universe_metadata=universe_metadata,
        board_benchmark_summary_path=board_benchmark_summary_path,
    )
    universe = loaded["universe"]
    trading_dates = (
        universe.filter(pl.col("trade_date").is_between(start, end))["trade_date"]
        .unique().sort().to_list()
    )
    if not trading_dates:
        raise RuntimeError("L1_REQUESTED_DATES_EMPTY")
    latest_size = universe.filter(pl.col("trade_date") == trading_dates[-1]).height
    if _derived_params_override is None:
        derived_snapshot = protocol.derive(
            factor_count=len(candidate_ids), maximum_evaluation_horizon_days=1,
            cross_section_size=latest_size,
            available_estimation_days=max(len(trading_dates), 1),
        )
        derived_params = dict(derived_snapshot["values"])
        derived_params["new_listing_days_by_regime"] = json.loads(
            listing_policy_path.read_text(encoding="utf-8")
        )["d_star_reference"]["values"]
        derived_params["unresolved_required"] = []
    else:
        derived_params = dict(_derived_params_override)
    risk_factor_set_id = f"risk_candidates_{json_hash(risk_candidates)[:16]}"
    builds: list[L1DailyBuild] = []
    for trade_date in trading_dates:
        daily_metric = metric_weights.filter(pl.col("exposure_date") == trade_date)
        metric_by_asset = (
            None if daily_metric.is_empty() else dict(zip(
                daily_metric["asset_id"].to_list(),
                daily_metric["candidate_weight"].to_list(),
            ))
        )
        metric_scheme_id = (
            weight_contract.missing_date_fallback_scheme_id
            if metric_by_asset is None else weight_contract.orthogonalization_scheme_id
        )
        builds.append(build_l1_day(
            trade_date=trade_date,
            universe_day=universe.filter(pl.col("trade_date") == trade_date),
            universe_history=universe.filter(pl.col("trade_date") <= trade_date),
            valuation=loaded["valuation"].filter(pl.col("trade_date") <= trade_date),
            returns=loaded["returns"].filter(pl.col("trade_date") <= trade_date),
            board_benchmarks=loaded["board"].filter(pl.col("trade_date") <= trade_date),
            benchmark_membership=loaded["membership"],
            registry=registry,
            derived_params=derived_params,
            new_listing_policy_path=listing_policy_path,
            risk_factor_set_id=risk_factor_set_id,
            metric_weights_by_asset=metric_by_asset,
            metric_weight_scheme_id=metric_scheme_id,
        ))
    exposure = pl.concat([item.exposure for item in builds], how="diagonal_relaxed")
    quality = pl.concat([item.quality for item in builds], how="diagonal_relaxed")
    checks = [check for item in builds for check in item.checks]
    daily_validity = exposure.group_by("trade_date").agg(
        pl.col("is_valid").all().alias("is_valid")
    ).sort("trade_date")
    invalid_dates = daily_validity.filter(~pl.col("is_valid"))["trade_date"].to_list()
    invalid_ratio = len(invalid_dates) / len(trading_dates)
    if not _allow_latest_invalid and trading_dates[-1] in invalid_dates:
        raise RuntimeError("L1_LATEST_DATE_INVALID")
    if not _allow_latest_invalid and invalid_ratio > float(
        derived_params["maximum_invalid_date_ratio"]
    ):
        raise RuntimeError(
            f"L1_INVALID_DATE_RATIO_FAILED observed={invalid_ratio} "
            f"maximum={derived_params['maximum_invalid_date_ratio']}"
        )

    config_paths = (
        l1_config_path, listing_policy_path, risk_candidates_path, protocol_path,
        Path("config/risk_descriptor_parameters_v1.json"),
        weight_metric_contract_path,
    )
    silver_manifest_path = lake.metadata / "silver_versions" / f"{version['version_id']}.json"
    input_hashes = {
        "silver_version_manifest": _input_record(silver_manifest_path),
        "tradable_universe": _input_record(loaded["universe_path"]),
        "tradable_universe_metadata": _input_record(universe_metadata_path),
        "board_benchmark_daily": _input_record(loaded["board_path"]),
        "board_benchmark_summary": _input_record(board_benchmark_summary_path),
        "registry_snapshot": _input_record(registry_snapshot_path),
        "weight_metric_artifact": _input_record(weight_artifact_path),
        **{f"config:{path.name}": _input_record(path) for path in config_paths},
    }
    identity = {
        "start": start.isoformat(), "end": end.isoformat(),
        "config_sha": json_hash({key: value["sha256"] for key, value in input_hashes.items()}),
        "code_sha": l1_calculation_code_hash(),
        "input_hashes": {key: value["sha256"] for key, value in input_hashes.items()},
        "derived_params": derived_params,
        "universe_variant": universe_variant,
    }
    run_id = f"l1_risk_exposure_{end:%Y%m%d}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "gold" / "risk_exposure_matrix" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    exposure_path = run_dir / "risk_exposure_matrix_v1.parquet"
    quality_path = run_dir / "exposure_quality_v1.parquet"
    checks_path = run_dir / "quality_checks.json"
    _write_parquet_atomic(exposure_path, exposure)
    _write_parquet_atomic(quality_path, quality)
    lake.write_immutable_json(checks_path, {
        "status": "passed", "checks": checks,
        "invalid_dates": [value.isoformat() for value in invalid_dates],
        "invalid_date_ratio": invalid_ratio,
    })
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "l1_risk_exposure",
        "execution_status": (
            "passed_with_invalid_dates" if invalid_dates else "passed"
        ),
        "is_history_partition": _allow_latest_invalid,
        "publish_mode": "formal_current" if publish_current else "validation_only",
        "start": start.isoformat(), "end": end.isoformat(),
        "created_at": utc_now().isoformat(),
        "silver_version_id": version["version_id"],
        "tradable_universe_run_id": universe_metadata["run_id"],
        "board_benchmark_run_id": loaded["board_summary"]["run_id"],
        "registry_snapshot_id": registry_snapshot_id,
        "risk_factor_set_id": risk_factor_set_id,
        "risk_factor_set_status": risk_candidates["status"],
        "risk_metric_id": weight_contract.risk_metric_id,
        "weight_metric_contract": weight_contract.as_identity(),
        "universe_variant": universe_variant,
        "config_sha": identity["config_sha"],
        "code_sha": identity["code_sha"],
        "input_hashes": input_hashes,
        "derived_params": derived_params,
        "l1_contract": asdict(l1_config),
        "outputs": {
            "risk_exposure_matrix_v1": lake.artifact_record(exposure_path),
            "exposure_quality_v1": lake.artifact_record(quality_path),
            "quality_checks": lake.artifact_record(checks_path),
        },
        "counts": {
            "dates": len(trading_dates), "rows": exposure.height,
            "invalid_dates": len(invalid_dates),
            "invalid_date_ratio": invalid_ratio,
            "quality_rows": quality.height, "risk_columns": len(
                [column for column in exposure.columns if column.startswith("risk_")]
            ),
        },
    }
    lake.write_immutable_json(manifest_path, manifest)
    lake.register_calculation(
        run_id=run_id, job="l1_risk_exposure", as_of=end,
        mode="formal" if publish_current else "validation_only",
        config={"start": start.isoformat(), "end": end.isoformat(),
                "universe_variant": universe_variant},
        config_hash=identity["config_sha"], code_hash=identity["code_sha"],
        parent_run_ids=[
            universe_metadata["run_id"], loaded["board_summary"]["run_id"],
            version["version_id"], registry_snapshot_id,
        ],
        inputs={key: Path(value["path"]) for key, value in input_hashes.items()},
        outputs={
            "risk_exposure_matrix_v1": exposure_path,
            "exposure_quality_v1": quality_path,
            "manifest": manifest_path,
        },
        quality_gate={"status": "passed", "checks": len(checks)},
        extra={"universe_variant": universe_variant,
               "registry_snapshot_id": registry_snapshot_id},
    )
    if publish_current:
        current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
        temporary = current_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "run_id": run_id,
            "manifest": str(manifest_path.relative_to(lake.root)),
            "universe_variant": universe_variant,
            "updated_at": utc_now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, current_path)
    return manifest_path
