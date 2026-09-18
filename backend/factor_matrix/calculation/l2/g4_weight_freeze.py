"""Freeze a selected G4.1a weight scheme and publish the resulting formal G3 run."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now
from .g4_weight_selection import (
    IMPLEMENTATION_ID, _feature_matrix, _fit_variance_models,
    _inverse_variance_weight, _normalize_and_clip,
)


FREEZE_IMPLEMENTATION_ID = "g4_1a_selected_weight_freeze_v1"


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def freeze_and_publish_g4_wls_weight(
    lake: DataLake, *, candidate_manifest_path: Path,
    config_path: Path = Path("config/g4_validation_protocol_v1.json"),
) -> Path:
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if (
        candidate.get("status") != "passed"
        or candidate.get("job") != "g4_1a_wls_weight_selection"
        or candidate.get("identity", {}).get("implementation_id") != IMPLEMENTATION_ID
    ):
        raise ValueError("G4_WEIGHT_FREEZE_CANDIDATE_INVALID")
    protocol_payload = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = protocol_payload["g4_1a_wls_weight_protocol"]
    if candidate["protocol"] != protocol:
        raise ValueError("G4_WEIGHT_FREEZE_PROTOCOL_MISMATCH")
    if candidate["winner"] != "structural":
        raise ValueError("G4_WEIGHT_FREEZE_WINNER_UNSUPPORTED")

    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("run_id") != candidate["baseline_run_id"]:
        raise RuntimeError("G4_WEIGHT_FREEZE_BASELINE_NOT_CURRENT")
    baseline_manifest_path = lake.root / current["manifest"]
    baseline = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    pit_path = lake.root / candidate["outputs"]["pit_variance_training_v1"]["path"]

    identity = {
        "candidate_run_id": candidate["run_id"],
        "winner": candidate["winner"],
        "baseline_run_id": baseline["run_id"],
        "input_hashes": {
            "candidate_manifest": file_sha256(candidate_manifest_path),
            "pit_training": file_sha256(pit_path),
            "protocol": file_sha256(config_path),
        },
        "implementation_id": FREEZE_IMPLEMENTATION_ID,
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_wls_freeze_{baseline['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_wls_freeze" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)

    dates = (
        pl.scan_parquet(pit_path).select("exposure_date").unique().sort("exposure_date")
        .collect()["exposure_date"].to_list()
    )
    train_days = int(protocol["evaluation"]["fit_window_trading_days"])
    refit_days = int(protocol["evaluation"]["test_window_trading_days"])
    minimum_prior = int(protocol["pit_policy"]["minimum_prior_residual_observations"])
    low, high = map(float, protocol["pit_policy"]["candidate_weight_ratio_clip"])
    sigma_floor = float(protocol["pit_policy"]["sigma_floor_relative_to_cross_section_median"])
    weight_frames: list[pl.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for start_index in range(0, len(dates), refit_days):
        test_dates = dates[start_index:start_index + refit_days]
        test = pl.scan_parquet(pit_path).filter(
            pl.col("exposure_date").is_in(test_dates)
        ).collect()
        use_structural = start_index >= train_days
        coefficients: np.ndarray | None = None
        if use_structural:
            training_dates = dates[start_index - train_days:start_index]
            training = pl.scan_parquet(pit_path).filter(
                pl.col("exposure_date").is_in(training_dates)
            ).collect()
            _, coefficients = _fit_variance_models(
                training, minimum_prior_observations=minimum_prior
            )
            coefficient_rows.append({
                "application_start": test_dates[0], "application_end": test_dates[-1],
                "training_start": training_dates[0], "training_end": training_dates[-1],
                "structural_coefficients_json": json.dumps(coefficients.tolist()),
                "fallback_sqrt_cap": False,
            })
        else:
            coefficient_rows.append({
                "application_start": test_dates[0], "application_end": test_dates[-1],
                "training_start": None, "training_end": None,
                "structural_coefficients_json": None, "fallback_sqrt_cap": True,
            })
        for key, daily in test.group_by("exposure_date", maintain_order=True):
            exposure_date = key[0] if isinstance(key, tuple) else key
            estimation = daily["in_estimation_domain"].to_numpy().astype(bool)
            cap = daily["float_mkt_cap"].to_numpy()
            raw = np.sqrt(cap) if coefficients is None else _inverse_variance_weight(
                _feature_matrix(daily) @ coefficients, estimation, sigma_floor
            )
            weight = _normalize_and_clip(raw, estimation, low, high)
            weight_frames.append(pl.DataFrame({
                "exposure_date": [exposure_date] * daily.height,
                "asset_id": daily["asset_id"],
                "candidate_weight": weight,
                "weight_scheme_id": [
                    "sqrt_cap_fallback" if coefficients is None else "structural"
                ] * daily.height,
            }))

    weights_path = run_dir / "selected_wls_weights_v1.parquet"
    coefficients_path = run_dir / "variance_model_refit_schedule_v1.parquet"
    _write_parquet(weights_path, pl.concat(weight_frames))
    _write_parquet(coefficients_path, pl.DataFrame(coefficient_rows))
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g4_1a_wls_weight_freeze",
        "status": "passed_pending_l1_rebuild", "created_at": utc_now().isoformat(), "identity": identity,
        "candidate_run_id": candidate["run_id"], "selected_candidate": "structural",
        "acceptance_diagnostic_passed": candidate["acceptance_diagnostic_passed"],
        "known_residuals": candidate["known_residuals"],
        "old_g3_run_id": baseline["run_id"],
        "publication": "weight_artifact_frozen_not_current",
        "required_next_steps": [
            "update weight_metric_contract artifact hash", "rerun L1", "rerun G2",
            "publish new risk_basis_id", "rerun G3", "run immutable reconciliation",
        ],
        "production_refit_days": refit_days,
        "initial_history_fallback": "normalized_clipped_sqrt_cap",
        "outputs": {
            "selected_wls_weights_v1": lake.artifact_record(weights_path),
            "variance_model_refit_schedule_v1": lake.artifact_record(coefficients_path),
        },
    }
    return lake.write_immutable_json(manifest_path, manifest)
