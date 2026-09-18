"""Quantify the known heteroskedasticity residual after the frozen G4.1a weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .g3_runner import load_current_g3_manifest


IMPLEMENTATION_ID = "g4_structural_weight_residual_v1"


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def run_g4_structural_weight_residual(
    lake: DataLake, *, candidate_manifest_path: Path,
    freeze_manifest_path: Path, reconciliation_manifest_path: Path,
) -> Path:
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_manifest_path.read_text(encoding="utf-8"))
    if freeze.get("new_g3_run_id") != g3["run_id"]:
        raise ValueError("G4_STRUCTURAL_RESIDUAL_FREEZE_CURRENT_MISMATCH")
    if freeze.get("candidate_run_id") != candidate.get("run_id"):
        raise ValueError("G4_STRUCTURAL_RESIDUAL_CANDIDATE_MISMATCH")
    if reconciliation.get("new_run_id") != g3["run_id"]:
        raise ValueError("G4_STRUCTURAL_RESIDUAL_RECONCILIATION_MISMATCH")

    specific_path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    weight_path = lake.root / freeze["outputs"]["selected_wls_weights_v1"]["path"]
    l1_current = json.loads(
        (lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json").read_text()
    )
    l1_manifest_path = lake.root / l1_current["manifest"]
    l1 = json.loads(l1_manifest_path.read_text())
    exposure_path = lake.root / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    candidate_daily_path = lake.root / candidate["outputs"]["candidate_daily_diagnostics_v1"]["path"]
    below_path = lake.root / reconciliation["outputs"]["factors_below_correlation_threshold_v1"]["path"]
    identity = {
        "g3_run_id": g3["run_id"], "candidate_run_id": candidate["run_id"],
        "freeze_run_id": freeze["run_id"], "reconciliation_run_id": reconciliation["run_id"],
        "implementation_id": IMPLEMENTATION_ID,
        "input_hashes": {
            "g3_manifest": file_sha256(g3_manifest_path),
            "candidate_manifest": file_sha256(candidate_manifest_path),
            "freeze_manifest": file_sha256(freeze_manifest_path),
            "reconciliation_manifest": file_sha256(reconciliation_manifest_path),
            "specific_returns": file_sha256(specific_path),
            "selected_weights": file_sha256(weight_path),
            "risk_exposure": file_sha256(exposure_path),
        },
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_structural_residual_{g3['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_structural_weight_residual" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)

    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        base = f"""
          WITH joined AS (
            SELECT s.trade_date,s.exposure_date,s.asset_id,s.specific_return,
                   s.is_outlier_flagged,w.candidate_weight,w.weight_scheme_id,
                   e.risk_liquidity
            FROM read_parquet('{_q(specific_path)}') s
            JOIN read_parquet('{_q(weight_path)}') w USING(exposure_date,asset_id)
            JOIN read_parquet('{_q(exposure_path)}') e
              ON e.trade_date=s.exposure_date AND e.asset_id=s.asset_id
            WHERE s.in_estimation_domain AND s.specific_return IS NOT NULL
          ), ranked AS (
            SELECT *,ntile(10) OVER(
              PARTITION BY exposure_date ORDER BY risk_liquidity,asset_id
            ) liquidity_decile,
            sqrt(candidate_weight)*specific_return whitened_residual
            FROM joined
          ), daily AS (
            SELECT weight_scheme_id,exposure_date,liquidity_decile,count(*) n,
                   var_pop(specific_return) raw_variance,
                   var_pop(whitened_residual) whitened_variance,
                   avg(candidate_weight) mean_weight,
                   quantile_cont(candidate_weight,.5) median_weight,
                   avg((abs(candidate_weight-.1)<1e-12)::INTEGER) lower_clip_rate,
                   avg((abs(candidate_weight-10)<1e-12)::INTEGER) upper_clip_rate,
                   avg(is_outlier_flagged::INTEGER) downweight_rate
            FROM ranked GROUP BY weight_scheme_id,exposure_date,liquidity_decile
          )
        """
        liquidity = connection.execute(base + """
          SELECT weight_scheme_id,liquidity_decile,sum(n) n_observations,
                 median(raw_variance) median_daily_raw_variance,
                 median(whitened_variance) median_daily_whitened_variance,
                 median(mean_weight) median_daily_mean_weight,
                 median(median_weight) median_daily_median_weight,
                 sum(n*lower_clip_rate)/sum(n) lower_clip_rate,
                 sum(n*upper_clip_rate)/sum(n) upper_clip_rate,
                 sum(n*downweight_rate)/sum(n) downweight_rate
          FROM daily GROUP BY weight_scheme_id,liquidity_decile
          ORDER BY weight_scheme_id,liquidity_decile
        """).pl()
        binding = connection.execute(base + """
          SELECT weight_scheme_id,count(*) n_observations,
                 quantile_cont(candidate_weight,.01) weight_p01,
                 quantile_cont(candidate_weight,.5) weight_p50,
                 quantile_cont(candidate_weight,.99) weight_p99,
                 min(candidate_weight) minimum_weight,max(candidate_weight) maximum_weight,
                 avg((abs(candidate_weight-.1)<1e-12)::INTEGER) lower_clip_rate,
                 avg((abs(candidate_weight-10)<1e-12)::INTEGER) upper_clip_rate
          FROM ranked GROUP BY weight_scheme_id ORDER BY weight_scheme_id
        """).pl()
    finally:
        connection.close()

    candidate_daily = pl.read_parquet(candidate_daily_path)
    secondary_guard = candidate_daily.group_by("candidate").agg(
        pl.len().alias("n_daily_evaluations"),
        pl.col("secondary_guard_score").mean().alias("mean_secondary_guard_score"),
        pl.col("secondary_guard_score").median().alias("median_secondary_guard_score"),
        pl.col("primary_score").mean().alias("mean_primary_score"),
    ).sort("mean_secondary_guard_score")
    below = pl.read_parquet(below_path).with_columns(
        pl.when(pl.col("factor_id").str.starts_with("risk_industry_"))
        .then(pl.lit("industry"))
        .when(pl.col("factor_id").str.starts_with("risk_index_"))
        .then(pl.lit("index_membership"))
        .when(pl.col("factor_id").is_in([
            "risk_nonlinear_size", "risk_listing_age", "risk_residual_volatility",
            "risk_liquidity", "risk_beta", "risk_size",
        ])).then(pl.lit("style"))
        .otherwise(pl.lit("other")).alias("factor_class")
    )
    structural = liquidity.filter(pl.col("weight_scheme_id") == "structural")
    raw_ratio = float(
        structural["median_daily_raw_variance"].max()
        / structural["median_daily_raw_variance"].min()
    )
    whitened_ratio = float(
        structural["median_daily_whitened_variance"].max()
        / structural["median_daily_whitened_variance"].min()
    )
    group_weight_ratio = float(
        structural["median_daily_median_weight"].max()
        / structural["median_daily_median_weight"].min()
    )
    structural_binding = binding.filter(pl.col("weight_scheme_id") == "structural").row(
        0, named=True
    )
    diagnosis = (
        "variance_gradient_partially_narrowed_auxiliary_model_residual"
        if whitened_ratio < raw_ratio and whitened_ratio > 1.5
        else "variance_gradient_flattened_check_huber_scale"
        if whitened_ratio <= 1.5
        else "variance_gradient_not_narrowed"
    )
    output_records: dict[str, Any] = {}
    for name, frame in {
        "liquidity_whitening_diagnostic_v1": liquidity,
        "weight_clip_binding_v1": binding,
        "secondary_guard_summary_v1": secondary_guard,
        "factor_reconciliation_below_threshold_v1": below,
    }.items():
        path = run_dir / f"{name}.parquet"
        _write(path, frame)
        output_records[name] = lake.artifact_record(path)
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "g4_structural_weight_known_residual", "status": "passed",
        "created_at": utc_now().isoformat(), "identity": identity,
        "g3_run_id": g3["run_id"], "candidate_run_id": candidate["run_id"],
        "non_decisional": True,
        "headline": {
            "structural_raw_variance_liquidity_range_ratio": raw_ratio,
            "structural_whitened_variance_liquidity_range_ratio": whitened_ratio,
            "structural_group_median_weight_range_ratio": group_weight_ratio,
            "structural_observation_weight_p99_over_p01": (
                float(structural_binding["weight_p99"] / structural_binding["weight_p01"])
            ),
            "structural_lower_clip_rate": float(structural_binding["lower_clip_rate"]),
            "structural_upper_clip_rate": float(structural_binding["upper_clip_rate"]),
            "diagnosis": diagnosis,
            "secondary_guard_winner": secondary_guard["candidate"][0],
            "factors_below_0_99_count": below.height,
            "style_factors_below_0_99": below.filter(
                pl.col("factor_class") == "style"
            )["factor_id"].to_list(),
        },
        "known_residual": {
            "id": "structural_weight_partial_heteroskedasticity_absorption",
            "action": "quantify_and_continue_to_g4_2_without_reopening_weight_selection",
        },
        "outputs": output_records,
    }
    return lake.write_immutable_json(manifest_path, manifest)
