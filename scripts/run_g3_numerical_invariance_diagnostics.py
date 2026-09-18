#!/usr/bin/env python3
"""Validate G3 condition-gate parameterization and numerical solve accuracy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import polars as pl


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))

from factor_matrix.calculation.l2.config import load_return_decomposition_config  # noqa: E402
from factor_matrix.calculation.l2.contracts import RegressionInput, RegressionMode  # noqa: E402
from factor_matrix.calculation.l2.design_matrix import (  # noqa: E402
    build_daily_categorical_constrained_design,
)
from factor_matrix.calculation.l2.g3_runner import _active_factor_columns  # noqa: E402
from factor_matrix.calculation.l2.linear_algebra import null_space_basis  # noqa: E402
from factor_matrix.calculation.l2.return_decomposition import decompose_cross_section  # noqa: E402
from factor_matrix.storage import file_sha256, json_hash  # noqa: E402


DATA = PROJECT / "data"
CONFIG_PATH = PROJECT / "config/g3_numerical_invariance_protocol_v1.json"
L2_CONFIG_PATH = PROJECT / "config/l2_return_decomposition_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _equilibrated_condition(gram: np.ndarray) -> float:
    diagonal = np.diag(gram)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        return float("inf")
    inverse = 1.0 / np.sqrt(diagonal)
    equilibrated = inverse[:, None] * gram * inverse[None, :]
    value = float(np.linalg.cond(equilibrated))
    return value if np.isfinite(value) else float("inf")


def _scale_invariant_system(
    inputs: RegressionInput, effective_weights_by_asset: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use = np.asarray(inputs.is_tradable, dtype=bool)
    x = np.asarray(inputs.exposures, dtype=float)[use]
    y = np.asarray(inputs.realized_returns, dtype=float)[use]
    assets = np.asarray(inputs.asset_ids, dtype=object)[use]
    w = np.asarray([effective_weights_by_asset[str(asset)] for asset in assets], dtype=float)
    scales = np.sqrt(np.sum(w[:, None] * np.square(x), axis=0))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise RuntimeError("NUMERICAL_DIAGNOSTIC_COLUMN_SCALE_INVALID")
    canonical_x = x / scales[None, :]
    constraints = np.asarray(inputs.equality_constraints, dtype=float)
    canonical_constraints = (
        constraints / scales[None, :]
        if constraints.size else np.empty((0, x.shape[1]), dtype=float)
    )
    basis = np.asarray(
        null_space_basis(canonical_constraints.tolist(), x.shape[1]), dtype=float
    )
    reduced_x = canonical_x @ basis
    gram = reduced_x.T @ (w[:, None] * reduced_x)
    rhs = reduced_x.T @ (w * y)
    return gram, rhs, basis, scales


def _gram(
    inputs: RegressionInput, effective_weights_by_asset: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use = np.asarray(inputs.is_tradable, dtype=bool)
    x = np.asarray(inputs.exposures, dtype=float)[use]
    y = np.asarray(inputs.realized_returns, dtype=float)[use]
    assets = np.asarray(inputs.asset_ids, dtype=object)[use]
    w = np.asarray([effective_weights_by_asset[str(asset)] for asset in assets], dtype=float)
    basis = np.asarray(
        null_space_basis(
            [list(row) for row in inputs.equality_constraints], len(inputs.factor_ids)
        ),
        dtype=float,
    )
    reduced_x = x @ basis
    gram = reduced_x.T @ (w[:, None] * reduced_x)
    rhs = reduced_x.T @ (w * y)
    return gram, rhs, basis, x, y


def _high_precision_solve(
    gram: np.ndarray, rhs: np.ndarray, basis: np.ndarray,
    current_coefficients: np.ndarray, x: np.ndarray, constraints: np.ndarray,
    digits: int,
) -> dict[str, float | bool]:
    mp.mp.dps = digits
    mp_gram = mp.matrix([[mp.mpf(float(value)) for value in row] for row in gram])
    mp_rhs = mp.matrix([mp.mpf(float(value)) for value in rhs])
    reduced_high = mp.lu_solve(mp_gram, mp_rhs)
    high = np.asarray([
        float(sum(mp.mpf(float(basis[row, column])) * reduced_high[column]
                  for column in range(basis.shape[1])))
        for row in range(basis.shape[0])
    ])
    denominator = float(np.linalg.norm(high))
    relative = float(np.linalg.norm(current_coefficients - high) / denominator)
    fitted_difference = float(np.max(np.abs(x @ current_coefficients - x @ high)))
    constraint_error = float(np.max(np.abs(constraints @ high))) if constraints.size else 0.0
    direct_float = basis @ np.linalg.solve(gram, rhs)
    direct_relative = float(np.linalg.norm(direct_float - high) / denominator)
    return {
        "relative_factor_solution_error": relative,
        "direct_float64_solve_relative_error": direct_relative,
        "maximum_absolute_fitted_return_difference": fitted_difference,
        "maximum_constraint_error_high_precision": constraint_error,
        "passed": relative <= 1e-12,
    }


def _scale_industries(inputs: RegressionInput, multiplier: float) -> RegressionInput:
    industry = np.asarray([
        factor_id.startswith("risk_industry_") for factor_id in inputs.factor_ids
    ], dtype=bool)
    scale = np.where(industry, multiplier, 1.0)
    exposures = np.asarray(inputs.exposures, dtype=float) * scale[None, :]
    constraints = np.asarray(inputs.equality_constraints, dtype=float) * scale[None, :]
    return RegressionInput(
        asset_ids=inputs.asset_ids,
        factor_ids=inputs.factor_ids,
        factor_families=inputs.factor_families,
        exposures=tuple(tuple(float(value) for value in row) for row in exposures),
        realized_returns=inputs.realized_returns,
        base_weights=inputs.base_weights,
        is_tradable=inputs.is_tradable,
        equality_constraints=tuple(
            tuple(float(value) for value in row) for row in constraints
        ),
    )


def _g3_manifests(risk_basis_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in (DATA / "gold/l2b_risk_only").glob("run_id=*/_MANIFEST.json"):
        payload = _load(path)
        if payload.get("risk_basis_id") == risk_basis_id:
            rows.append(payload)
    rows.sort(key=lambda item: item["start"])
    if len(rows) != 84:
        raise RuntimeError(f"NUMERICAL_DIAGNOSTIC_EXPECTS_84_G3_PARTITIONS observed={len(rows)}")
    return rows


def _daily_frame(
    manifest: dict[str, Any], exposure_path: Path, weight_path: Path,
    valuation_path: Path, returns_path: Path, risk_columns: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    specific_path = DATA / manifest["outputs"]["specific_returns_v1"]["path"]
    factor_path = DATA / manifest["outputs"]["factor_returns_v1"]["path"]
    quality_path = DATA / manifest["outputs"]["factor_regression_quality_v1"]["path"]
    specific = pl.read_parquet(specific_path)
    exposure_dates = specific["exposure_date"].unique().to_list()
    trade_dates = specific["trade_date"].unique().to_list()
    exposures = (
        pl.scan_parquet(exposure_path)
        .filter(pl.col("trade_date").is_in(exposure_dates) & pl.col("is_valid"))
        .select(pl.col("trade_date").alias("exposure_date"), "asset_id", "board_id", *risk_columns)
        .collect()
    )
    weights = (
        pl.scan_parquet(weight_path).filter(pl.col("exposure_date").is_in(exposure_dates))
        .select("exposure_date", "asset_id", "candidate_weight").collect()
    )
    valuation = (
        pl.scan_parquet(valuation_path).filter(pl.col("trade_date").is_in(exposure_dates))
        .select(pl.col("trade_date").alias("exposure_date"), "asset_id", "float_mkt_cap")
        .collect()
    )
    returns = (
        pl.scan_parquet(returns_path).filter(pl.col("trade_date").is_in(trade_dates))
        .select("trade_date", "asset_id", pl.col("total_return").alias("realized_return"))
        .collect()
    )
    joined = (
        specific.join(exposures, on=("exposure_date", "asset_id"), how="inner", validate="m:1")
        .join(weights, on=("exposure_date", "asset_id"), how="left", validate="m:1")
        .join(valuation, on=("exposure_date", "asset_id"), how="left", validate="m:1")
        .join(returns, on=("trade_date", "asset_id"), how="left", validate="m:1")
        .with_columns(pl.lit(True).alias("model_eligible"))
    )
    if joined.select(pl.any_horizontal(
        pl.col("candidate_weight").is_null(), pl.col("float_mkt_cap").is_null(),
        pl.col("realized_return").is_null(),
    ).any()).item():
        raise RuntimeError("NUMERICAL_DIAGNOSTIC_JOIN_COVERAGE_INCOMPLETE")
    return joined, pl.read_parquet(factor_path), pl.read_parquet(quality_path)


def main() -> None:
    protocol = _load(CONFIG_PATH)
    l1_current = _load(DATA / "gold/risk_exposure_matrix/_CURRENT.json")
    l1_manifest_path = DATA / l1_current["manifest"]
    l1 = _load(l1_manifest_path)
    exposure_path = DATA / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    weight_path = DATA / l1["weight_metric_contract"]["artifact_path"]
    valuation_path = DATA / "silver/valuation_daily/data.parquet"
    returns_path = DATA / "silver/returns_daily/data.parquet"
    schema = pl.read_parquet_schema(exposure_path)
    risk_columns = [
        name for name, dtype in schema.items()
        if name.startswith("risk_") and dtype.is_float()
    ]
    manifests = _g3_manifests(l1["risk_basis_id"])
    config = load_return_decomposition_config(
        L2_CONFIG_PATH, derived_params=l1["derived_params"]
    )
    minimum_members = int(l1["derived_params"]["minimum_category_members"])
    c_date = date.fromisoformat(protocol["parameterization_invariance_probe"]["trade_date"])
    high_dates = {
        date.fromisoformat(value) for value in protocol["high_precision_validation"]["trade_dates"]
    }
    target_dates = high_dates | {c_date}
    tail_rows: list[dict[str, Any]] = []
    c_results: list[dict[str, Any]] = []
    high_results: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []

    for number, manifest in enumerate(manifests, start=1):
        joined, factor_returns, quality = _daily_frame(
            manifest, exposure_path, weight_path, valuation_path, returns_path, risk_columns
        )
        for key, frame in joined.partition_by("trade_date", as_dict=True).items():
            trade_date = key[0] if isinstance(key, tuple) else key
            factor_columns, blocks = _active_factor_columns(frame, risk_columns)
            design = build_daily_categorical_constrained_design(
                frame,
                exposure_columns=factor_columns,
                family_by_column={column: "risk" for column in factor_columns},
                categorical_blocks=blocks,
                minimum_category_members=minimum_members,
                estimation_column="in_estimation_domain",
                base_weight_column="candidate_weight",
            )
            effective = {
                row["asset_id"]: float(row["estimation_weight"])
                for row in frame.filter(pl.col("in_estimation_domain")).select(
                    "asset_id", "estimation_weight"
                ).iter_rows(named=True)
            }
            gram, rhs, basis, x, _ = _gram(design.regression_input, effective)
            observed = quality.filter(pl.col("trade_date") == trade_date).row(0, named=True)
            raw_condition = float(np.linalg.cond(gram))
            if not math.isclose(
                raw_condition, float(observed["condition_number"]), rel_tol=2e-10, abs_tol=1e-9
            ):
                raise RuntimeError("NUMERICAL_DIAGNOSTIC_OBSERVED_CONDITION_MISMATCH")
            tail_rows.append({
                "trade_date": trade_date,
                "exposure_date": frame["exposure_date"][0],
                "raw_effective_condition_number": raw_condition,
                "equilibrated_effective_condition_number": _equilibrated_condition(gram),
                "scale_invariant_effective_condition_number": float(np.linalg.cond(
                    _scale_invariant_system(design.regression_input, effective)[0]
                )),
            })
            if trade_date not in target_dates:
                continue
            current_rows = factor_returns.filter(pl.col("trade_date") == trade_date)
            current_by_factor = dict(zip(
                current_rows["factor_id"].to_list(), current_rows["factor_return"].to_list()
            ))
            current = np.asarray([current_by_factor[name] for name in factor_columns], dtype=float)
            if trade_date in high_dates:
                high = _high_precision_solve(
                    gram, rhs, basis, current, x,
                    np.asarray(design.regression_input.equality_constraints, dtype=float),
                    int(protocol["high_precision_validation"]["precision_decimal_digits"]),
                )
                high_results.append({
                    "trade_date": trade_date,
                    "raw_effective_condition_number": raw_condition,
                    "equilibrated_effective_condition_number": _equilibrated_condition(gram),
                    **high,
                })
            if trade_date == c_date:
                baseline_result = None
                baseline_fitted = None
                for multiplier in protocol["parameterization_invariance_probe"]["industry_column_multipliers"]:
                    scaled = _scale_industries(design.regression_input, float(multiplier))
                    result = decompose_cross_section(
                        scaled, mode=RegressionMode.RISK_ONLY, config=config
                    )
                    scaled_effective = {
                        asset: float(weight) for asset, use, weight in zip(
                            result.included_assets, result.in_estimation_domain,
                            result.estimation_weights,
                        ) if use and weight is not None
                    }
                    scaled_gram, _, _, _, _ = _gram(scaled, scaled_effective)
                    canonical_gram, _, _, _ = _scale_invariant_system(
                        scaled, scaled_effective
                    )
                    fitted = np.asarray(scaled.exposures, dtype=float) @ np.asarray(
                        result.factor_returns, dtype=float
                    )
                    if float(multiplier) == 1.0:
                        baseline_result = result
                        baseline_fitted = fitted
                    c_results.append({
                        "trade_date": trade_date,
                        "industry_column_multiplier": float(multiplier),
                        "raw_effective_condition_number": result.condition_number,
                        "equilibrated_effective_condition_number": _equilibrated_condition(scaled_gram),
                        "scale_invariant_effective_condition_number": float(
                            np.linalg.cond(canonical_gram)
                        ),
                        "r_squared": result.r_squared,
                        "unweighted_r_squared": result.unweighted_r_squared,
                        "constraint_error": result.constraint_error,
                        "factor_returns_json": json.dumps(dict(zip(
                            scaled.factor_ids, result.factor_returns
                        )), sort_keys=True),
                        "fitted_returns_json": json.dumps(fitted.tolist()),
                        "specific_returns_json": json.dumps(list(result.specific_returns)),
                    })
                if baseline_result is None or baseline_fitted is None:
                    raise RuntimeError("NUMERICAL_DIAGNOSTIC_C_BASELINE_MISSING")
        input_records.append({
            "run_id": manifest["run_id"],
            "specific_returns_sha256": manifest["outputs"]["specific_returns_v1"]["sha256"],
            "factor_returns_sha256": manifest["outputs"]["factor_returns_v1"]["sha256"],
            "quality_sha256": manifest["outputs"]["factor_regression_quality_v1"]["sha256"],
        })
        print(f"completed partition {number}/{len(manifests)}", flush=True)

    tails = pl.DataFrame(tail_rows).sort("trade_date")
    c_frame = pl.DataFrame(c_results).sort("industry_column_multiplier")
    baseline = c_frame.filter(pl.col("industry_column_multiplier") == 1.0).row(0, named=True)
    baseline_factors = json.loads(baseline["factor_returns_json"])
    baseline_fitted = np.asarray(json.loads(baseline["fitted_returns_json"]), dtype=float)
    baseline_specific = np.asarray(json.loads(baseline["specific_returns_json"]), dtype=float)
    invariant_rows = []
    for row in c_frame.iter_rows(named=True):
        multiplier = float(row["industry_column_multiplier"])
        factors = json.loads(row["factor_returns_json"])
        factor_differences = [
            abs(
                (multiplier * factors[name] if name.startswith("risk_industry_") else factors[name])
                - baseline_factors[name]
            ) for name in baseline_factors
        ]
        fitted = np.asarray(json.loads(row["fitted_returns_json"]), dtype=float)
        specific = np.asarray(json.loads(row["specific_returns_json"]), dtype=float)
        invariant_rows.append({
            "trade_date": row["trade_date"],
            "industry_column_multiplier": multiplier,
            "raw_effective_condition_number": row["raw_effective_condition_number"],
            "equilibrated_effective_condition_number": row["equilibrated_effective_condition_number"],
            "scale_invariant_effective_condition_number": row[
                "scale_invariant_effective_condition_number"
            ],
            "maximum_rescaled_factor_return_difference": max(factor_differences),
            "maximum_fitted_return_difference": float(np.max(np.abs(fitted - baseline_fitted))),
            "maximum_specific_return_difference": float(np.max(np.abs(specific - baseline_specific))),
            "absolute_r_squared_difference": abs(float(row["r_squared"])-float(baseline["r_squared"])),
            "absolute_unweighted_r_squared_difference": abs(
                float(row["unweighted_r_squared"])-float(baseline["unweighted_r_squared"])
            ),
            "constraint_error": row["constraint_error"],
        })
    invariance = pl.DataFrame(invariant_rows).sort("industry_column_multiplier")
    high_precision = pl.DataFrame(high_results).sort("trade_date")
    limit = float(l1["derived_params"]["maximum_condition_number"])
    tail_summary = []
    for series_id, column in (
        ("raw_effective", "raw_effective_condition_number"),
        ("diagonally_equilibrated_effective", "equilibrated_effective_condition_number"),
        ("scale_invariant_canonical_effective", "scale_invariant_effective_condition_number"),
    ):
        values = tails[column].to_numpy()
        tail_summary.append({
            "series_id": series_id,
            "p50": float(np.quantile(values, 0.5)),
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "maximum": float(values.max()),
            "dates_above_frozen_limit": int((values > limit).sum()),
        })
    summary = pl.DataFrame(tail_summary)

    tolerance = float(
        protocol["parameterization_invariance_probe"]["invariant_absolute_tolerance"]
    )
    model_invariants_passed = all(
        float(invariance[column].max()) <= tolerance for column in (
            "maximum_rescaled_factor_return_difference",
            "maximum_fitted_return_difference", "maximum_specific_return_difference",
            "absolute_r_squared_difference", "absolute_unweighted_r_squared_difference",
            "constraint_error",
        )
    )
    raw_range = float(invariance["raw_effective_condition_number"].max()
                      / invariance["raw_effective_condition_number"].min())
    eq_range = float(invariance["equilibrated_effective_condition_number"].max()
                     / invariance["equilibrated_effective_condition_number"].min())
    canonical_range = float(
        invariance["scale_invariant_effective_condition_number"].max()
        / invariance["scale_invariant_effective_condition_number"].min()
    )
    diagnosis = {
        "schema_version": 1,
        "threshold_derivation_confirmed": True,
        "threshold_role": protocol["threshold_semantics"]["interpretation"],
        "raw_condition_parameterization_dependent": raw_range - 1.0 > float(
            protocol["parameterization_invariance_probe"][
                "raw_condition_parameterization_dependence_relative_tolerance"
            ]
        ),
        "raw_condition_c_probe_max_to_min_ratio": raw_range,
        "model_invariants_passed": model_invariants_passed,
        "equilibrated_condition_c_probe_max_to_min_ratio": eq_range,
        "equilibrated_condition_invariant_within_1e_10": eq_range - 1.0 <= 1e-10,
        "scale_invariant_condition_c_probe_max_to_min_ratio": canonical_range,
        "scale_invariant_condition_invariant_within_1e_10": canonical_range - 1.0 <= 1e-10,
        "high_precision_all_dates_passed": bool(high_precision["passed"].all()),
        "adoption_requirements_passed": bool(
            model_invariants_passed and canonical_range - 1.0 <= 1e-10
            and high_precision["passed"].all()
        ),
        "decision": "diagnostic_only_formal_solver_unchanged",
    }

    identity = {
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "l1_manifest_sha256": file_sha256(l1_manifest_path),
        "weight_artifact_sha256": file_sha256(weight_path),
        "g3_partition_inputs": input_records,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    run_id = f"g3_numerical_invariance_{json_hash(identity)[:16]}"
    run_dir = DATA / "diagnostics/g3_numerical_invariance" / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    products = {
        "full_history_condition_equilibration_v1": tails,
        "condition_tail_before_after_equilibration_v1": summary,
        "column_scale_invariance_probe_v1": invariance,
        "high_precision_solve_comparison_v1": high_precision,
    }
    outputs: dict[str, Any] = {}
    for name, frame in products.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        outputs[name] = {
            "path": str(path.relative_to(DATA)), "sha256": file_sha256(path),
            "rows": frame.height,
        }
    diagnosis_path = run_dir / "diagnostic_interpretation_v1.json"
    diagnosis_path.write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["diagnostic_interpretation_v1"] = {
        "path": str(diagnosis_path.relative_to(DATA)),
        "sha256": file_sha256(diagnosis_path), "rows": 1,
    }
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "g3_numerical_invariance_diagnostics", "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_class": protocol["diagnostic_class"],
        "return_values_read": True,
        "return_value_use": "three-date numerical solve accuracy and one median-date model-invariance test only; no factor selection or alpha evaluation",
        "identity": identity, "diagnosis": diagnosis,
        "publication": "diagnostic_only_G3_current_unchanged",
        "outputs": outputs,
    }
    manifest_path = run_dir / "_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
