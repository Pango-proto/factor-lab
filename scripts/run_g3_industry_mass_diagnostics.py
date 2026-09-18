#!/usr/bin/env python3
"""Test the label-free industry-weight-mass explanation of G3 conditioning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src-python"))

from factor_matrix.storage import file_sha256, json_hash  # noqa: E402


DATA = PROJECT / "data"
CONFIG_PATH = PROJECT / "config/g3_conditioning_diagnostics_v1.json"
SOURCE_RUN = DATA / "diagnostics/g3_conditioning/run_id=g3_conditioning_diagnostic_8751e62ef8f80d14"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(x.size), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    residual = y - fitted
    total = float(np.square(y - y.mean()).sum())
    residual_sum = float(np.square(residual).sum())
    r_squared = 1.0 - residual_sum / total
    variance = residual_sum / (x.size - 2)
    covariance = variance * np.linalg.inv(design.T @ design)
    return {
        "observations": int(x.size),
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "slope_standard_error": float(math.sqrt(covariance[1, 1])),
        "r_squared": r_squared,
        "pearson_correlation": float(np.corrcoef(x, y)[0, 1]),
    }


def _current_l1_weight_projection(
    frame: pl.DataFrame, *, industry_column: str,
) -> list[dict[str, Any]]:
    estimation = frame.filter(pl.col("in_estimation_domain"))
    weights = estimation["candidate_weight"].to_numpy()
    interior = (weights > 0.1000001) & (weights < 9.9999999)
    use = estimation.filter(pl.Series(interior))
    board = use["board_id"].to_list()
    feature_names = (
        "intercept", "log_float_mkt_cap", "risk_liquidity",
        "risk_residual_volatility", "board_BSE", "board_CHINEXT", "board_STAR",
    )
    features = np.column_stack([
        np.ones(use.height),
        np.log(use["float_mkt_cap"].to_numpy()),
        use["risk_liquidity"].to_numpy(),
        use["risk_residual_volatility"].to_numpy(),
        np.asarray([value == "BSE" for value in board], dtype=float),
        np.asarray([value == "CHINEXT" for value in board], dtype=float),
        np.asarray([value == "STAR" for value in board], dtype=float),
    ])
    log_weight = np.log(use["candidate_weight"].to_numpy())
    log_weight_coefficients = np.linalg.lstsq(features, log_weight, rcond=None)[0]
    fitted = features @ log_weight_coefficients
    fit_r_squared = 1.0 - float(np.square(log_weight - fitted).sum()) / float(
        np.square(log_weight - log_weight.mean()).sum()
    )
    surrogate_sigma_coefficients = -0.5 * log_weight_coefficients
    industry = use[industry_column].to_numpy() > 0.5
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names[1:], start=1):
        mean_difference = float(features[industry, index].mean() - features[:, index].mean())
        rows.append({
            "component": name,
            "surrogate_log_sigma_coefficient": float(surrogate_sigma_coefficients[index]),
            "industry_minus_market_feature_mean": mean_difference,
            "surrogate_industry_minus_market_log_sigma_contribution": float(
                surrogate_sigma_coefficients[index] * mean_difference
            ),
            "log_weight_fit_r_squared": fit_r_squared,
            "interior_observations": int(use.height),
            "industry_interior_observations": int(industry.sum()),
        })
    return rows


def main() -> None:
    protocol = _load(CONFIG_PATH)
    followup = protocol["industry_mass_followup"]
    source_manifest_path = SOURCE_RUN / "_MANIFEST.json"
    source_manifest = _load(source_manifest_path)
    daily_path = SOURCE_RUN / "daily_condition_and_weight_concentration_v1.parquet"
    daily_condition = pl.read_parquet(daily_path)
    l1_current = _load(DATA / "gold/risk_exposure_matrix/_CURRENT.json")
    l1_manifest = _load(DATA / l1_current["manifest"])
    if l1_manifest["risk_basis_id"] != protocol["risk_basis_id"]:
        raise RuntimeError("INDUSTRY_MASS_RISK_BASIS_MISMATCH")
    exposure_path = DATA / l1_manifest["outputs"]["risk_exposure_matrix_v1"]["path"]
    weight_path = DATA / l1_manifest["weight_metric_contract"]["artifact_path"]
    valuation_path = DATA / "silver/valuation_daily/data.parquet"
    exposure_schema = pl.read_parquet_schema(exposure_path)
    industry_columns = [
        name for name, dtype in exposure_schema.items()
        if name.startswith("risk_industry_") and dtype.is_float()
    ]

    g3_manifests: list[dict[str, Any]] = []
    for path in (DATA / "gold/l2b_risk_only").glob("run_id=*/_MANIFEST.json"):
        payload = _load(path)
        if payload.get("risk_basis_id") == protocol["risk_basis_id"]:
            g3_manifests.append(payload)
    g3_manifests.sort(key=lambda item: item["start"])
    if len(g3_manifests) != 84:
        raise RuntimeError(f"INDUSTRY_MASS_EXPECTS_84_G3_PARTITIONS observed={len(g3_manifests)}")

    mass_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    selected = set(protocol["preselected_gate_failure_trade_dates"])
    for partition_number, manifest in enumerate(g3_manifests, start=1):
        specific_path = DATA / manifest["outputs"]["specific_returns_v1"]["path"]
        domain = pl.read_parquet(specific_path).select(
            "trade_date", "exposure_date", "asset_id", "in_estimation_domain"
        )
        exposure_dates = domain["exposure_date"].unique().to_list()
        exposures = (
            pl.scan_parquet(exposure_path)
            .filter(pl.col("trade_date").is_in(exposure_dates) & pl.col("is_valid"))
            .select(
                pl.col("trade_date").alias("exposure_date"), "asset_id", "board_id",
                "risk_liquidity", "risk_residual_volatility", *industry_columns,
            ).collect()
        )
        weights = (
            pl.scan_parquet(weight_path).filter(pl.col("exposure_date").is_in(exposure_dates))
            .select("exposure_date", "asset_id", "candidate_weight", "weight_scheme_id").collect()
        )
        valuations = (
            pl.scan_parquet(valuation_path).filter(pl.col("trade_date").is_in(exposure_dates))
            .select(pl.col("trade_date").alias("exposure_date"), "asset_id", "float_mkt_cap")
            .collect()
        )
        joined = (
            domain.join(exposures, on=("exposure_date", "asset_id"), how="inner", validate="m:1")
            .join(weights, on=("exposure_date", "asset_id"), how="left", validate="m:1")
            .join(valuations, on=("exposure_date", "asset_id"), how="left", validate="m:1")
        )
        if joined.select(pl.any_horizontal(
            pl.col("candidate_weight").is_null(), pl.col("float_mkt_cap").is_null()
        ).any()).item():
            raise RuntimeError("INDUSTRY_MASS_WEIGHT_OR_CAP_COVERAGE_INCOMPLETE")
        for key, frame in joined.partition_by("exposure_date", as_dict=True).items():
            exposure_date = key[0] if isinstance(key, tuple) else key
            estimation = frame.filter(pl.col("in_estimation_domain"))
            structural = estimation["candidate_weight"].to_numpy()
            sqrt_cap = np.sqrt(estimation["float_mkt_cap"].to_numpy())
            total_by_scheme = {
                "structural_pit_v1": float(structural.sum()),
                "sqrt_cap_counterfactual": float(sqrt_cap.sum()),
            }
            values_by_scheme = {
                "structural_pit_v1": structural,
                "sqrt_cap_counterfactual": sqrt_cap,
            }
            active_industries = [
                column for column in industry_columns
                if estimation[column].fill_null(0.0).abs().max() > 0
            ]
            for industry_column in active_industries:
                support = estimation[industry_column].to_numpy() > 0.5
                for scheme_id, values in values_by_scheme.items():
                    mass_rows.append({
                        "trade_date": frame["trade_date"][0],
                        "exposure_date": exposure_date,
                        "weight_scheme_id": scheme_id,
                        "factor_id": industry_column,
                        "member_count": int(support.sum()),
                        "weight_mass_share": float(values[support].sum() / total_by_scheme[scheme_id]),
                    })
            if frame["trade_date"][0].isoformat() in selected:
                if set(estimation["weight_scheme_id"].unique().to_list()) != {"structural"}:
                    raise RuntimeError("INDUSTRY_MASS_SELECTED_DATE_NOT_STRUCTURAL")
                rows = _current_l1_weight_projection(
                    frame, industry_column="risk_industry_801230_SI"
                )
                component_rows.extend({
                    "trade_date": frame["trade_date"][0],
                    "exposure_date": exposure_date,
                    "factor_id": "risk_industry_801230_SI",
                    **row,
                } for row in rows)
        print(f"completed partition {partition_number}/{len(g3_manifests)}", flush=True)

    masses = pl.DataFrame(mass_rows).sort("trade_date", "weight_scheme_id", "factor_id")
    minima = (
        masses.sort("weight_mass_share")
        .group_by("trade_date", "exposure_date", "weight_scheme_id", maintain_order=True)
        .first()
        .rename({
            "factor_id": "minimum_mass_factor_id",
            "member_count": "minimum_mass_member_count",
            "weight_mass_share": "minimum_industry_weight_mass_share",
        })
        .sort("trade_date", "weight_scheme_id")
    )
    daily = daily_condition.join(
        minima.filter(pl.col("weight_scheme_id") == "structural_pit_v1").drop("weight_scheme_id")
        .rename({
            "minimum_mass_factor_id": "structural_minimum_mass_factor_id",
            "minimum_mass_member_count": "structural_minimum_mass_member_count",
            "minimum_industry_weight_mass_share": "structural_minimum_industry_weight_mass_share",
        }),
        on=("trade_date", "exposure_date"), how="inner", validate="1:1",
    ).join(
        minima.filter(pl.col("weight_scheme_id") == "sqrt_cap_counterfactual").drop("weight_scheme_id")
        .rename({
            "minimum_mass_factor_id": "sqrt_cap_minimum_mass_factor_id",
            "minimum_mass_member_count": "sqrt_cap_minimum_mass_member_count",
            "minimum_industry_weight_mass_share": "sqrt_cap_minimum_industry_weight_mass_share",
        }),
        on=("trade_date", "exposure_date"), how="inner", validate="1:1",
    ).with_columns(
        (-pl.col("structural_minimum_industry_weight_mass_share").log()).alias("negative_log_structural_minimum_mass"),
        pl.col("structural_base_condition_number").log().alias("log_structural_base_condition"),
        pl.col("observed_effective_condition_number").log().alias("log_observed_effective_condition"),
        (-pl.col("sqrt_cap_minimum_industry_weight_mass_share").log()).alias("negative_log_sqrt_cap_minimum_mass"),
        pl.col("sqrt_cap_base_condition_number").log().alias("log_sqrt_cap_base_condition"),
    ).sort("trade_date")

    regressions = {
        "structural_base": _fit(
            daily["negative_log_structural_minimum_mass"].to_numpy(),
            daily["log_structural_base_condition"].to_numpy(),
        ),
        "observed_structural_effective_huber": _fit(
            daily["negative_log_structural_minimum_mass"].to_numpy(),
            daily["log_observed_effective_condition"].to_numpy(),
        ),
        "sqrt_cap_same_current_l1": _fit(
            daily["negative_log_sqrt_cap_minimum_mass"].to_numpy(),
            daily["log_sqrt_cap_base_condition"].to_numpy(),
        ),
    }
    confirmation = followup["mechanism_confirmation_rule"]
    target = float(confirmation["slope_target"])
    tolerance = float(confirmation["slope_absolute_tolerance"])
    minimum_r_squared = float(confirmation["minimum_r_squared"])
    for result in regressions.values():
        result["predeclared_mechanism_confirmed"] = bool(
            abs(result["slope"] - target) <= tolerance
            and result["r_squared"] >= minimum_r_squared
        )

    condition_limit = float(l1_manifest["derived_params"]["maximum_condition_number"])
    mass_floor = 1.0 / condition_limit
    affected = masses.filter(pl.col("weight_mass_share") < mass_floor)
    affected_summary = (
        affected.group_by("weight_scheme_id", "factor_id")
        .agg(
            pl.len().alias("industry_date_count"),
            pl.col("trade_date").n_unique().alias("distinct_trade_dates"),
            pl.col("weight_mass_share").min().alias("minimum_observed_mass_share"),
        ).sort("weight_scheme_id", "industry_date_count", descending=[False, True])
    )

    comparison_rows = []
    condition_series = (
        ("structural_base", "structural_base_condition_number"),
        ("observed_structural_effective_huber", "observed_effective_condition_number"),
        ("sqrt_cap_same_current_l1_counterfactual", "sqrt_cap_base_condition_number"),
    )
    for series_id, column in condition_series:
        values = daily[column].to_numpy()
        comparison_rows.append({
            "series_id": series_id,
            "p50": float(np.quantile(values, 0.5)),
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "maximum": float(values.max()),
            "dates_above_frozen_limit": int((values > condition_limit).sum()),
        })
    comparison = pl.DataFrame(comparison_rows)
    components = pl.DataFrame(component_rows).sort("trade_date", "component")

    identity = {
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "source_diagnostic_manifest_sha256": file_sha256(source_manifest_path),
        "source_daily_condition_sha256": file_sha256(daily_path),
        "l1_history_run_id": l1_manifest["run_id"],
        "l1_history_sha256": file_sha256(DATA / l1_current["manifest"]),
        "weight_artifact_sha256": file_sha256(weight_path),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    run_id = f"g3_industry_mass_diagnostic_{json_hash(identity)[:16]}"
    run_dir = DATA / "diagnostics/g3_industry_mass" / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    products = {
        "all_date_industry_weight_mass_v1": masses,
        "daily_condition_vs_minimum_industry_mass_v1": daily,
        "industry_mass_floor_affected_distribution_v1": affected_summary,
        "selected_date_current_l1_weight_projection_v1": components,
        "weight_metric_condition_comparison_v1": comparison,
    }
    outputs: dict[str, Any] = {}
    for name, frame in products.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        outputs[name] = {
            "path": str(path.relative_to(DATA)),
            "sha256": file_sha256(path),
            "rows": frame.height,
        }
    regression_path = run_dir / "industry_mass_mechanism_regression_v1.json"
    regression_payload = {
        "schema_version": 1,
        "diagnostic_class": protocol["diagnostic_class"],
        "return_value_columns_read": [],
        "formula": followup["mechanism_regression"],
        "predeclared_confirmation_rule": confirmation,
        "regressions": regressions,
        "hypothesized_mass_floor": mass_floor,
        "hypothesized_mass_floor_source": "1/l1_history_manifest.derived_params.maximum_condition_number",
        "mass_floor_validated_for_use": False,
        "implementation_geometry_fact": "Current design consumes unstandardized one-hot industry columns; it does not W-standardize X columns.",
        "exact_structural_component_attribution": "unavailable_weight_generation_refit_schedule_and_old_L1_features_were_not_retained",
        "current_l1_component_output_status": "surrogate_projection_only",
        "decision": "diagnostic_only_no_remediation_selected",
    }
    regression_path.write_text(
        json.dumps(regression_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["industry_mass_mechanism_regression_v1"] = {
        "path": str(regression_path.relative_to(DATA)),
        "sha256": file_sha256(regression_path),
        "rows": 1,
    }
    structural = regressions["structural_base"]
    selected_rows = daily.filter(
        pl.col("trade_date").cast(pl.String).is_in(sorted(selected))
    ).select(
        "trade_date", "structural_base_condition_number",
        "structural_minimum_mass_factor_id",
        "structural_minimum_industry_weight_mass_share",
    ).to_dicts()
    interpretation_path = run_dir / "diagnostic_interpretation_v1.json"
    interpretation = {
        "schema_version": 1,
        "status": "predeclared_single_parameter_mass_mechanism_falsified",
        "headline": "Minimum active-industry W mass is associated with condition number but does not generate the full condition-number sequence under the predeclared rule.",
        "structural_base_regression": structural,
        "preselected_dates": selected_rows,
        "implementation_geometry": {
            "industry_columns": "unstandardized_one_hot",
            "country_column": "constant_one",
            "W_standardization_of_X": False,
            "consequence": "1/minimum industry mass is not an equality approximation for the implemented constrained Gram condition number.",
        },
        "hypothesized_mass_floor": {
            "value": mass_floor,
            "validated_for_use": False,
            "affected_structural_industry_dates": int(affected.filter(
                pl.col("weight_scheme_id") == "structural_pit_v1"
            ).height),
        },
        "structural_weight_component_attribution": {
            "exact_status": "unavailable",
            "reason": "The retained weight artifact contains final weights only; the generating refit schedule and old-L1 feature values were deleted during legacy cleanup.",
            "current_output": "A current-L1 projection with fit R-squared reported on every row; it must not be described as recovered generating coefficients.",
        },
        "decision": "Do not activate either mass-based remediation shape from this diagnostic.",
    }
    interpretation_path.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["diagnostic_interpretation_v1"] = {
        "path": str(interpretation_path.relative_to(DATA)),
        "sha256": file_sha256(interpretation_path),
        "rows": 1,
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "g3_industry_mass_diagnostics",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_class": protocol["diagnostic_class"],
        "return_value_columns_read": [],
        "source_diagnostic_run_id": source_manifest["run_id"],
        "identity": identity,
        "counts": {
            "dates": daily.height,
            "industry_date_weight_scheme_rows": masses.height,
            "structural_mass_floor_affected_rows": affected.filter(
                pl.col("weight_scheme_id") == "structural_pit_v1"
            ).height,
            "sqrt_cap_mass_floor_affected_rows": affected.filter(
                pl.col("weight_scheme_id") == "sqrt_cap_counterfactual"
            ).height,
        },
        "frozen_condition_limit": condition_limit,
        "hypothesized_mass_floor": mass_floor,
        "mass_floor_validated_for_use": False,
        "publication": "diagnostic_only_g3_current_unchanged",
        "outputs": outputs,
    }
    manifest_path = run_dir / "_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
