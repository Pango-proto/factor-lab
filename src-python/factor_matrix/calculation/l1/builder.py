"""One-day risk exposure build. History orchestration and publication live elsewhere."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from ...canonical_definitions import (
    exposure_orthogonalization_weight, listing_age_calendar_days,
)
from ...factor_engine.contracts import FactorRole
from ...factor_engine.registry import FactorRegistry
from .descriptors import (
    index_membership_descriptors, liquidity_descriptor,
    market_sensitivity_descriptors,
)
from .style_math import transform_style_column, weighted_mean, weighted_sd
from .universe_variants import apply_universe_variant


STYLE_ORDER = (
    "size", "beta", "residual_volatility", "liquidity",
    "nonlinear_size", "listing_age",
)
ORTHOGONALITY_TOLERANCE = 1e-8
MOMENT_TOLERANCE = 1e-8
TRANSFORM_VERSION = "l1_role_specific_transform_v1"


@dataclass(frozen=True)
class L1DailyBuild:
    exposure: pl.DataFrame
    quality: pl.DataFrame
    checks: tuple[dict[str, Any], ...]


def _safe_column_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _one_hot(values: list[str], categories: list[str]) -> list[list[float]]:
    return [[1.0 if value == category else 0.0 for category in categories] for value in values]


def _impute_industry_median(
    frame: pl.DataFrame, column: str,
) -> tuple[list[float] | None, int, int, float, bool]:
    valid = frame.get_column(column).is_not_null() & frame.get_column(column).is_finite()
    n_valid = int(valid.sum())
    coverage = n_valid / frame.height if frame.height else 0.0
    medians = {
        row["sw_l1_code"]: row["median_value"]
        for row in frame.filter(valid).group_by("sw_l1_code").agg(
            pl.col(column).median().alias("median_value")
        ).to_dicts()
    }
    valid_values = frame.filter(valid).get_column(column).to_list()
    if not valid_values:
        return None, 0, frame.height, 0.0, True
    market_median = float(pl.Series(valid_values).median())
    used_fallback = False
    output: list[float] = []
    for row in frame.select("sw_l1_code", column).iter_rows(named=True):
        value = row[column]
        if value is not None and math.isfinite(float(value)):
            output.append(float(value))
            continue
        replacement = medians.get(row["sw_l1_code"])
        if replacement is None:
            replacement = market_median
            used_fallback = True
        output.append(float(replacement))
    return output, n_valid, frame.height - n_valid, coverage, used_fallback


def _quality_row(
    *, trade_date: date, factor_id: str, role: str, formula_version: str,
    n_valid: int, coverage_ratio: float, n_imputed: int = 0,
    mean_w: float | None = None, sd_w: float | None = None,
    max_corr: float | None = None, winsor_low: int = 0, winsor_high: int = 0,
    n_categories: int | None = None, min_category_members: int | None = None,
    used_fallback: bool = False,
    n_input: int | None = None, n_excluded: int = 0,
    status: str = "passed", failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "factor_id": factor_id,
        "family": "risk",
        "role": role,
        "n_valid": n_valid,
        "coverage_ratio": coverage_ratio,
        "n_imputed": n_imputed,
        "n_input": n_input if n_input is not None else n_valid + n_excluded,
        "n_excluded": n_excluded,
        "mean_w": mean_w,
        "sd_w": sd_w,
        "max_abs_corr_with_predecessors": max_corr,
        "n_winsorized_low": winsor_low,
        "n_winsorized_high": winsor_high,
        "n_categories": n_categories,
        "min_category_members": min_category_members,
        "used_fallback": used_fallback,
        "status": status,
        "failure_reason": failure_reason,
        "formula_version": formula_version,
        "transform_version": TRANSFORM_VERSION,
    }


def build_l1_day(
    *,
    trade_date: date,
    universe_day: pl.DataFrame,
    universe_history: pl.DataFrame,
    valuation: pl.DataFrame,
    returns: pl.DataFrame,
    board_benchmarks: pl.DataFrame,
    benchmark_membership: pl.DataFrame,
    registry: FactorRegistry,
    derived_params: dict[str, Any],
    new_listing_policy_path,
    risk_factor_set_id: str,
    metric_weights_by_asset: dict[str, float] | None = None,
    metric_weight_scheme_id: str = "sqrt_cap",
) -> L1DailyBuild:
    formal = apply_universe_variant(
        universe_day,
        universe_variant="frozen_d0",
        policy_path=new_listing_policy_path,
    ).filter("is_variant_tradable").sort("asset_id")
    if formal.is_empty():
        raise ValueError(f"L1_FORMAL_UNIVERSE_EMPTY date={trade_date}")
    required = {"asset_id", "board_id", "sw_l1_code", "sw_l2_code"}
    if not required <= set(formal.columns):
        raise ValueError("L1_FORMAL_UNIVERSE_METADATA_MISSING")
    formal_count = formal.height
    structure_complete = formal.filter(
        pl.col("sw_l1_code").is_not_null() & pl.col("board_id").is_not_null()
    )
    structure_excluded = formal_count - structure_complete.height

    cross_section = structure_complete.select(
        "trade_date", "asset_id", "board_id", "sw_l1_code", "sw_l2_code",
        "exchange_list_date", "days_since_exchange_list",
    ).join(
        valuation.filter(pl.col("trade_date") == trade_date).select(
            "asset_id", "float_mkt_cap"
        ),
        on="asset_id", how="left", validate="1:1",
    )
    valid_market_cap = (
        pl.col("float_mkt_cap").is_not_null()
        & pl.col("float_mkt_cap").is_finite()
        & (pl.col("float_mkt_cap") > 0)
    )
    before_market_cap = cross_section.height
    cross_section = cross_section.filter(valid_market_cap)
    market_cap_excluded = before_market_cap - cross_section.height
    if cross_section.is_empty():
        raise ValueError(f"L1_COMPLETE_CASE_UNIVERSE_EMPTY date={trade_date}")
    complete_case_coverage = cross_section.height / formal_count
    structure_coverage = structure_complete.height / formal_count
    if metric_weights_by_asset is None:
        weights = [
            exposure_orthogonalization_weight(value)
            for value in cross_section["float_mkt_cap"].to_list()
        ]
    else:
        asset_ids = cross_section["asset_id"].to_list()
        caps = cross_section["float_mkt_cap"].to_list()
        weights = [
            float(metric_weights_by_asset[asset_id])
            if asset_id in metric_weights_by_asset
            else exposure_orthogonalization_weight(cap)
            for asset_id, cap in zip(asset_ids, caps)
        ]
        checks_metric_fallback_rows = sum(
            asset_id not in metric_weights_by_asset for asset_id in asset_ids
        )
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("L1_METRIC_WEIGHT_INVALID")
    industries = sorted(cross_section["sw_l1_code"].unique().to_list())
    boards = sorted(cross_section["board_id"].unique().to_list())
    industry_values = cross_section["sw_l1_code"].to_list()
    board_values = cross_section["board_id"].to_list()
    industry_matrix = _one_hot(industry_values, industries)
    board_matrix = _one_hot(board_values, boards)
    if any(sum(row) != 1 for row in industry_matrix + board_matrix):
        raise RuntimeError("L1_STRUCTURE_ONE_HOT_ROW_SUM_FAILED")
    category_counts = cross_section.group_by("sw_l1_code").len()["len"].to_list()
    minimum_category_members = int(derived_params["minimum_category_members"])
    if min(category_counts) < minimum_category_members:
        raise RuntimeError("L1_MINIMUM_INDUSTRY_MEMBERS_FAILED")

    membership = index_membership_descriptors(
        as_of=trade_date, assets=cross_section, membership=benchmark_membership
    )
    cross_section = cross_section.join(membership, on="asset_id", how="left")
    market = market_sensitivity_descriptors(
        as_of=trade_date,
        assets=cross_section,
        universe_history=universe_history,
        returns=returns,
        board_benchmarks=board_benchmarks,
        lookback_days=120,
        minimum_observations=60,
    )
    liquidity = liquidity_descriptor(
        as_of=trade_date,
        assets=cross_section,
        valuation=valuation,
        lookback_days=20,
        minimum_observations=12,
    )
    cross_section = cross_section.join(market, on="asset_id", how="left").join(
        liquidity, on="asset_id", how="left"
    ).with_columns(
        pl.col("float_mkt_cap").log().alias("size_raw"),
        pl.struct("trade_date", "exchange_list_date").map_elements(
            lambda row: listing_age_calendar_days(
                row["trade_date"], row["exchange_list_date"]
            ),
            return_dtype=pl.Int64,
        ).cast(pl.Float64).log1p().alias("listing_age_raw"),
    )

    exposure_columns: dict[str, list[float] | list[bool]] = {
        "risk_country": [1.0] * cross_section.height,
        **{
            f"risk_industry_{_safe_column_name(category)}": [row[index] for row in industry_matrix]
            for index, category in enumerate(industries)
        },
        **{
            f"risk_board_{_safe_column_name(category)}": [row[index] for row in board_matrix]
            for index, category in enumerate(boards)
        },
        "risk_index_CSI300": cross_section["index_CSI300"].cast(pl.Float64).to_list(),
        "risk_index_CSI500": cross_section["index_CSI500"].cast(pl.Float64).to_list(),
    }
    quality: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    if metric_weights_by_asset is not None:
        checks.append({
            "check_id": "metric_weight_row_fallback",
            "passed": True,
            "observed": checks_metric_fallback_rows,
        })
    structure = (
        ("country", FactorRole.COUNTRY.value, [cross_section.height]),
        ("industry_sw1", FactorRole.INDUSTRY.value, category_counts),
        ("board", FactorRole.BOARD.value,
         cross_section.group_by("board_id").len()["len"].to_list()),
        ("index_membership", FactorRole.MEMBERSHIP.value, [cross_section.height]),
    )
    for factor_id, role, counts in structure:
        spec = registry.get(factor_id).spec
        quality.append(_quality_row(
            trade_date=trade_date, factor_id=factor_id, role=role,
            formula_version=f"{spec.version}:{spec.code_sha[:12]}",
            n_valid=cross_section.height,
            coverage_ratio=(structure_coverage if factor_id == "industry_sw1"
                            else complete_case_coverage),
            n_input=formal_count,
            n_excluded=(structure_excluded if factor_id == "industry_sw1"
                        else formal_count - cross_section.height),
            n_categories=(
                len(industries) if factor_id == "industry_sw1" else
                len(boards) if factor_id == "board" else
                2 if factor_id == "index_membership" else 1
            ),
            min_category_members=min(counts),
        ))

    coverage_min = float(derived_params["coverage_min"])
    quality.append(_quality_row(
        trade_date=trade_date,
        factor_id="model_input_complete_case",
        role="model_input",
        formula_version="l1_complete_case_v1",
        n_valid=cross_section.height,
        coverage_ratio=complete_case_coverage,
        n_input=formal_count,
        n_excluded=structure_excluded + market_cap_excluded,
        status="passed" if complete_case_coverage >= coverage_min else "invalid",
        failure_reason=(None if complete_case_coverage >= coverage_min
                        else f"coverage_below_{coverage_min}"),
    ))

    raw_columns = {
        "size": "size_raw",
        "beta": "beta_raw",
        "residual_volatility": "residual_volatility_raw",
        "liquidity": "liquidity_raw",
        "listing_age": "listing_age_raw",
    }
    transformed: dict[str, tuple[float, ...]] = {}
    standardized_size: tuple[float, ...] | None = None
    mad_k = float(derived_params["mad_k"])
    raw_cache = {
        factor_id: _impute_industry_median(cross_section, column)
        for factor_id, column in raw_columns.items()
    }
    invalid_coverage = {
        factor_id: values[3]
        for factor_id, values in raw_cache.items()
        if values[3] < coverage_min
    }
    if complete_case_coverage < coverage_min:
        invalid_coverage["model_input_complete_case"] = complete_case_coverage
    if invalid_coverage:
        for factor_id in STYLE_ORDER:
            spec = registry.get(factor_id).spec
            if factor_id == "nonlinear_size":
                n_valid, n_imputed, coverage, used_fallback = (
                    cross_section.height, 0, 1.0, False
                )
                reason = "upstream_style_coverage_invalid"
            else:
                _, n_valid, n_imputed, coverage, used_fallback = raw_cache[factor_id]
                reason = (
                    f"coverage_below_{coverage_min}"
                    if coverage < coverage_min else "date_invalid_due_to_other_factor"
                )
            quality.append(_quality_row(
                trade_date=trade_date, factor_id=factor_id,
                role=FactorRole.STYLE_RISK.value,
                formula_version=f"{spec.version}:{spec.code_sha[:12]}",
                n_valid=n_valid, coverage_ratio=coverage, n_imputed=n_imputed,
                used_fallback=used_fallback, status="invalid", failure_reason=reason,
            ))
            exposure_columns[f"risk_{factor_id}"] = [None] * cross_section.height
        exposure = cross_section.select(
            "trade_date", "asset_id", "board_id", "sw_l1_code", "sw_l2_code",
        ).with_columns(
            pl.lit("frozen_d0").alias("universe_variant"),
            pl.lit(risk_factor_set_id).alias("risk_factor_set_id"),
            pl.lit("candidate_v1").alias("risk_factor_set_version"),
            pl.lit(False).alias("is_valid"),
            pl.lit(metric_weight_scheme_id).alias("orthogonalization_weight_scheme_id"),
            *[pl.Series(name, values) for name, values in exposure_columns.items()],
        )
        checks.append({
            "check_id": "style_coverage",
            "passed": False,
            "observed": invalid_coverage,
        })
        return L1DailyBuild(
            exposure=exposure, quality=pl.DataFrame(quality), checks=tuple(checks)
        )
    for factor_id in STYLE_ORDER:
        spec = registry.get(factor_id).spec
        predecessors = tuple(spec.orthogonalize_after)
        predecessor_columns = [transformed[item] for item in predecessors]
        control_columns = [
            tuple(row[index] for row in industry_matrix)
            for index in range(len(industries))
        ] + predecessor_columns
        controls = [list(row) for row in zip(*control_columns)]
        if factor_id == "nonlinear_size":
            if standardized_size is None:
                raise RuntimeError("L1_NONLINEAR_SIZE_ORDER_INVALID")
            raw = [value ** 3 for value in standardized_size]
            n_valid, n_imputed, coverage, used_fallback = (
                cross_section.height, 0, 1.0, False
            )
            result = transform_style_column(
                raw,
                controls=controls,
                control_columns=control_columns,
                weights=weights,
                mad_k=mad_k,
                already_standardized=True,
            )
            if abs(result.pre_scale_mean_w) > MOMENT_TOLERANCE:
                raise RuntimeError("L1_NONLINEAR_PRE_SCALE_MEAN_FAILED")
            step1_sd = weighted_sd(standardized_size, weights)
            step2_sd = weighted_sd(raw, weights)
            if abs(step1_sd - 1.0) > MOMENT_TOLERANCE or step2_sd <= 1.0:
                raise RuntimeError("L1_NONLINEAR_INTERMEDIATE_ORDER_FAILED")
            checks.extend([
                {"check_id": "nonlinear_size.step1_sd", "passed": True,
                 "observed": step1_sd},
                {"check_id": "nonlinear_size.step2_cubed_sd_gt_one", "passed": True,
                 "observed": step2_sd},
                {"check_id": "nonlinear_size.step3_pre_scale_mean", "passed": True,
                 "observed": result.pre_scale_mean_w},
                {"check_id": "nonlinear_size.step4_post_scale_mean", "passed": True,
                 "observed": result.mean_w},
            ])
        else:
            raw, n_valid, n_imputed, coverage, used_fallback = raw_cache[factor_id]
            if raw is None:
                raise RuntimeError(f"L1_STYLE_RAW_UNEXPECTEDLY_EMPTY factor={factor_id}")
            try:
                result = transform_style_column(
                    raw,
                    controls=controls,
                    control_columns=control_columns,
                    weights=weights,
                    mad_k=mad_k,
                )
            except (ValueError, RuntimeError) as exc:
                raise type(exc)(f"{exc} factor={factor_id}") from exc
            if factor_id == "size":
                standardized_size = result.standardized_raw
        if abs(result.mean_w) > MOMENT_TOLERANCE:
            raise RuntimeError(f"L1_STYLE_WEIGHTED_MEAN_FAILED factor={factor_id}")
        if abs(result.sd_w - 1.0) > MOMENT_TOLERANCE:
            raise RuntimeError(f"L1_STYLE_WEIGHTED_SD_FAILED factor={factor_id}")
        if result.max_abs_corr_with_controls > ORTHOGONALITY_TOLERANCE:
            raise RuntimeError(f"L1_STYLE_ORTHOGONALITY_FAILED factor={factor_id}")
        transformed[factor_id] = result.values
        exposure_columns[f"risk_{factor_id}"] = list(result.values)
        quality.append(_quality_row(
            trade_date=trade_date, factor_id=factor_id,
            role=FactorRole.STYLE_RISK.value,
            formula_version=f"{spec.version}:{spec.code_sha[:12]}",
            n_valid=n_valid, coverage_ratio=coverage, n_imputed=n_imputed,
            mean_w=result.mean_w, sd_w=result.sd_w,
            max_corr=result.max_abs_corr_with_controls,
            winsor_low=result.winsorized_low,
            winsor_high=result.winsorized_high,
            used_fallback=used_fallback,
        ))
        checks.extend([
            {"check_id": f"{factor_id}.mean_w", "passed": True,
             "observed": result.mean_w},
            {"check_id": f"{factor_id}.sd_w", "passed": True,
             "observed": result.sd_w},
            {"check_id": f"{factor_id}.orthogonality", "passed": True,
             "observed": result.max_abs_corr_with_controls},
        ])

    exposure = cross_section.select(
        "trade_date", "asset_id", "board_id", "sw_l1_code", "sw_l2_code",
    ).with_columns(
        pl.lit("frozen_d0").alias("universe_variant"),
        pl.lit(risk_factor_set_id).alias("risk_factor_set_id"),
        pl.lit("candidate_v1").alias("risk_factor_set_version"),
        pl.lit(True).alias("is_valid"),
        pl.lit(metric_weight_scheme_id).alias("orthogonalization_weight_scheme_id"),
        *[pl.Series(name, values) for name, values in exposure_columns.items()],
    )
    checks.extend([
        {"check_id": "country_all_one", "passed": True, "observed": 1.0},
        {"check_id": "industry_one_hot", "passed": True,
         "observed": len(industries)},
        {"check_id": "board_one_hot", "passed": True, "observed": len(boards)},
        {"check_id": "minimum_industry_members", "passed": True,
         "observed": min(category_counts)},
    ])
    return L1DailyBuild(
        exposure=exposure,
        quality=pl.DataFrame(quality),
        checks=tuple(checks),
    )
