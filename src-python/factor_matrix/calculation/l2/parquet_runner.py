from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from .contracts import RegressionMode, ReturnDecompositionConfig
from .design_matrix import (
    build_daily_categorical_constrained_design,
    build_daily_industry_constrained_design,
)
from .g3_runner import _active_factor_columns
from .return_decomposition import cross_section_statistics, decompose_cross_section
from .label_policy import apply_l2b_label_policy, load_return_label_policy


# Legacy four-file adapter; remove after all non-canonical callers are retired.
@dataclass(frozen=True)
class L2BParquetInputs:
    exposure_path: Path
    tradable_universe_path: Path
    realized_returns_path: Path
    valuation_path: Path
    return_prices_path: Path | None = None

    def paths(self) -> dict[str, Path]:
        paths = {
            "exposure_matrix_v1": self.exposure_path,
            "tradable_universe_v1": self.tradable_universe_path,
            "silver.returns_daily": self.realized_returns_path,
            "silver.valuation_daily": self.valuation_path,
        }
        if self.return_prices_path is not None:
            paths["silver.prices_daily"] = self.return_prices_path
        return paths


def _next_date_mapping(exposure_dates: Sequence[date], return_dates: Sequence[date]) -> pl.DataFrame:
    ordered_returns = sorted(set(return_dates))
    rows = []
    for exposure_date in sorted(set(exposure_dates)):
        position = bisect_right(ordered_returns, exposure_date)
        if position < len(ordered_returns):
            rows.append({"trade_date": exposure_date, "return_date": ordered_returns[position]})
    if not rows:
        raise ValueError("L2B_NO_NEXT_DAY_RETURN_LABELS")
    return pl.DataFrame(rows)


def run_l2b_from_parquet(
    inputs: L2BParquetInputs,
    *,
    exposure_columns: Sequence[str],
    family_by_column: Mapping[str, str],
    mode: RegressionMode,
    config: ReturnDecompositionConfig,
    base_weight_path: Path | None = None,
    categorical_prefixes: Mapping[str, str] | None = None,
    minimum_category_members: int = 5,
) -> dict[str, pl.DataFrame]:
    """Stateless adapter using the canonical design and optional frozen weights.

    ``categorical_prefixes`` enables the production G3 contract.  In that mode
    the adapter must receive the same one-hot risk columns, price labels, and
    frozen base weights as ``g3_runner``; the dynamic categorical selection and
    estimation-domain rules are then delegated to the same G3 helper.
    """
    if categorical_prefixes is not None and inputs.return_prices_path is None:
        raise ValueError("L2B_CANONICAL_G3_PRICES_REQUIRED")
    for path in inputs.paths().values():
        if not path.exists():
            raise FileNotFoundError(path)
    exposure = pl.read_parquet(inputs.exposure_path)
    if base_weight_path is not None:
        if not base_weight_path.exists():
            raise FileNotFoundError(base_weight_path)
        weights = pl.read_parquet(base_weight_path).select(
            pl.col("exposure_date").alias("trade_date"), "asset_id", "candidate_weight"
        )
        exposure = exposure.join(
            weights, on=("trade_date", "asset_id"), how="left", validate="1:1"
        )
    universe = pl.read_parquet(inputs.tradable_universe_path)
    returns = pl.read_parquet(inputs.realized_returns_path).select(
        "trade_date", "asset_id", "total_return", "return_source"
    )
    return_prices = None
    if inputs.return_prices_path is not None:
        return_prices = pl.read_parquet(inputs.return_prices_path).select(
            "trade_date", "asset_id", "raw_open", "raw_high", "raw_low", "raw_close",
            "limit_up", "limit_down",
        )
    valuation = pl.read_parquet(inputs.valuation_path).select(
        "trade_date", "asset_id", "float_mkt_cap"
    )
    required_exposure = {"trade_date", "asset_id", *exposure_columns}
    if not required_exposure <= set(exposure.columns):
        missing = sorted(required_exposure - set(exposure.columns))
        raise ValueError(f"L2B_EXPOSURE_COLUMNS_MISSING columns={','.join(missing)}")
    required_universe = {
        "trade_date", "asset_id", "is_tradable", "board_id", "sw_l1_code",
        "exchange_list_date",
    }
    if not required_universe <= set(universe.columns):
        missing = sorted(required_universe - set(universe.columns))
        raise ValueError(f"L2B_UNIVERSE_COLUMNS_MISSING columns={','.join(missing)}")
    date_map = _next_date_mapping(
        exposure.get_column("trade_date").to_list(), returns.get_column("trade_date").to_list()
    )
    exposure_select = ["trade_date", "asset_id", *exposure_columns]
    if categorical_prefixes is not None:
        exposure_select.append("is_valid")
    join_how = "left" if categorical_prefixes is not None else "inner"
    joined = (
        exposure.select(*exposure_select, *(
            ["candidate_weight"] if base_weight_path is not None else []
        ))
        .join(date_map, on="trade_date", how="inner", validate="m:1")
        .join(
            returns.rename({"trade_date": "return_date"}),
            on=("return_date", "asset_id"), how=join_how, validate="m:1",
        )
    )
    if return_prices is not None:
        joined = joined.join(
            return_prices.rename({"trade_date": "return_date"}),
            on=("return_date", "asset_id"), how="left", validate="m:1",
        )
    joined = (
        joined
        .join(
            universe.select(
                "trade_date", "asset_id", "is_tradable", "board_id", "sw_l1_code",
                "exchange_list_date",
            ),
            on=("trade_date", "asset_id"), how=join_how, validate="m:1",
        )
        .join(valuation, on=("trade_date", "asset_id"), how=join_how, validate="m:1")
        .rename({"sw_l1_code": "industry_id"})
    )
    joined = (
        apply_l2b_label_policy(
            joined, load_return_label_policy(), label_date_column="return_date"
        )
        .rename({"total_return": "realized_return"})
    )
    if categorical_prefixes is None:
        joined = joined.filter("label_eligible")
    else:
        joined = joined.with_columns(
            (pl.col("is_valid") & pl.col("label_eligible")).alias("model_eligible"),
            (
                (
                    pl.col("limit_up").is_not_null()
                    & (pl.col("raw_open") >= pl.col("limit_up"))
                    & (pl.col("raw_high") >= pl.col("limit_up"))
                    & (pl.col("raw_low") >= pl.col("limit_up"))
                    & (pl.col("raw_close") >= pl.col("limit_up"))
                )
                | (
                    pl.col("limit_down").is_not_null()
                    & (pl.col("raw_open") <= pl.col("limit_down"))
                    & (pl.col("raw_high") <= pl.col("limit_down"))
                    & (pl.col("raw_low") <= pl.col("limit_down"))
                    & (pl.col("raw_close") <= pl.col("limit_down"))
                )
            ).fill_null(False).alias("is_limit_locked"),
        ).with_columns(
            (pl.col("model_eligible") & ~pl.col("is_limit_locked")).alias(
                "in_estimation_domain"
            )
        )
    if joined.is_empty():
        raise ValueError("L2B_REAL_DATA_JOIN_EMPTY")

    factor_rows: list[dict[str, object]] = []
    specific_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    for key, daily in joined.partition_by("trade_date", as_dict=True).items():
        exposure_date = key[0] if isinstance(key, tuple) else key
        return_date = daily.get_column("return_date")[0]
        if categorical_prefixes is None:
            design = build_daily_industry_constrained_design(
                daily,
                exposure_columns=exposure_columns,
                family_by_column=family_by_column,
                base_weight_column=("candidate_weight" if base_weight_path is not None else None),
            )
        else:
            factor_columns, blocks = _active_factor_columns(
                daily, list(exposure_columns)
            )
            expected_blocks = {
                block_id: tuple(
                    column for column in factor_columns
                    if column.startswith(prefix)
                )
                for block_id, prefix in categorical_prefixes.items()
            }
            design = build_daily_categorical_constrained_design(
                daily,
                exposure_columns=factor_columns,
                family_by_column={column: family_by_column[column] for column in factor_columns},
                categorical_blocks={
                    block_id: columns for block_id, columns in expected_blocks.items()
                    if columns
                },
                estimation_column="in_estimation_domain",
                base_weight_column=("candidate_weight" if base_weight_path is not None else None),
                minimum_category_members=minimum_category_members,
            )
        result = None
        failure_reason = ""
        try:
            result = decompose_cross_section(design.regression_input, mode=mode, config=config)
        except ValueError as exc:
            if not str(exc).startswith(("WLS_", "LINEAR_SYSTEM_")):
                raise
            failure_reason = str(exc)
        family_by_factor = dict(zip(
            design.regression_input.factor_ids, design.regression_input.factor_families
        ))
        factor_values = result.factor_returns if result else (None,) * len(
            design.regression_input.factor_ids
        )
        factor_rows.extend({
            "trade_date": return_date,
            "factor_id": factor_id,
            "regression_mode": mode.value,
            "factor_family": family_by_factor[factor_id],
            "factor_return": value,
        } for factor_id, value in zip(design.regression_input.factor_ids, factor_values))
        included_assets = result.included_assets if result else design.regression_input.asset_ids
        specific_values = result.specific_returns if result else (None,) * len(included_assets)
        specific_rows.extend({
            "trade_date": return_date,
            "asset_id": asset_id,
            "regression_mode": mode.value,
            "specific_return": value,
            "in_estimation_domain": in_estimation,
            "exclusion_reason": None,
            "estimation_weight": estimation_weight,
            "huber_weight_multiplier": multiplier,
            "is_outlier_flagged": multiplier is not None and multiplier < 1.0,
        } for asset_id, value, in_estimation, estimation_weight, multiplier in zip(
            included_assets,
            specific_values,
            result.in_estimation_domain if result else (False,) * len(included_assets),
            result.estimation_weights if result else (None,) * len(included_assets),
            result.huber_weight_multipliers if result else (None,) * len(included_assets),
        ))
        quality_rows.append({
            "trade_date": return_date,
            "regression_mode": mode.value,
            "status": result.status if result else "invalid",
            "sample_count": result.sample_count if result else len(included_assets),
            "label_sample_count": result.label_sample_count if result else len(included_assets),
            "estimation_excluded_count": (
                result.label_sample_count - result.sample_count if result else 0
            ),
            "excluded_count": len(result.excluded_assets) if result else 0,
            "factor_count": len(design.regression_input.factor_ids),
            "constraint_count": len(design.regression_input.equality_constraints),
            "expected_matrix_rank": design.expected_matrix_rank,
            "constraint_identification_passed": design.constraint_identification_passed,
            "matrix_rank": result.matrix_rank if result else design.matrix_rank,
            "condition_number": result.condition_number if result else float("inf"),
            "r_squared": result.r_squared if result else None,
            "full_label_r_squared": result.r_squared if result else None,
            "estimation_domain_r_squared": (
                result.estimation_domain_r_squared if result else None
            ),
            "full_label_unweighted_r_squared": (
                result.unweighted_r_squared if result else None
            ),
            "estimation_domain_unweighted_r_squared": (
                result.estimation_domain_unweighted_r_squared if result else None
            ),
            "huber_downweight_count": (
                sum(value is not None and value < 1.0 for value in result.huber_weight_multipliers)
                if result else 0
            ),
            "huber_downweight_rate": (
                sum(value is not None and value < 1.0 for value in result.huber_weight_multipliers)
                / result.sample_count if result and result.sample_count else None
            ),
            "r_squared_in_expected_range": result.r_squared_in_expected_range if result else False,
            "constraint_error": result.constraint_error if result else None,
            "regression_identity_error": result.regression_identity_error if result else None,
            "base_weight_alpha_risk_cross_gram_max": (
                result.base_weight_alpha_risk_cross_gram_max if result else None
            ),
            "effective_weight_alpha_risk_cross_gram_max": (
                result.effective_weight_alpha_risk_cross_gram_max if result else None
            ),
            "maximum_group_residual_correlation": None,
            "exposure_date": exposure_date,
            "industry_member_counts_json": str(dict(design.industry_member_counts)),
            "industry_wls_weight_sums_json": str(dict(design.industry_wls_weight_sums)),
            "industry_constraint_weights_json": str(dict(design.industry_constraint_weights)),
            "warnings": "|".join((*design.warnings, failure_reason) if failure_reason else design.warnings),
        })
        valid_daily = daily.filter(
            (
                pl.col("model_eligible") if categorical_prefixes is not None
                else pl.col("is_tradable")
            ) & pl.col("realized_return").is_finite()
        )
        stats_rows.extend(cross_section_statistics(
            valid_daily.get_column("realized_return").to_list(),
            valid_daily.get_column("board_id").to_list(),
            trade_date=return_date,
            regression_mode=mode.value,
        ))
    return {
        "factor_returns_v1": pl.DataFrame(factor_rows),
        "specific_returns_v1": pl.DataFrame(specific_rows, infer_schema_length=None),
        "factor_regression_quality_v1": pl.DataFrame(quality_rows),
        "cross_section_stats_v1": pl.DataFrame(stats_rows),
    }
