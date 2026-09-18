from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping, Sequence

import polars as pl
import numpy as np

from ...canonical_definitions import constraint_weights, wls_weight
from .contracts import DailyDesign, RegressionInput
from .linear_algebra import matrix_rank


def build_daily_categorical_constrained_design(
    frame: pl.DataFrame,
    *,
    exposure_columns: Sequence[str],
    family_by_column: Mapping[str, str],
    categorical_blocks: Mapping[str, Sequence[str]],
    return_column: str = "realized_return",
    market_cap_column: str = "float_mkt_cap",
    base_weight_column: str | None = None,
    eligible_column: str = "model_eligible",
    estimation_column: str | None = None,
    minimum_category_members: int = 5,
) -> DailyDesign:
    """Consume L1's exact columns and build one cap-weighted constraint per block."""
    required = {
        "asset_id", return_column, market_cap_column, eligible_column, *exposure_columns,
    }
    if base_weight_column is not None:
        required.add(base_weight_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"L2B_DESIGN_COLUMNS_MISSING columns={','.join(missing)}")
    if set(exposure_columns) != set(family_by_column):
        raise ValueError("L2B_EXPOSURE_FAMILY_METADATA_MISMATCH")
    if "risk_country" not in exposure_columns:
        raise ValueError("L2B_COUNTRY_FACTOR_REQUIRED")
    assigned = [column for columns in categorical_blocks.values() for column in columns]
    if len(assigned) != len(set(assigned)) or not set(assigned) <= set(exposure_columns):
        raise ValueError("L2B_CATEGORICAL_BLOCK_COLUMNS_INVALID")

    label_domain = frame.filter(
        pl.col(eligible_column)
        & pl.col(return_column).is_finite()
        & pl.col(market_cap_column).is_finite()
        & (pl.col(market_cap_column) > 0)
    )
    for column in exposure_columns:
        label_domain = label_domain.filter(pl.col(column).is_finite())
    if label_domain.is_empty():
        raise ValueError("L2B_NO_ELIGIBLE_ROWS")
    if estimation_column is None:
        estimation = label_domain
    else:
        if estimation_column not in frame.columns:
            raise ValueError(f"L2B_ESTIMATION_DOMAIN_COLUMN_MISSING column={estimation_column}")
        estimation = label_domain.filter(pl.col(estimation_column))
    if estimation.is_empty():
        raise ValueError("L2B_NO_ESTIMATION_ROWS")
    if any(
        abs(float(value) - 1.0) > 1e-12
        for value in label_domain.get_column("risk_country")
    ):
        raise ValueError("L2B_COUNTRY_FACTOR_NOT_CONSTANT_ONE")

    caps = estimation.get_column(market_cap_column).to_list()
    constraints: list[tuple[float, ...]] = []
    block_member_counts: dict[str, dict[str, int]] = {}
    block_constraint_weights: dict[str, dict[str, float]] = {}
    block_wls_weight_sums: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    column_positions = {column: index for index, column in enumerate(exposure_columns)}
    for block_id, columns_value in categorical_blocks.items():
        columns = tuple(columns_value)
        if not columns:
            continue
        row_sums = estimation.select(pl.sum_horizontal(*columns).alias("row_sum"))["row_sum"]
        if row_sums.min() != 1.0 or row_sums.max() != 1.0:
            raise ValueError(f"L2B_CATEGORICAL_ONE_HOT_FAILED block={block_id}")
        counts = {
            column: int(estimation.get_column(column).sum()) for column in columns
        }
        if any(count <= 0 for count in counts.values()):
            raise ValueError(f"L2B_EMPTY_CATEGORY_NOT_REMOVED block={block_id}")
        cap_sums = [
            sum(cap * float(value) for cap, value in zip(caps, estimation.get_column(column)))
            for column in columns
        ]
        wls_sums = {
            column: sum(
                wls_weight(cap) * float(value)
                for cap, value in zip(caps, estimation.get_column(column))
            )
            for column in columns
        }
        normalized = constraint_weights(tuple(cap_sums))
        weights_by_column = dict(zip(columns, normalized))
        constraint = [0.0] * len(exposure_columns)
        for column, weight in weights_by_column.items():
            constraint[column_positions[column]] = weight
        constraints.append(tuple(constraint))
        block_member_counts[block_id] = counts
        block_constraint_weights[block_id] = weights_by_column
        block_wls_weight_sums[block_id] = wls_sums
        warnings.extend(
            f"CATEGORY_MEMBER_COUNT_LOW block={block_id} category={column} count={count}"
            for column, count in counts.items() if count < minimum_category_members
        )

    estimation_rows = [
        tuple(float(record[column]) for column in exposure_columns)
        for record in estimation.iter_rows(named=True)
    ]
    base_weights = tuple(
        wls_weight(value) for value in caps
    ) if base_weight_column is None else tuple(
        float(value) for value in estimation.get_column(base_weight_column)
    )
    if any(not np.isfinite(value) or value <= 0 for value in base_weights):
        raise ValueError("L2B_BASE_WEIGHT_INVALID")
    raw_rank = matrix_rank([list(row) for row in estimation_rows])
    constraint_rank = matrix_rank([list(row) for row in constraints]) if constraints else 0
    expected_rank = len(exposure_columns) - constraint_rank
    x_array = np.asarray(estimation_rows, dtype=float)
    weight_array = np.asarray(base_weights, dtype=float)
    gram = (x_array.T @ (weight_array[:, None] * x_array)).tolist()
    stacked_rank = matrix_rank(gram + [list(row) for row in constraints])
    constraint_count = len(constraints)
    kkt = [
        gram[row] + [constraints[column][row] for column in range(constraint_count)]
        for row in range(len(exposure_columns))
    ] + [list(row) + [0.0] * constraint_count for row in constraints]
    kkt_rank = matrix_rank(kkt)
    identified = (
        constraint_rank == constraint_count
        and raw_rank == expected_rank
        and stacked_rank == len(exposure_columns)
        and kkt_rank == len(exposure_columns) + constraint_count
    )
    if not identified:
        warnings.append(
            "CONSTRAINT_IDENTIFICATION_FAILED "
            f"raw_rank={raw_rank} expected={expected_rank} "
            f"constraint_rank={constraint_rank} constraint_count={constraint_count} "
            f"stacked_rank={stacked_rank} kkt_rank={kkt_rank}"
        )

    industry_columns = tuple(categorical_blocks.get("industry", ()))
    industry_counts = block_member_counts.get("industry", {})
    industry_constraint = block_constraint_weights.get("industry", {})
    industry_wls_sums = block_wls_weight_sums.get("industry", {})
    return DailyDesign(
        regression_input=RegressionInput(
            asset_ids=tuple(label_domain.get_column("asset_id").to_list()),
            factor_ids=tuple(exposure_columns),
            factor_families=tuple(family_by_column[column] for column in exposure_columns),
            exposures=tuple(
                tuple(float(record[column]) for column in exposure_columns)
                for record in label_domain.iter_rows(named=True)
            ),
            realized_returns=tuple(label_domain.get_column(return_column).to_list()),
            base_weights=(tuple(
                wls_weight(value)
                for value in label_domain.get_column(market_cap_column).to_list()
            ) if base_weight_column is None else tuple(
                float(value) for value in label_domain.get_column(base_weight_column)
            )),
            is_tradable=tuple(
                True if estimation_column is None else bool(value)
                for value in (
                    [True] * label_domain.height
                    if estimation_column is None
                    else label_domain.get_column(estimation_column).to_list()
                )
            ),
            equality_constraints=tuple(constraints),
            identification_prevalidated=identified,
            prevalidated_matrix_rank=raw_rank if identified else None,
        ),
        present_industries=industry_columns,
        industry_member_counts=industry_counts,
        industry_wls_weight_sums=industry_wls_sums,
        industry_constraint_weights=industry_constraint,
        matrix_rank=raw_rank,
        expected_matrix_rank=expected_rank,
        constraint_identification_passed=identified,
        warnings=tuple(warnings),
        categorical_blocks={key: tuple(value) for key, value in categorical_blocks.items()},
        category_member_counts=block_member_counts,
        category_constraint_weights=block_constraint_weights,
        category_wls_weight_sums=block_wls_weight_sums,
        constraint_rank=constraint_rank,
        stacked_rank=stacked_rank,
        kkt_rank=kkt_rank,
    )


def build_daily_industry_constrained_design(
    frame: pl.DataFrame,
    *,
    exposure_columns: Sequence[str],
    family_by_column: Mapping[str, str],
    industry_column: str = "industry_id",
    return_column: str = "realized_return",
    market_cap_column: str = "float_mkt_cap",
    tradable_column: str = "is_tradable",
    base_weight_column: str | None = None,
    minimum_industry_members_warning: int = 5,
) -> DailyDesign:
    """Compatibility adapter into the canonical categorical design builder."""
    required = {
        "asset_id", industry_column, return_column, market_cap_column, tradable_column,
        *exposure_columns,
    }
    if base_weight_column is not None:
        required.add(base_weight_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"L2B_DESIGN_COLUMNS_MISSING columns={','.join(missing)}")
    if set(exposure_columns) != set(family_by_column):
        raise ValueError("L2B_EXPOSURE_FAMILY_METADATA_MISMATCH")
    if "country" not in exposure_columns:
        raise ValueError("L2B_COUNTRY_FACTOR_REQUIRED")
    if any(column.startswith("industry_") for column in exposure_columns):
        raise ValueError("L2B_PRECOMPUTED_INDUSTRY_DUMMIES_FORBIDDEN")

    eligible = frame.filter(pl.col(tradable_column) & pl.col(industry_column).is_not_null())
    industries = tuple(sorted(eligible.get_column(industry_column).unique().to_list()))
    if not industries:
        raise ValueError("L2B_NO_PRESENT_INDUSTRY")
    rename = {
        column: ("risk_country" if column == "country" else f"risk_{column}")
        for column in exposure_columns
    }
    normalized = eligible.rename(rename).with_columns(
        (pl.col(tradable_column) & pl.col(industry_column).is_not_null()).alias("model_eligible")
    )
    industry_columns = tuple(f"risk_industry_{industry}" for industry in industries)
    normalized = normalized.with_columns(*[
        pl.when(pl.col(industry_column) == industry).then(1.0).otherwise(0.0).alias(column)
        for industry, column in zip(industries, industry_columns, strict=True)
    ])
    internal_exposure_columns = tuple(rename[column] for column in exposure_columns) + industry_columns
    design = build_daily_categorical_constrained_design(
        normalized,
        exposure_columns=internal_exposure_columns,
        family_by_column={column: "risk" for column in internal_exposure_columns},
        categorical_blocks={"industry": industry_columns},
        return_column=return_column,
        market_cap_column=market_cap_column,
        eligible_column="model_eligible",
        base_weight_column=base_weight_column,
        minimum_category_members=minimum_industry_members_warning,
    )

    def public_name(column: str) -> str:
        if column == "risk_country":
            return "country"
        if column.startswith("risk_industry_"):
            return "industry_" + column.removeprefix("risk_industry_")
        return column.removeprefix("risk_")

    def public_industry_name(column: str) -> str:
        return column.removeprefix("risk_industry_")

    regression_input = replace(
        design.regression_input,
        factor_ids=tuple(public_name(column) for column in design.regression_input.factor_ids),
    )
    warnings = tuple(
        warning.replace(
            "CATEGORY_MEMBER_COUNT_LOW block=industry category=risk_industry_",
            "INDUSTRY_MEMBER_COUNT_LOW industry=",
        )
        for warning in design.warnings
        if "CATEGORY_MEMBER_COUNT_LOW" in warning
    ) + tuple(
        warning for warning in design.warnings if "CATEGORY_MEMBER_COUNT_LOW" not in warning
    )
    return replace(
        design,
        regression_input=regression_input,
        present_industries=industries,
        industry_member_counts={
            public_industry_name(key): value
            for key, value in design.industry_member_counts.items()
        },
        industry_wls_weight_sums={
            public_industry_name(key): value
            for key, value in design.industry_wls_weight_sums.items()
        },
        industry_constraint_weights={
            public_industry_name(key): value
            for key, value in design.industry_constraint_weights.items()
        },
        warnings=warnings,
    )
