from __future__ import annotations

import math
import numpy as np
from collections.abc import Sequence

from ...canonical_definitions import cross_section_sigma_r
from .contracts import (
    RegressionInput, RegressionMode, RegressionResult, ReturnDecompositionConfig,
)
from .linear_algebra import (
    matvec, null_space_basis, prepare_constrained_design,
    solve_prepared_weighted_least_squares,
)


HUBER_IMPLEMENTATION_ID = "huber_irls_whitened_v1"


def _weighted_r_squared(
    returns: list[float], fitted: list[float], weights: list[float],
) -> float:
    weight_sum = sum(weights)
    mean = sum(weight * value for weight, value in zip(weights, returns)) / weight_sum
    total = sum(weight * (value - mean) ** 2 for weight, value in zip(weights, returns))
    residual = sum(
        weight * (value - prediction) ** 2
        for weight, value, prediction in zip(weights, returns, fitted)
    )
    return 1.0 - residual / total if total > 0 else 0.0


def _unweighted_r_squared(returns: list[float], fitted: list[float]) -> float:
    mean = sum(returns) / len(returns)
    total = sum((value - mean) ** 2 for value in returns)
    residual = sum(
        (value - prediction) ** 2
        for value, prediction in zip(returns, fitted)
    )
    return 1.0 - residual / total if total > 0 else 0.0


def _alpha_risk_cross_gram_max(
    exposures: list[list[float]], weights: list[float], families: Sequence[str],
) -> float | None:
    alpha = [index for index, family in enumerate(families) if family == "alpha"]
    risk = [index for index, family in enumerate(families) if family == "risk"]
    if not alpha or not risk:
        return None
    x = np.asarray(exposures, dtype=float)
    w = np.asarray(weights, dtype=float)
    maximum = 0.0
    for left in alpha:
        left_norm = float(np.sqrt(np.sum(w * np.square(x[:, left]))))
        for right in risk:
            right_norm = float(np.sqrt(np.sum(w * np.square(x[:, right]))))
            denominator = left_norm * right_norm
            if denominator > 0:
                maximum = max(
                    maximum,
                    abs(float(np.sum(w * x[:, left] * x[:, right]))) / denominator,
                )
    return maximum


def _whitened_mad_scale(residuals: np.ndarray, base_weights: np.ndarray) -> float:
    """Normal-consistent robust scale in sqrt(W) residual space about zero."""
    whitened_residuals = np.sqrt(base_weights) * residuals
    absolute_residuals = np.abs(whitened_residuals)
    # Use the mathematical median for even cross-sections as well.  Selecting
    # the upper middle order statistic introduces a size-dependent upward bias
    # into the Huber threshold and changes the downweighting rule by sample size.
    return float(np.median(absolute_residuals)) / 0.6744897501960817


def _robust_fit(
    exposures: list[list[float]], returns: list[float], base_weights: list[float],
    constraints: list[list[float]], config: ReturnDecompositionConfig,
    *, prevalidated_basis: np.ndarray | None = None,
    prevalidated_rank: int | None = None,
) -> tuple[list[float], list[float], float, int, list[float], list[float]]:
    base = np.asarray(base_weights, dtype=float)
    multipliers = np.ones_like(base)
    previous: np.ndarray | None = None
    encountered_singular_gram = False
    if prevalidated_basis is None or prevalidated_rank is None:
        x, _, basis, rank = prepare_constrained_design(exposures, constraints)
    else:
        x = np.asarray(exposures, dtype=float)
        basis, rank = prevalidated_basis, prevalidated_rank
    y = np.asarray(returns, dtype=float)
    constraint_array = (
        np.asarray(constraints, dtype=float)
        if constraints else np.empty((0, x.shape[1]), dtype=float)
    )
    for _ in range(config.maximum_iterations):
        coefficients_array, numeric_condition = solve_prepared_weighted_least_squares(
            x, y, base * multipliers, basis, constraint_array,
            compute_condition=False,
        )
        encountered_singular_gram |= math.isinf(numeric_condition)
        fitted = x @ coefficients_array
        residuals_array = y - fitted
        whitened_residuals = np.sqrt(base) * residuals_array
        scale = _whitened_mad_scale(residuals_array, base)
        if scale <= 1e-15:
            break
        updated_multipliers = np.minimum(
            1.0,
            config.huber_delta * scale / np.maximum(np.abs(whitened_residuals), 1e-15),
        )
        converged = (
            previous is not None
            and float(np.max(np.abs(coefficients_array - previous)))
            <= config.convergence_tolerance
        )
        multipliers = updated_multipliers
        if converged:
            break
        previous = coefficients_array
    robust_weights = base * multipliers
    coefficients_array, numeric_condition = solve_prepared_weighted_least_squares(
        x, y, robust_weights, basis, constraint_array,
    )
    if encountered_singular_gram:
        numeric_condition = float("inf")
    residuals_array = y - x @ coefficients_array
    return (
        coefficients_array.tolist(), residuals_array.tolist(), numeric_condition,
        rank, robust_weights.tolist(), multipliers.tolist(),
    )


def decompose_cross_section(
    inputs: RegressionInput,
    *,
    mode: RegressionMode,
    config: ReturnDecompositionConfig,
) -> RegressionResult:
    row_count = len(inputs.asset_ids)
    if not (
        row_count == len(inputs.exposures) == len(inputs.realized_returns)
        == len(inputs.base_weights) == len(inputs.is_tradable)
    ):
        raise ValueError("L2B_ROW_AXIS_MISMATCH")
    if len(inputs.factor_ids) != len(inputs.factor_families):
        raise ValueError("L2B_FACTOR_AXIS_MISMATCH")
    factor_count = len(inputs.factor_ids)
    if factor_count == 0 or any(len(row) != factor_count for row in inputs.exposures):
        raise ValueError("L2B_DESIGN_MATRIX_INVALID")
    if mode is RegressionMode.RISK_ONLY and any(
        family != "risk" for family in inputs.factor_families
    ):
        raise ValueError("L2B_ALPHA_COLUMN_FORBIDDEN_IN_RISK_ONLY")
    if mode is RegressionMode.RISK_PLUS_ALPHA and "alpha" not in inputs.factor_families:
        raise ValueError("L2B_FULL_MODE_REQUIRES_ALPHA")

    label_indices = [
        index for index in range(row_count)
        if inputs.base_weights[index] > 0
        and math.isfinite(inputs.realized_returns[index])
        and all(math.isfinite(value) for value in inputs.exposures[index])
    ]
    estimation_indices = [index for index in label_indices if inputs.is_tradable[index]]
    if len(estimation_indices) <= factor_count + len(inputs.equality_constraints):
        raise ValueError("L2B_INSUFFICIENT_CROSS_SECTION")
    estimation_exposures = [list(inputs.exposures[index]) for index in estimation_indices]
    estimation_returns = [inputs.realized_returns[index] for index in estimation_indices]
    estimation_base_weights = [inputs.base_weights[index] for index in estimation_indices]
    constraints = [list(row) for row in inputs.equality_constraints]
    if any(len(row) != factor_count for row in constraints):
        raise ValueError("L2B_CONSTRAINT_SHAPE_INVALID")

    prevalidated_basis = None
    prevalidated_rank = None
    if inputs.identification_prevalidated:
        if inputs.prevalidated_matrix_rank is None:
            raise ValueError("L2B_PREVALIDATED_RANK_MISSING")
        prevalidated_basis = np.asarray(
            null_space_basis(constraints, factor_count), dtype=float
        )
        prevalidated_rank = inputs.prevalidated_matrix_rank
    coefficients, _, numeric_condition, rank, robust_weights, multipliers = _robust_fit(
        estimation_exposures, estimation_returns, estimation_base_weights, constraints, config,
        prevalidated_basis=prevalidated_basis,
        prevalidated_rank=prevalidated_rank,
    )
    label_exposures = [list(inputs.exposures[index]) for index in label_indices]
    label_returns = [inputs.realized_returns[index] for index in label_indices]
    label_base_weights = [inputs.base_weights[index] for index in label_indices]
    label_fitted = matvec(label_exposures, coefficients)
    residuals = [target - fitted for target, fitted in zip(label_returns, label_fitted)]
    estimation_fitted = matvec(estimation_exposures, coefficients)
    r_squared = _weighted_r_squared(label_returns, label_fitted, label_base_weights)
    estimation_r_squared = _weighted_r_squared(
        estimation_returns, estimation_fitted, estimation_base_weights,
    )
    unweighted_r_squared = _unweighted_r_squared(label_returns, label_fitted)
    estimation_unweighted_r_squared = _unweighted_r_squared(
        estimation_returns, estimation_fitted,
    )
    constraint_error = max(
        (abs(sum(value * coefficient for value, coefficient in zip(row, coefficients)))
         for row in constraints),
        default=0.0,
    )
    status = "passed"
    if not math.isfinite(numeric_condition) or numeric_condition > config.maximum_condition_number:
        status = "invalid"
    if constraint_error > config.constraint_tolerance:
        status = "invalid"
    regression_identity_error = max(
        (abs(target - prediction - residual)
         for target, prediction, residual in zip(label_returns, label_fitted, residuals)),
        default=0.0,
    )
    if regression_identity_error > config.attribution_identity_tolerance:
        status = "invalid"
    base_cross_gram = _alpha_risk_cross_gram_max(
        estimation_exposures, estimation_base_weights, inputs.factor_families,
    )
    effective_cross_gram = _alpha_risk_cross_gram_max(
        estimation_exposures, robust_weights, inputs.factor_families,
    )
    estimation_position = {index: position for position, index in enumerate(estimation_indices)}
    return RegressionResult(
        mode=mode,
        factor_returns=tuple(coefficients),
        specific_returns=tuple(residuals),
        estimation_weights=tuple(
            robust_weights[estimation_position[index]] if index in estimation_position else None
            for index in label_indices
        ),
        huber_weight_multipliers=tuple(
            multipliers[estimation_position[index]] if index in estimation_position else None
            for index in label_indices
        ),
        in_estimation_domain=tuple(index in estimation_position for index in label_indices),
        included_assets=tuple(inputs.asset_ids[index] for index in label_indices),
        excluded_assets=tuple(
            inputs.asset_ids[index] for index in range(row_count) if index not in label_indices
        ),
        sample_count=len(estimation_indices),
        label_sample_count=len(label_indices),
        matrix_rank=rank,
        condition_number=numeric_condition,
        r_squared=r_squared,
        estimation_domain_r_squared=estimation_r_squared,
        unweighted_r_squared=unweighted_r_squared,
        estimation_domain_unweighted_r_squared=estimation_unweighted_r_squared,
        constraint_error=constraint_error,
        regression_identity_error=regression_identity_error,
        base_weight_alpha_risk_cross_gram_max=base_cross_gram,
        effective_weight_alpha_risk_cross_gram_max=effective_cross_gram,
        r_squared_in_expected_range=(
            config.expected_r_squared_min <= r_squared <= config.expected_r_squared_max
        ),
        status=status,
    )


def attribution_identity_error(
    portfolio_weights: Sequence[float],
    realized_returns: Sequence[float],
    exposures: Sequence[Sequence[float]],
    factor_returns: Sequence[float],
    specific_returns: Sequence[float],
) -> float:
    if not (
        len(portfolio_weights) == len(realized_returns) == len(exposures)
        == len(specific_returns)
    ):
        raise ValueError("L2B_ATTRIBUTION_AXIS_MISMATCH")
    portfolio_return = sum(weight * value for weight, value in zip(portfolio_weights, realized_returns))
    portfolio_exposure = [
        sum(weight * row[column] for weight, row in zip(portfolio_weights, exposures))
        for column in range(len(factor_returns))
    ]
    explained = sum(value * factor_return for value, factor_return in zip(portfolio_exposure, factor_returns))
    specific = sum(weight * value for weight, value in zip(portfolio_weights, specific_returns))
    return abs(portfolio_return - explained - specific)


def grouped_residual_correlation(
    residuals_by_asset: dict[str, Sequence[float]],
    group_by_asset: dict[str, str],
) -> dict[str, float]:
    """Average pairwise residual correlation within each preregistered group."""
    grouped: dict[str, list[str]] = {}
    for asset_id, group_id in group_by_asset.items():
        if asset_id in residuals_by_asset:
            grouped.setdefault(group_id, []).append(asset_id)
    result: dict[str, float] = {}
    for group_id, asset_ids in grouped.items():
        correlations: list[float] = []
        for left_index, left_id in enumerate(asset_ids):
            left = residuals_by_asset[left_id]
            for right_id in asset_ids[left_index + 1:]:
                right = residuals_by_asset[right_id]
                if len(left) != len(right) or len(left) < 3:
                    continue
                left_mean = sum(left) / len(left)
                right_mean = sum(right) / len(right)
                numerator = sum(
                    (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
                )
                denominator = math.sqrt(
                    sum((x - left_mean) ** 2 for x in left)
                    * sum((y - right_mean) ** 2 for y in right)
                )
                if denominator > 0:
                    correlations.append(numerator / denominator)
        result[group_id] = (
            sum(correlations) / len(correlations) if correlations else float("nan")
        )
    return result


def cross_section_statistics(
    realized_returns: Sequence[float], board_ids: Sequence[str], *,
    trade_date: object, regression_mode: str,
) -> list[dict[str, object]]:
    if len(realized_returns) != len(board_ids):
        raise ValueError("L2B_CROSS_SECTION_STATS_AXIS_MISMATCH")
    grouped: dict[str, list[float]] = {}
    for value, board_id in zip(realized_returns, board_ids):
        if math.isfinite(value):
            grouped.setdefault(board_id, []).append(float(value))
    rows = []
    for board_id, values in sorted(grouped.items()):
        count = len(values)
        mean = sum(values) / count
        centered = [value - mean for value in values]
        sigma = cross_section_sigma_r(values)
        skew = sum(value ** 3 for value in centered) / count / sigma ** 3 if sigma > 0 else 0.0
        kurtosis = sum(value ** 4 for value in centered) / count / sigma ** 4 if sigma > 0 else 0.0
        rows.append({
            "trade_date": trade_date,
            "board_id": board_id,
            "regression_mode": regression_mode,
            "sigma_r": sigma,
            "n_valid": count,
            "mean_abs_return": sum(abs(value) for value in values) / count,
            "skew": skew,
            "kurtosis": kurtosis,
        })
    return rows
