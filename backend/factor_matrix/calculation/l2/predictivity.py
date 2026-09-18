from __future__ import annotations

import math
import numpy as np
from collections.abc import Sequence

from ...factor_engine import FactorRegistryStore
from .contracts import (
    FamaMacBethPoint, NeutralizationFidelityPoint, PredictivityConfig,
    PredictivityPoint,
)
from .linear_algebra import constrained_weighted_least_squares


def _rank(values: Sequence[float]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        for cursor in range(position, end):
            result[indexed[cursor][0]] = average_rank
        position = end
    return tuple(result)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return covariance / denominator if denominator > 0 else None


def spearman_ic(exposures: Sequence[float], forward_returns: Sequence[float]) -> float | None:
    pairs = [
        (float(exposure), float(label))
        for exposure, label in zip(exposures, forward_returns)
        if math.isfinite(exposure) and math.isfinite(label)
    ]
    if len(pairs) < 3:
        return None
    left, right = zip(*pairs)
    return _correlation(_rank(left), _rank(right))


def weighted_residualize_against_frozen_composite(
    candidate: Sequence[float], incumbent_composite: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, ...]:
    """Label-free incremental exposure against a previously frozen composite."""
    if not (len(candidate) == len(incumbent_composite) == len(weights)) or len(candidate) < 3:
        raise ValueError("L2A_INCREMENTAL_COMPOSITE_AXIS_INVALID")
    if any(not math.isfinite(value) for value in (*candidate, *incumbent_composite, *weights)):
        raise ValueError("L2A_INCREMENTAL_COMPOSITE_NONFINITE")
    if any(value <= 0 for value in weights):
        raise ValueError("L2A_INCREMENTAL_COMPOSITE_WEIGHT_INVALID")
    x = np.column_stack([np.ones(len(candidate)), np.asarray(incumbent_composite)])
    y = np.asarray(candidate)
    root = np.sqrt(np.asarray(weights))
    coefficients = np.linalg.lstsq(root[:, None] * x, root * y, rcond=None)[0]
    return tuple((y - x @ coefficients).tolist())


def rolling_out_of_sample_ic(
    exposures_by_date: Sequence[Sequence[float]],
    forward_returns_by_date: Sequence[Sequence[float]],
    *,
    horizon_days: int,
    config: PredictivityConfig,
) -> tuple[PredictivityPoint, ...]:
    """Estimate each date only from labels fully known before its embargo boundary."""
    if len(exposures_by_date) != len(forward_returns_by_date):
        raise ValueError("L2A_DATE_AXIS_MISMATCH")
    if horizon_days not in config.reported_horizons_days:
        raise ValueError("L2A_UNREGISTERED_HORIZON")
    daily_ic = tuple(
        spearman_ic(exposure, label)
        for exposure, label in zip(exposures_by_date, forward_returns_by_date)
    )
    points: list[PredictivityPoint] = []
    for evaluation_index in range(len(daily_ic)):
        training_end = evaluation_index - horizon_days - config.embargo_days
        history = [
            value for value in daily_ic[: max(training_end + 1, 0)] if value is not None
        ]
        if len(history) < config.minimum_burn_in_days:
            points.append(PredictivityPoint(
                evaluation_index=evaluation_index,
                training_end_index=None,
                raw_ic=None,
                shrunk_ic=0.0,
                standard_error=None,
                confidence_low=None,
                confidence_high=None,
                factor_weight=0.0,
                status_signal="burn_in",
            ))
            continue
        mean_ic = sum(history) / len(history)
        variance = sum((value - mean_ic) ** 2 for value in history) / max(len(history) - 1, 1)
        standard_error = math.sqrt(variance / len(history))
        shrinkage = config.shrinkage_prior_precision / (
            config.shrinkage_prior_precision + variance / len(history)
        )
        shrunk = shrinkage * mean_ic
        # A negative/insignificant IC is not enough to decide whether the
        # economic dimension decayed or was absorbed into the risk basis.
        status_signal = "requires_absorption_classification" if shrunk < 0 else "active"
        factor_weight = max(shrunk, 0.0) / standard_error if standard_error > 0 else 0.0
        points.append(PredictivityPoint(
            evaluation_index=evaluation_index,
            training_end_index=training_end,
            raw_ic=mean_ic,
            shrunk_ic=shrunk,
            standard_error=standard_error,
            confidence_low=shrunk - 1.96 * standard_error,
            confidence_high=shrunk + 1.96 * standard_error,
            factor_weight=factor_weight,
            status_signal=status_signal,
        ))
    return tuple(points)


def classify_alpha_decay_or_riskification(
    *, nw_t_significant: bool, factor_return_sd: float,
    incremental_r2: float, exposure_persistence: float,
    regime_count: int, months_observed: int, minimum_months: int,
) -> str:
    """Separate pure decay from a candidate alpha-to-risk migration."""
    if (
        not nw_t_significant and factor_return_sd > 0.0
        and incremental_r2 >= 0.005 and exposure_persistence >= 0.8
        and regime_count >= 2 and months_observed >= minimum_months
    ):
        return "riskification_candidate"
    if (
        not nw_t_significant
        and (factor_return_sd <= 0.0 or incremental_r2 < 0.005)
    ):
        return "invalidated_pure_decay"
    return "requires_more_preregistered_evidence"


def neutralization_fidelity(
    raw: Sequence[PredictivityPoint], neutralized: Sequence[PredictivityPoint], *,
    minimum_absolute_raw_ic: float, minimum_retention_ratio: float,
) -> tuple[NeutralizationFidelityPoint, ...]:
    """Derived, preregistered comparison from one audited label-access event."""
    if len(raw) != len(neutralized):
        raise ValueError("L2A_NEUTRALIZATION_FIDELITY_AXIS_MISMATCH")
    if minimum_absolute_raw_ic <= 0 or not 0 <= minimum_retention_ratio <= 1:
        raise ValueError("L2A_NEUTRALIZATION_FIDELITY_THRESHOLD_INVALID")
    rows = []
    for raw_point, residual_point in zip(raw, neutralized):
        if raw_point.evaluation_index != residual_point.evaluation_index:
            raise ValueError("L2A_NEUTRALIZATION_FIDELITY_INDEX_MISMATCH")
        raw_ic = raw_point.shrunk_ic
        residual_ic = residual_point.shrunk_ic
        if raw_point.training_end_index is None or residual_point.training_end_index is None:
            ratio, sign, status = None, None, "burn_in"
        elif abs(raw_ic) < minimum_absolute_raw_ic:
            ratio, sign, status = None, None, "raw_ic_below_ratio_floor"
        else:
            ratio = abs(residual_ic) / abs(raw_ic)
            sign = raw_ic * residual_ic >= 0
            status = (
                "passed" if sign and ratio >= minimum_retention_ratio
                else "degraded_neutralization_fidelity"
            )
        rows.append(NeutralizationFidelityPoint(
            evaluation_index=raw_point.evaluation_index,
            raw_shrunk_ic=raw_ic,
            neutralized_shrunk_ic=residual_ic,
            absolute_ic_retention_ratio=ratio,
            sign_preserved=sign,
            status=status,
        ))
    return tuple(rows)


def purged_k_fold_indices(
    sample_count: int, *, folds: int, horizon_days: int, embargo_days: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if sample_count < folds or folds < 2:
        raise ValueError("L2A_FOLD_COUNT_INVALID")
    fold_size, remainder = divmod(sample_count, folds)
    result = []
    start = 0
    for fold in range(folds):
        stop = start + fold_size + (1 if fold < remainder else 0)
        test = tuple(range(start, stop))
        excluded_start = max(0, start - horizon_days)
        excluded_stop = min(sample_count, stop + embargo_days)
        train = tuple(index for index in range(sample_count) if not excluded_start <= index < excluded_stop)
        result.append((train, test))
        start = stop
    return tuple(result)


def newey_west_mean(values: Sequence[float], lag: int) -> tuple[float, float, float]:
    if not values:
        raise ValueError("L2A_NW_EMPTY_SERIES")
    if lag < 0:
        raise ValueError("L2A_NW_LAG_INVALID")
    count = len(values)
    mean = sum(values) / count
    centered = [value - mean for value in values]
    long_run_variance = sum(value * value for value in centered) / count
    for offset in range(1, min(lag, count - 1) + 1):
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, count)
        ) / count
        long_run_variance += 2 * (1 - offset / (lag + 1)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    return mean, standard_error, mean / standard_error if standard_error > 0 else 0.0


def rolling_fama_macbeth(
    exposures_by_date: Sequence[Sequence[Sequence[float]]],
    forward_returns_by_date: Sequence[Sequence[float]],
    weights_by_date: Sequence[Sequence[float]],
    *,
    equality_constraints: Sequence[Sequence[float]],
    horizon_days: int,
    config: PredictivityConfig,
) -> tuple[FamaMacBethPoint, ...]:
    """Cross-sectional coefficients with trailing-only Newey-West aggregation."""
    if not (
        len(exposures_by_date) == len(forward_returns_by_date) == len(weights_by_date)
    ):
        raise ValueError("L2A_FM_DATE_AXIS_MISMATCH")
    if horizon_days not in config.reported_horizons_days:
        raise ValueError("L2A_UNREGISTERED_HORIZON")
    daily_coefficients: list[tuple[float, ...] | None] = []
    factor_count = 0
    for exposures, targets, weights in zip(
        exposures_by_date, forward_returns_by_date, weights_by_date
    ):
        if not exposures:
            daily_coefficients.append(None)
            continue
        factor_count = len(exposures[0])
        try:
            coefficients, _, rank = constrained_weighted_least_squares(
                [list(row) for row in exposures], list(targets), list(weights),
                [list(row) for row in equality_constraints],
            )
        except ValueError:
            daily_coefficients.append(None)
            continue
        daily_coefficients.append(tuple(coefficients) if rank == factor_count else None)

    points: list[FamaMacBethPoint] = []
    for evaluation_index in range(len(daily_coefficients)):
        training_end = evaluation_index - horizon_days - config.embargo_days
        history = [
            coefficients for coefficients in daily_coefficients[:max(training_end + 1, 0)]
            if coefficients is not None
        ]
        if len(history) < config.minimum_burn_in_days or factor_count == 0:
            points.append(FamaMacBethPoint(
                evaluation_index, None, (0.0,) * factor_count,
                (None,) * factor_count, (None,) * factor_count,
            ))
            continue
        estimates = [
            newey_west_mean(
                [coefficients[index] for coefficients in history],
                lag=max(horizon_days - 1, 0),
            )
            for index in range(factor_count)
        ]
        points.append(FamaMacBethPoint(
            evaluation_index=evaluation_index,
            training_end_index=training_end,
            coefficients=tuple(item[0] for item in estimates),
            standard_errors=tuple(item[1] for item in estimates),
            nw_t_values=tuple(item[2] for item in estimates),
        ))
    return tuple(points)


def estimate_signal_half_life(ic_by_horizon: Sequence[tuple[int, float]]) -> int | None:
    """Predeclared descriptive rule; it reports decay and never selects the scoring horizon."""
    if not ic_by_horizon:
        return None
    ordered = sorted(ic_by_horizon)
    peak = max(abs(value) for _, value in ordered)
    if peak == 0:
        return None
    peak_position = next(
        position for position, (_, value) in enumerate(ordered) if abs(value) == peak
    )
    for horizon, value in ordered[peak_position:]:
        if abs(value) <= 0.5 * peak:
            return horizon
    return ordered[-1][0]


def descriptive_ic_term_structure(
    ic_by_horizon: Sequence[tuple[int, float]],
) -> dict[str, object]:
    """Description only: this object is forbidden as an evaluation-horizon input."""
    return {
        "descriptive_only": True,
        "may_select_evaluation_horizon": False,
        "signal_half_life_days": estimate_signal_half_life(ic_by_horizon),
        "points": [{"horizon_days": horizon, "ic": value}
                   for horizon, value in sorted(ic_by_horizon)],
    }


def normalize_diagonal_icir_weights(icir_by_factor: Sequence[float]) -> tuple[float, ...]:
    if any(value < 0 for value in icir_by_factor):
        raise ValueError("L2A_NEGATIVE_IC_INVALIDATE_FACTOR")
    denominator = sum(icir_by_factor)
    if denominator == 0:
        return tuple(0.0 for _ in icir_by_factor)
    return tuple(value / denominator for value in icir_by_factor)


def fdr_denominator(
    store: FactorRegistryStore, *, family_root_id: str | None = None,
) -> int:
    """Return the multiple-testing budget from every attempted variant.

    Unsubmitted attempts are intentionally included.  The legacy in-memory
    ``FactorSpec.variant_count`` is not an accounting source.
    """
    return store.research_attempt_count(family_root_id)


def benjamini_hochberg_passes(
    p_values: Sequence[float], *, variant_count_total: int, q: float,
) -> tuple[bool, ...]:
    if variant_count_total < len(p_values) or variant_count_total < 1:
        raise ValueError("L2A_FDR_DENOMINATOR_INVALID")
    if not 0 < q < 1 or any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("L2A_FDR_INPUT_INVALID")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    maximum_passing_rank = 0
    for rank, (_, p_value) in enumerate(ordered, start=1):
        if p_value <= rank * q / variant_count_total:
            maximum_passing_rank = rank
    passing_indices = {index for index, _ in ordered[:maximum_passing_rank]}
    return tuple(index in passing_indices for index in range(len(p_values)))
