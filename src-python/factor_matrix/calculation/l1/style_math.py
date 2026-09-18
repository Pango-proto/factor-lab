"""Small deterministic primitives for role-specific L1 style transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from collections.abc import Sequence

from .projection import weighted_residualize, weighted_scale_only


@dataclass(frozen=True)
class StyleTransformResult:
    values: tuple[float, ...]
    standardized_raw: tuple[float, ...]
    residual_before_scale: tuple[float, ...]
    winsorized_low: int
    winsorized_high: int
    mean_w: float
    sd_w: float
    pre_scale_mean_w: float
    max_abs_corr_with_controls: float


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if len(values) != len(weights) or not values or total <= 0:
        raise ValueError("L1_WEIGHTED_MOMENT_INPUT_INVALID")
    return sum(value * weight for value, weight in zip(values, weights)) / total


def weighted_sd(
    values: Sequence[float], weights: Sequence[float], *, center: float | None = None,
) -> float:
    resolved_center = weighted_mean(values, weights) if center is None else center
    total = sum(weights)
    variance = sum(
        weight * (value - resolved_center) ** 2
        for value, weight in zip(values, weights)
    ) / total
    return math.sqrt(max(0.0, variance))


def weighted_standardize(
    values: Sequence[float], weights: Sequence[float],
) -> tuple[float, ...]:
    center = weighted_mean(values, weights)
    scale = weighted_sd(values, weights, center=center)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("L1_WEIGHTED_STANDARD_DEVIATION_ZERO")
    return tuple((value - center) / scale for value in values)


def mad_winsorize(
    values: Sequence[float], mad_k: float,
) -> tuple[tuple[float, ...], int, int]:
    if not values or mad_k <= 0 or any(not math.isfinite(value) for value in values):
        raise ValueError("L1_MAD_INPUT_INVALID")
    center = median(values)
    mad = median(abs(value - center) for value in values)
    if mad <= 0:
        raise ValueError("L1_MAD_ZERO")
    lower, upper = center - mad_k * mad, center + mad_k * mad
    low_count = sum(value < lower for value in values)
    high_count = sum(value > upper for value in values)
    return (
        tuple(min(upper, max(lower, value)) for value in values),
        low_count,
        high_count,
    )


def weighted_corr(
    left: Sequence[float], right: Sequence[float], weights: Sequence[float],
) -> float:
    left_mean, right_mean = weighted_mean(left, weights), weighted_mean(right, weights)
    left_sd = weighted_sd(left, weights, center=left_mean)
    right_sd = weighted_sd(right, weights, center=right_mean)
    if left_sd <= 0 or right_sd <= 0:
        return 0.0
    covariance = sum(
        weight * (x - left_mean) * (y - right_mean)
        for x, y, weight in zip(left, right, weights)
    ) / sum(weights)
    return covariance / (left_sd * right_sd)


def transform_style_column(
    raw_values: Sequence[float],
    *,
    controls: Sequence[Sequence[float]],
    control_columns: Sequence[Sequence[float]],
    weights: Sequence[float],
    mad_k: float,
    already_standardized: bool = False,
) -> StyleTransformResult:
    if already_standardized:
        standardized = tuple(float(value) for value in raw_values)
        low_count = high_count = 0
    else:
        clipped, low_count, high_count = mad_winsorize(raw_values, mad_k)
        standardized = weighted_standardize(clipped, weights)
    residual = weighted_residualize(
        standardized, controls, weights, controls_are_full_rank=True
    )
    scaled = weighted_scale_only(residual, weights)
    correlations = [
        abs(weighted_corr(scaled, column, weights)) for column in control_columns
    ]
    return StyleTransformResult(
        values=scaled,
        standardized_raw=standardized,
        residual_before_scale=residual,
        winsorized_low=low_count,
        winsorized_high=high_count,
        mean_w=weighted_mean(scaled, weights),
        sd_w=weighted_sd(scaled, weights),
        pre_scale_mean_w=weighted_mean(residual, weights),
        max_abs_corr_with_controls=max(correlations, default=0.0),
    )
