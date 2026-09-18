from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

def apply_alpha_direction(values: Sequence[float], direction: int) -> tuple[float, ...]:
    if direction not in (-1, 1):
        raise ValueError("L1_ALPHA_DIRECTION_INVALID")
    return tuple(direction * value for value in values)

def weighted_residualize(
    values: Sequence[float],
    controls: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    controls_are_full_rank: bool = False,
) -> tuple[float, ...]:
    """Project on the control column space; redundant country/industry columns are harmless."""
    if not (len(values) == len(controls) == len(weights)) or not values:
        raise ValueError("L1_PROJECTION_ROW_AXIS_INVALID")
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("L1_PROJECTION_WEIGHT_INVALID")
    design = np.asarray(controls, dtype=float)
    target = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weighted_design = design * np.sqrt(weight_array)[:, None]
    weighted_target = target * np.sqrt(weight_array)
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_target, rcond=1e-12
    )
    if controls_are_full_rank and rank != design.shape[1]:
        raise RuntimeError(
            f"L1_CONTROL_MATRIX_UNEXPECTED_RANK rank={rank} columns={design.shape[1]}"
        )
    residuals = tuple((target - design @ coefficients).tolist())
    for column in range(len(controls[0])):
        inner_product = sum(
            weight * controls[row][column] * residuals[row]
            for row, weight in enumerate(weights)
        )
        control_norm = math.sqrt(sum(
            weight * controls[row][column] ** 2
            for row, weight in enumerate(weights)
        ))
        residual_norm = math.sqrt(sum(
            weight * residual ** 2 for residual, weight in zip(residuals, weights)
        ))
        normalized_inner_product = (
            abs(inner_product) / (control_norm * residual_norm)
            if control_norm > 0 and residual_norm > 0 else 0.0
        )
        if normalized_inner_product > 1e-10:
            raise RuntimeError("L1_WEIGHTED_ORTHOGONALITY_FAILED")
    return residuals


def weighted_scale_only(values: Sequence[float], weights: Sequence[float]) -> tuple[float, ...]:
    if len(values) != len(weights) or not values:
        raise ValueError("L1_SCALE_ROW_AXIS_INVALID")
    denominator = sum(weights)
    variance = sum(weight * value * value for value, weight in zip(values, weights)) / denominator
    scale = math.sqrt(variance)
    if scale <= 0:
        raise ValueError("L1_RESIDUAL_SCALE_ZERO")
    return tuple(value / scale for value in values)
