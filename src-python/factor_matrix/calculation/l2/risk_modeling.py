from __future__ import annotations

import math
import random
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


SPECIFIC_VARIANCE_IMPLEMENTATION_ID = (
    "winsorized_residual_squared_ewma_group_shrinkage_v1"
)


def validate_specific_variance_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))["specific_variance"]
    if payload.get("implementation_id") != SPECIFIC_VARIANCE_IMPLEMENTATION_ID:
        raise ValueError("L2C_SPECIFIC_VARIANCE_IMPLEMENTATION_CONFIG_MISMATCH")
    return payload


@dataclass(frozen=True)
class BiasTestResult:
    standardized_return_std: float
    lower_bound: float
    upper_bound: float
    observation_count: int
    passed: bool


@dataclass(frozen=True)
class ResidualCorrelationResult:
    group_id: str
    member_count: int
    mean_residual_correlation: float
    permutation_p_value: float
    bh_passed: bool


def _ewma_weights(count: int, half_life_days: float) -> list[float]:
    if count < 2 or half_life_days <= 0:
        raise ValueError("L2C_EWMA_INPUT_INVALID")
    decay = math.exp(math.log(0.5) / half_life_days)
    raw = [decay ** (count - 1 - index) for index in range(count)]
    total = sum(raw)
    return [value / total for value in raw]


def ewma_factor_covariance(
    returns_by_factor: Mapping[str, Sequence[float]], half_life_days: float,
) -> dict[tuple[str, str], float]:
    factor_ids = sorted(returns_by_factor)
    if not factor_ids:
        raise ValueError("L2C_FACTOR_RETURNS_EMPTY")
    count = len(returns_by_factor[factor_ids[0]])
    if any(len(returns_by_factor[factor]) != count for factor in factor_ids):
        raise ValueError("L2C_FACTOR_RETURN_AXIS_MISMATCH")
    weights = _ewma_weights(count, half_life_days)
    means = {
        factor: sum(weight * value for weight, value in zip(weights, returns_by_factor[factor]))
        for factor in factor_ids
    }
    return {
        (left, right): sum(
            weight * (x - means[left]) * (y - means[right])
            for weight, x, y in zip(weights, returns_by_factor[left], returns_by_factor[right])
        )
        for left in factor_ids for right in factor_ids
    }


def ewma_specific_variance(
    returns_by_asset: Mapping[str, Sequence[float]], half_life_days: float,
    *, group_by_asset: Mapping[str, str] | None = None,
    lookback_days: int = 252, minimum_observations: int = 60,
    clip_multiple: float = 5.0, prior_effective_observations: float = 120.0,
) -> dict[str, float]:
    """PIT-winsorized squared-residual EWMA with group-target shrinkage."""
    comparison = specific_variance_winsorization_comparison(
        returns_by_asset, half_life_days, group_by_asset=group_by_asset,
        lookback_days=lookback_days, minimum_observations=minimum_observations,
        clip_multiple=clip_multiple,
        prior_effective_observations=prior_effective_observations,
    )
    return {
        asset_id: values["winsorized_variance"]
        for asset_id, values in comparison.items()
    }


def specific_variance_winsorization_comparison(
    returns_by_asset: Mapping[str, Sequence[float]], half_life_days: float,
    *, group_by_asset: Mapping[str, str] | None = None,
    lookback_days: int = 252, minimum_observations: int = 60,
    clip_multiple: float = 5.0, prior_effective_observations: float = 120.0,
) -> dict[str, dict[str, float]]:
    """Return like-for-like shrunk risk with and without PIT winsorization."""
    winsorized_raw_variance: dict[str, float] = {}
    nonwinsorized_raw_variance: dict[str, float] = {}
    effective_observations: dict[str, float] = {}
    for asset_id, values in returns_by_asset.items():
        if len(values) < 2:
            raise ValueError("L2C_SPECIFIC_VARIANCE_HISTORY_INSUFFICIENT")
        robust_squared: list[float] = []
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise ValueError("L2C_SPECIFIC_VARIANCE_NONFINITE_RETURN")
            prior = [
                float(item) for item in values[max(0, index-lookback_days):index]
                if math.isfinite(item)
            ]
            clipped = float(value)
            if len(prior) >= minimum_observations:
                center = statistics.median(prior)
                scale = 1.482602218505602 * statistics.median(
                    abs(item - center) for item in prior
                )
                if scale > 0:
                    clipped = min(
                        max(clipped, center - clip_multiple * scale),
                        center + clip_multiple * scale,
                    )
            robust_squared.append(clipped * clipped)
        weights = _ewma_weights(len(robust_squared), half_life_days)
        winsorized_raw_variance[asset_id] = sum(
            weight * squared for weight, squared in zip(weights, robust_squared)
        )
        nonwinsorized_raw_variance[asset_id] = sum(
            weight * float(value) * float(value)
            for weight, value in zip(weights, values)
        )
        effective_observations[asset_id] = 1.0 / sum(weight * weight for weight in weights)
    groups = group_by_asset or {
        asset_id: "all" for asset_id in winsorized_raw_variance
    }
    if set(groups) != set(winsorized_raw_variance):
        raise ValueError("L2C_SPECIFIC_VARIANCE_GROUP_AXIS_MISMATCH")
    winsorized_targets: dict[str, float] = {}
    nonwinsorized_targets: dict[str, float] = {}
    for group_id in set(groups.values()):
        members = [
            asset_id for asset_id in winsorized_raw_variance
            if groups[asset_id] == group_id
        ]
        winsorized_targets[group_id] = sum(
            winsorized_raw_variance[asset_id] for asset_id in members
        ) / len(members)
        nonwinsorized_targets[group_id] = sum(
            nonwinsorized_raw_variance[asset_id] for asset_id in members
        ) / len(members)

    def shrunk(asset_id: str, value: float, target: float) -> float:
        effective = effective_observations[asset_id]
        return (
            effective * value + prior_effective_observations * target
        ) / (effective + prior_effective_observations)

    return {
        asset_id: {
            "winsorized_variance": (winsorized := shrunk(
                asset_id, winsorized_raw_variance[asset_id],
                winsorized_targets[groups[asset_id]],
            )),
            "nonwinsorized_variance": (nonwinsorized := shrunk(
                asset_id, nonwinsorized_raw_variance[asset_id],
                nonwinsorized_targets[groups[asset_id]],
            )),
            "winsorized_sigma": math.sqrt(winsorized),
            "nonwinsorized_sigma": math.sqrt(nonwinsorized),
            "winsorized_to_nonwinsorized_sigma_ratio": (
                math.sqrt(winsorized / nonwinsorized)
                if nonwinsorized > 0 else math.nan
            ),
        }
        for asset_id in winsorized_raw_variance
    }


def summarize_specific_risk_ratio_by_group(
    comparison_by_asset: Mapping[str, Mapping[str, float]],
    group_by_asset: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Aggregate the winsorization diagnostic on a frozen grouping axis."""
    if set(comparison_by_asset) != set(group_by_asset):
        raise ValueError("L2C_SPECIFIC_RISK_DIAGNOSTIC_GROUP_AXIS_MISMATCH")
    grouped: dict[str, list[float]] = {}
    for asset_id, values in comparison_by_asset.items():
        grouped.setdefault(group_by_asset[asset_id], []).append(
            float(values["winsorized_to_nonwinsorized_sigma_ratio"])
        )
    return {
        group_id: {
            "asset_count": float(len(values)),
            "median_winsorized_to_nonwinsorized_sigma_ratio": statistics.median(values),
            "mean_winsorized_to_nonwinsorized_sigma_ratio": sum(values) / len(values),
        }
        for group_id, values in sorted(grouped.items())
    }


def bias_test(
    realized_portfolio_returns: Sequence[float], predicted_variances: Sequence[float],
    *, lower_bound: float, upper_bound: float, minimum_observations: int,
) -> BiasTestResult:
    if len(realized_portfolio_returns) != len(predicted_variances):
        raise ValueError("L2C_BIAS_TEST_AXIS_MISMATCH")
    standardized = [
        realized / math.sqrt(variance)
        for realized, variance in zip(realized_portfolio_returns, predicted_variances)
        if variance > 0 and math.isfinite(realized) and math.isfinite(variance)
    ]
    if len(standardized) < minimum_observations:
        raise ValueError("L2C_BIAS_TEST_INSUFFICIENT_HISTORY")
    mean = sum(standardized) / len(standardized)
    standard_deviation = math.sqrt(sum(
        (value - mean) ** 2 for value in standardized
    ) / (len(standardized) - 1))
    return BiasTestResult(
        standardized_return_std=standard_deviation,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        observation_count=len(standardized),
        passed=lower_bound <= standard_deviation <= upper_bound,
    )


def _pairwise_mean_correlation(
    asset_ids: Sequence[str], residuals_by_asset: Mapping[str, Sequence[float]],
) -> float:
    correlations = []
    for left_index, left_id in enumerate(asset_ids):
        left = residuals_by_asset[left_id]
        for right_id in asset_ids[left_index + 1:]:
            right = residuals_by_asset[right_id]
            if len(left) != len(right) or len(left) < 3:
                continue
            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)
            numerator = sum((x-left_mean)*(y-right_mean) for x, y in zip(left, right))
            denominator = math.sqrt(
                sum((x-left_mean)**2 for x in left) * sum((y-right_mean)**2 for y in right)
            )
            if denominator > 0:
                correlations.append(numerator / denominator)
    if not correlations:
        raise ValueError("L2C_RESIDUAL_GROUP_CORRELATION_UNAVAILABLE")
    return sum(correlations) / len(correlations)


def stratified_residual_permutation_test(
    residuals_by_asset: Mapping[str, Sequence[float]],
    candidate_group_by_asset: Mapping[str, str],
    stratum_by_asset: Mapping[str, tuple[str, ...]],
    *, permutations: int, random_seed: int, bh_q: float,
) -> tuple[ResidualCorrelationResult, ...]:
    if permutations < 1 or not 0 < bh_q < 1:
        raise ValueError("L2C_PERMUTATION_CONFIG_INVALID")
    universe_by_stratum: dict[tuple[str, ...], list[str]] = {}
    for asset_id in residuals_by_asset:
        if asset_id in stratum_by_asset:
            universe_by_stratum.setdefault(stratum_by_asset[asset_id], []).append(asset_id)
    groups: dict[str, list[str]] = {}
    for asset_id, group_id in candidate_group_by_asset.items():
        if asset_id in residuals_by_asset and asset_id in stratum_by_asset:
            groups.setdefault(group_id, []).append(asset_id)
    rng = random.Random(random_seed)
    raw: list[tuple[str, int, float, float]] = []
    for group_id, members in sorted(groups.items()):
        observed = _pairwise_mean_correlation(members, residuals_by_asset)
        stratum_counts: dict[tuple[str, ...], int] = {}
        for member in members:
            stratum = stratum_by_asset[member]
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        null_values = []
        for _ in range(permutations):
            sample = []
            for stratum, count in stratum_counts.items():
                candidates = universe_by_stratum.get(stratum, [])
                if len(candidates) < count:
                    raise ValueError(
                        f"L2C_STRATIFIED_SAMPLE_INSUFFICIENT stratum={stratum}"
                    )
                sample.extend(rng.sample(candidates, count))
            null_values.append(_pairwise_mean_correlation(sample, residuals_by_asset))
        p_value = (1 + sum(value >= observed for value in null_values)) / (permutations + 1)
        raw.append((group_id, len(members), observed, p_value))
    ordered = sorted(enumerate(raw), key=lambda item: item[1][3])
    maximum_passing_rank = 0
    for rank, (_, (_, _, _, p_value)) in enumerate(ordered, start=1):
        if p_value <= rank * bh_q / len(raw):
            maximum_passing_rank = rank
    passing = {index for index, _ in ordered[:maximum_passing_rank]}
    return tuple(
        ResidualCorrelationResult(group_id, count, observed, p_value, index in passing)
        for index, (group_id, count, observed, p_value) in enumerate(raw)
    )


def risk_set_freeze_decision(
    *, bias_passed: bool, residual_results: Sequence[ResidualCorrelationResult],
    significant_dimensions_are_pit: bool, statistical_factor_already_added: bool,
) -> str:
    if not bias_passed:
        return "blocked_bias_test_failed"
    significant = [result for result in residual_results if result.bh_passed]
    if not significant:
        return "sufficient_can_freeze"
    if significant_dimensions_are_pit:
        return "iterate_add_pit_risk_factor"
    if not statistical_factor_already_added:
        return "iterate_add_preregistered_statistical_factor"
    return "can_freeze_with_documented_known_residual"
