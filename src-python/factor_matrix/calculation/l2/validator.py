from __future__ import annotations

from collections.abc import Mapping, Sequence


def cross_chain_magnitude_validation(
    gamma_by_factor: Mapping[str, float],
    horizon_by_factor: Mapping[str, int],
    alpha_factor_returns_by_factor: Mapping[str, Sequence[float]],
    *, lower_ratio: float = 0.2, upper_ratio: float = 5.0,
) -> tuple[dict[str, object], ...]:
    """Read-only comparison of already-published L2a and L2b artifacts."""
    factor_ids = sorted(set(gamma_by_factor) & set(alpha_factor_returns_by_factor))
    rows = []
    for factor_id in factor_ids:
        horizon = horizon_by_factor[factor_id]
        if horizon < 1:
            raise ValueError("L2_VALIDATOR_HORIZON_INVALID")
        gamma_per_day = gamma_by_factor[factor_id] / horizon
        returns = alpha_factor_returns_by_factor[factor_id]
        mean_return = sum(returns) / len(returns) if returns else 0.0
        ratio = (
            abs(mean_return / gamma_per_day) if gamma_per_day != 0 else float("inf")
        )
        rows.append({
            "factor_id": factor_id,
            "gamma_per_day": gamma_per_day,
            "mean_alpha_factor_return": mean_return,
            "magnitude_ratio": ratio,
            "passed": lower_ratio <= ratio <= upper_ratio,
        })
    return tuple(rows)
