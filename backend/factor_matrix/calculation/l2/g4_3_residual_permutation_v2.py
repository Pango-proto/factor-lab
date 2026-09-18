"""Pure v2 statistics for G4.3 residual permutation diagnostics.

v1 compared an absolute group-mean variance to an independent-observation
denominator.  That denominator is not invariant to the WLS+Huber fit used by
the committed G3 product.  v2 uses the conditional permutation distribution
as the denominator instead.

This module deliberately has no storage or forward-return dependencies.  The
I/O runner can be added only after these functions pass synthetic and small
panel checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np


V2_NULL_HYPOTHESIS = (
    "Conditional on the observed residual panel, missingness pattern, and "
    "daily size-by-liquidity strata, group labels are exchangeable within "
    "each date-stratum cell."
)


def _time_by_asset_labels(labels: np.ndarray, n_dates: int, n_assets: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim == 1:
        if values.shape != (n_assets,):
            raise ValueError("G43_V2_LABEL_AXIS_INVALID")
        return np.broadcast_to(values, (n_dates, n_assets)).copy()
    if values.shape != (n_dates, n_assets):
        raise ValueError("G43_V2_LABEL_PANEL_SHAPE_INVALID")
    return values.copy()


def group_temporal_variance(
    labels: np.ndarray,
    U: np.ndarray,
    mask: np.ndarray,
    min_obs_per_day: int,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-group Q, effective days, and median daily group size.

    ``U`` is N×T; labels may be static N or daily T×N.  The group identity is
    allowed to vary only for the permutation null, not for the observed static
    grouping.  Missing observations are excluded from that date's group mean.
    """
    residuals = np.asarray(U, dtype=np.float64)
    valid_mask = np.asarray(mask, dtype=bool)
    if residuals.ndim != 2 or valid_mask.shape != residuals.shape:
        raise ValueError("G43_V2_PANEL_SHAPE_INVALID")
    n_assets, n_dates = residuals.shape
    label_panel = _time_by_asset_labels(labels, n_dates, n_assets)
    means = np.full((n_groups, n_dates), np.nan, dtype=np.float64)
    sizes = np.zeros((n_groups, n_dates), dtype=np.float64)
    for t in range(n_dates):
        observed = valid_mask[:, t] & np.isfinite(residuals[:, t])
        day_labels = label_panel[t]
        for group in range(n_groups):
            selected = observed & (day_labels == group)
            count = int(selected.sum())
            sizes[group, t] = count
            if count >= min_obs_per_day:
                means[group, t] = float(np.mean(residuals[selected, t]))
    effective_days = np.sum(np.isfinite(means), axis=1).astype(np.int64)
    q = np.full(n_groups, np.nan, dtype=np.float64)
    for group in range(n_groups):
        if effective_days[group] > 0:
            q[group] = float(np.var(means[group, np.isfinite(means[group])]))
    median_size = np.asarray(
        [np.nanmedian(row[row > 0]) if np.any(row > 0) else np.nan for row in sizes],
        dtype=np.float64,
    )
    return q, effective_days, median_size


def permute_labels_by_daily_strata(
    labels: np.ndarray, strata: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels independently inside every date×stratum cell."""
    label_panel = np.asarray(labels, dtype=np.int64)
    if label_panel.ndim != 2:
        raise ValueError("G43_V2_PERMUTATION_REQUIRES_DAILY_LABELS")
    strata_panel = np.asarray(strata, dtype=np.int64)
    if strata_panel.shape != label_panel.shape:
        raise ValueError("G43_V2_STRATA_PANEL_SHAPE_INVALID")
    output = label_panel.copy()
    for t in range(label_panel.shape[0]):
        for stratum in np.unique(strata_panel[t][strata_panel[t] >= 0]):
            indices = np.flatnonzero(strata_panel[t] == stratum)
            output[t, indices] = label_panel[t, rng.permutation(indices)]
    return output


def permutation_test_v2(
    labels: np.ndarray,
    strata: np.ndarray,
    U: np.ndarray,
    mask: np.ndarray,
    *,
    n_groups: int,
    n_permutations: int,
    seed: int,
    min_obs_per_day: int,
) -> dict[str, Any]:
    """Compare observed group variance with its daily-stratified null."""
    residuals = np.asarray(U, dtype=np.float64)
    n_assets, n_dates = residuals.shape
    observed_labels = _time_by_asset_labels(labels, n_dates, n_assets)
    strata_panel = np.asarray(strata, dtype=np.int64)
    if strata_panel.shape != (n_dates, n_assets):
        raise ValueError("G43_V2_STRATA_PANEL_SHAPE_INVALID")
    observed, effective_days, median_size = group_temporal_variance(
        observed_labels, residuals, mask, min_obs_per_day, n_groups,
    )
    null = np.empty((n_permutations, n_groups), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in range(n_permutations):
        permuted = permute_labels_by_daily_strata(observed_labels, strata_panel, rng)
        null[index], _, _ = group_temporal_variance(
            permuted, residuals, mask, min_obs_per_day, n_groups,
        )
    null_median = np.full(n_groups, np.nan, dtype=np.float64)
    null_q95 = np.full(n_groups, np.nan, dtype=np.float64)
    for group in range(n_groups):
        finite_null = null[:, group][np.isfinite(null[:, group])]
        if finite_null.size:
            null_median[group] = float(np.median(finite_null))
            null_q95[group] = float(np.quantile(finite_null, 0.95))
    p_value = (1.0 + np.nansum(null >= observed, axis=0)) / (n_permutations + 1)
    return {
        "q_observed": observed,
        "p_value": p_value,
        "effective_days": effective_days,
        "median_group_size": median_size,
        "null_median": null_median,
        "null_q95": null_q95,
        "q_ratio": observed / np.maximum(null_median, 1e-300),
    }


def negative_control_gate_v2(
    control_results: dict[str, dict[str, Any]],
    band: tuple[float, float] = (0.80, 1.25),
) -> dict[str, Any]:
    """Gate controls on observed/null ratios, not an absolute variance scale."""
    lo, hi = band
    detail: dict[str, Any] = {}
    for name, result in control_results.items():
        ratio = np.asarray(result["q_ratio"], dtype=np.float64)
        median_ratio = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan
        detail[name] = {
            "q_ratio_median": median_ratio,
            "in_band": bool(np.isfinite(median_ratio) and lo <= median_ratio <= hi),
            "control_band": [lo, hi],
        }
    return {"passed": all(item["in_band"] for item in detail.values()), "detail": detail}
