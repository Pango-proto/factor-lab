from __future__ import annotations

import numpy as np

from factor_matrix.calculation.l2.g4_3_residual_permutation_v2 import (
    group_temporal_variance,
    negative_control_gate_v2,
    permute_labels_by_daily_strata,
    permutation_test_v2,
)


def test_daily_stratified_permutation_preserves_each_cell() -> None:
    labels = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]])
    strata = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]])
    shuffled = permute_labels_by_daily_strata(labels, strata, np.random.default_rng(3))
    for day in range(2):
        for cell in (0, 1):
            idx = strata[day] == cell
            assert sorted(shuffled[day, idx].tolist()) == sorted(labels[day, idx].tolist())


def test_exchangeable_panel_has_null_ratio_near_one() -> None:
    rng = np.random.default_rng(11)
    U = rng.normal(size=(120, 40))
    mask = np.ones_like(U, dtype=bool)
    labels = np.repeat(np.arange(4), 30)
    strata = np.tile(np.repeat(np.arange(4), 30), (40, 1))
    result = permutation_test_v2(
        labels, strata, U, mask, n_groups=4, n_permutations=150,
        seed=19, min_obs_per_day=5,
    )
    assert np.all(np.isfinite(result["q_ratio"]))
    assert np.all((result["q_ratio"] > 0.65) & (result["q_ratio"] < 1.45))
    assert negative_control_gate_v2({"board": result})["passed"]


def test_group_component_is_detected_against_conditional_null() -> None:
    rng = np.random.default_rng(17)
    n_assets, n_dates = 120, 40
    labels = np.repeat(np.arange(4), 30)
    group_shock = rng.normal(scale=1.0, size=(4, n_dates))
    U = rng.normal(scale=0.2, size=(n_assets, n_dates))
    U += group_shock[labels]
    strata = np.zeros((n_dates, n_assets), dtype=np.int64)
    result = permutation_test_v2(
        labels, strata, U, np.ones_like(U, dtype=bool), n_groups=4,
        n_permutations=150, seed=23, min_obs_per_day=5,
    )
    assert np.nanmedian(result["q_ratio"]) > 2.0
    assert np.any(result["p_value"] < 0.05)


def test_control_gate_uses_ratio_not_absolute_q() -> None:
    result = {"q_ratio": np.asarray([1.05, 0.98]), "q_observed": np.asarray([2.3, 2.1])}
    assert negative_control_gate_v2({"board": result})["passed"]


def test_group_variance_respects_minimum_daily_observations() -> None:
    U = np.ones((4, 3))
    mask = np.ones_like(U, dtype=bool)
    labels = np.asarray([0, 0, 1, 1])
    q, effective_days, _ = group_temporal_variance(labels, U, mask, 3, 2)
    assert np.isnan(q).all()
    assert effective_days.tolist() == [0, 0]
