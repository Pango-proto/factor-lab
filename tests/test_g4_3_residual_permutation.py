from __future__ import annotations

import numpy as np

from factor_matrix.calculation.l2.g4_3_residual_permutation import (
    G43Config,
    _apply_label_stability_gate,
    apply_fdr_bh,
    build_indicator,
    group_stat,
    permute_within_strata,
    prepare,
)


def test_permutation_moves_labels_only_within_strata() -> None:
    labels = np.asarray([0, 1, 0, 1, 2, 2])
    strata = np.asarray([0, 0, 1, 1, 1, 1])
    result = permute_within_strata(labels, strata, np.random.default_rng(7))
    for value in (0, 1):
        assert sorted(result[strata == value].tolist()) == sorted(labels[strata == value].tolist())


def test_group_stat_is_one_for_independent_equal_variance_panel() -> None:
    rng = np.random.default_rng(4)
    U = rng.normal(size=(100, 200))
    mask = np.ones_like(U, dtype=bool)
    var_i = np.var(U, axis=1)
    labels = np.repeat(np.arange(4), 25)
    stats, group_size = group_stat(build_indicator(labels, 4), U, mask, var_i, 5)
    assert group_size.tolist() == [25.0] * 4
    assert np.all(np.abs(stats - 1.0) < 0.35)


def test_prepare_returns_finite_standardized_panel() -> None:
    U = np.asarray([[1.0, 2.0, np.nan, 100.0], [-1.0, 0.0, 1.0, np.nan]])
    prepared, mask, variance = prepare(U, 5.0)
    assert np.isfinite(prepared).all()
    assert mask.tolist() == [[True, True, False, True], [True, True, True, False]]
    assert np.all(variance > 0)


def test_bh_is_applied_across_all_groupings() -> None:
    results = {
        "a": {"p_value": np.asarray([0.001, 0.5])},
        "b": {"p_value": np.asarray([0.01])},
    }
    applied = apply_fdr_bh(results, 0.05)
    assert applied["a"]["bh_passed"].tolist() == [True, False]
    assert applied["b"]["bh_passed"].tolist() == [True]


def test_config_rejects_non_pit_groupings() -> None:
    config = G43Config(
        input_run_id="g3",
        strata_definition="test",
        groupings=("concept",),
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "G43_GROUPING_NOT_PIT_SUPPORTED" in str(exc)
    else:
        raise AssertionError("non-PIT grouping was accepted")


def test_unstable_modal_labels_are_excluded_from_permutation() -> None:
    labels = np.asarray([0, 1, 0, 1])
    gated, metadata = _apply_label_stability_gate(
        labels,
        {"median_stability": 0.5, "assigned_assets": len(labels)},
    )
    assert gated.tolist() == [-1, -1, -1, -1]
    assert metadata["status"] == "dropped_label_stability_below_0.9"
    assert metadata["assigned_assets_after_gate"] == 0


def test_stable_modal_labels_remain_usable() -> None:
    labels = np.asarray([0, 1, 0, 1])
    gated, metadata = _apply_label_stability_gate(
        labels,
        {"median_stability": 1.0, "assigned_assets": len(labels)},
    )
    assert gated.tolist() == labels.tolist()
    assert metadata["status"] == "usable"
