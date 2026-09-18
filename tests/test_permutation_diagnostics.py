from datetime import date, timedelta
import json

import numpy as np
import polars as pl
import pytest

from factor_matrix.calculation.components.probe_evaluation import permutation_means
from factor_matrix.calculation.l2.estimation_panel import _spearman
from factor_matrix.calculation.services.permutation_diagnostics import (
    VARIANTS, draw_assignment, normalized_ranks, rank_correlation,
    replay_diagnostics, summarize_replay, verify_fixed_inputs, zero_center_counterexamples,
)
from factor_matrix.cli_parser import build_parser
from factor_matrix.storage import DataLake


def test_independent_rank_implementation_ties_and_immutable_inputs():
    left = np.array([3., 1., 1., 5., 2.])
    right = np.array([-2., 4., 1., 4., 0.])
    before = left.copy()
    assert rank_correlation(left, right) == pytest.approx(_spearman(left, right), abs=1e-15)
    assert normalized_ranks(left).sum() == pytest.approx(0., abs=1e-15)
    assert np.linalg.norm(normalized_ranks(left)) == pytest.approx(1.)
    np.testing.assert_array_equal(left, before)
    assert rank_correlation(np.ones(5), right) is None
    assert rank_correlation(left[:2], right[:2]) is None


def test_exact_counterexample_rejects_universal_zero_rank_center_claim():
    example = zero_center_counterexamples()
    time = example['within_asset_arithmetic_demeaning']
    assert time['enumerated_draws'] == 27
    assert time['asset_arithmetic_means'] == [0., 0., 0.]
    assert time['expected_unnormalized_covariance'] == pytest.approx(0., abs=1e-15)
    assert time['expected_spearman'] == pytest.approx(-1/9, abs=1e-15)
    assert example['date_axis_persistent_asset_order']['mean_spearman_after_swap'] == 1.


@pytest.mark.parametrize('strategy', VARIANTS)
def test_assignment_preserves_donors_and_matches_production_rng(strategy):
    source = np.arange(48, dtype=float).reshape(8, 6)
    source[::2, 1] = np.nan
    source[1::3, 4] = np.nan
    groups = [np.flatnonzero(np.isfinite(source[:, a])) for a in range(6)]
    assignment = draw_assignment(source, groups, np.random.default_rng(20260818), strategy)
    for a, group in enumerate(groups):
        np.testing.assert_array_equal(np.sort(assignment[assignment[:, a] >= 0, a]), group)
    if strategy == VARIANTS[0]:
        reference_rng = np.random.default_rng(20260818)
        for a, group in enumerate(groups):
            np.testing.assert_array_equal(assignment[group, a], reference_rng.permutation(group))
        np.testing.assert_array_equal(assignment >= 0, np.isfinite(source))
    else:
        order = assignment[:, 0]
        assert np.all(order != np.arange(8))
        np.testing.assert_array_equal(np.sort(order), np.arange(8))
        np.testing.assert_array_equal(assignment >= 0, np.isfinite(source[order]))
        for d in range(8):
            assert np.unique(assignment[d, assignment[d] >= 0]).tolist() == [order[d]]


@pytest.mark.parametrize('strategy', VARIANTS)
def test_independent_replay_and_exact_decomposition_on_unbalanced_panel(strategy):
    rng = np.random.default_rng(11)  # Synthetic fixture, never a research rerun.
    signal = rng.normal(size=(40, 8))
    labels = rng.normal(size=(40, 8))
    signal[::5, 2] = np.nan
    labels[::3, 4] = np.nan
    before_s, before_y = signal.copy(), labels.copy()
    config = {'seed':20260818, 'repetitions':20, 'lookback':20, 'primary_horizon':5}
    daily, repetitions, assets = replay_diagnostics(signal, labels, strategy=strategy,
        config=config, dates=[date(2020,1,1)+timedelta(days=i) for i in range(40)],
        assets=[f'a{i}' for i in range(8)], progress=lambda _: None)
    expected = permutation_means(signal, labels, strategy=strategy,
        seed=config['seed'], repetitions=config['repetitions'])
    result = summarize_replay(repetitions, daily, assets, expected, 1e-12)
    assert result['replay_matches_saved']
    assert result['max_replay_absolute_difference'] < 1e-14
    assert result['decomposition_max_absolute_error'] < 1e-14
    assert result['lag_decomposition_max_absolute_error'] < 1e-14
    counts = result['support_counts_over_all_repetitions']
    assert counts['donors_without_valid_signal'] > 0
    assert counts['n_valid'] == counts['baseline_cells']+counts['added_recipient_cells']-counts['dropped_recipient_cells']
    if strategy == VARIANTS[0]:
        assert counts['added_recipient_cells'] == counts['dropped_recipient_cells'] == 0
        assert result['max_absolute_centered_donor_mean'] < 1e-15
    else:
        assert counts['same_date_cells'] == 0
        assert counts['dropped_recipient_cells'] > 0
    np.testing.assert_array_equal(signal, before_s)
    np.testing.assert_array_equal(labels, before_y)
    assert repetitions['assignment_sha256'].n_unique() == 20
    assert daily.select((pl.col('overlap_cells') <= pl.col('n_valid')).all()).item()


def test_diagnostic_rejects_changed_input_before_any_artifact_write(tmp_path):
    config = {'fixed_files':{}, 'source_manifest':'source.json', 'source_manifest_sha256':'wrong'}
    (tmp_path/'source.json').write_text(json.dumps({}))
    with pytest.raises(ValueError, match='DIAGNOSTIC_PIN_MISMATCH'):
        verify_fixed_inputs(DataLake(tmp_path), config)
    assert not (tmp_path/'diagnostics').exists()


def test_diagnostic_cli_has_no_seed_window_or_threshold_override():
    parser = build_parser()
    args = parser.parse_args(['diagnose-probe-permutation-gates'])
    assert set(vars(args)) == {'command', 'data_root', 'config'}
    with pytest.raises(SystemExit):
        parser.parse_args(['diagnose-probe-permutation-gates', '--seed', '1'])
