import numpy as np
import pytest

from factor_matrix.calculation.l2.estimation_probe import (
    PermutationDay,
    repeated_within_asset_time_shuffle_mean_ics,
)
from factor_matrix.calculation.l2.estimation_panel import _rank, _residualize


def test_vectorized_rank_preserves_average_tie_ranks() -> None:
    assert _rank(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == pytest.approx(
        [3.0, 0.5, 0.5, 2.0]
    )


def test_cached_residualization_matches_least_squares_projection() -> None:
    exposures = np.array([[1.0, -1.0], [2.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    values = np.array([2.0, 1.0, 3.0, 7.0])
    expected = values - exposures @ np.linalg.lstsq(exposures, values, rcond=None)[0]
    first = _residualize(values, exposures)[0]
    second = _residualize(values * 2.0, exposures)[0]
    assert first == pytest.approx(expected)
    assert second == pytest.approx(expected * 2.0)


def test_repeated_time_shuffle_is_seeded_and_preserves_axis() -> None:
    days = [
        PermutationDay(
            asset_ids=np.array(["A", "B", "C"]),
            labels=np.array([0.1, 0.2, 0.3]),
            exposures=np.ones((3, 1)),
            signal_residual=np.array([-1.0, 0.0, 1.0]),
        ),
        PermutationDay(
            asset_ids=np.array(["A", "B", "C"]),
            labels=np.array([0.3, 0.1, 0.2]),
            exposures=np.ones((3, 1)),
            signal_residual=np.array([1.0, -1.0, 0.0]),
        ),
    ]
    first = repeated_within_asset_time_shuffle_mean_ics(days, repetitions=20, seed=7)
    second = repeated_within_asset_time_shuffle_mean_ics(days, repetitions=20, seed=7)
    assert first == second
    assert len(first) == 20
