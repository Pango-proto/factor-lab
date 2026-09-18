import pytest

from factor_matrix.calculation.l1 import (
    apply_alpha_direction, weighted_residualize, weighted_scale_only,
)


def test_redundant_country_and_all_industry_dummies_do_not_break_projection() -> None:
    controls = (
        (1.0, 1.0, 0.0, -1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, -1.0),
        (1.0, 0.0, 1.0, 1.0),
    )
    residuals = weighted_residualize(
        (1.0, 3.0, 2.0, 6.0), controls, (1.0, 2.0, 1.0, 2.0)
    )
    for column in range(4):
        assert sum(
            weight * row[column] * residual
            for weight, row, residual in zip((1.0, 2.0, 1.0, 2.0), controls, residuals)
        ) == pytest.approx(0.0, abs=1e-10)


def test_direction_is_applied_once_before_scale_only() -> None:
    directed = apply_alpha_direction((1.0, -2.0), -1)
    assert directed == (-1.0, 2.0)
    scaled = weighted_scale_only(directed, (1.0, 1.0))
    assert scaled[0] < 0 < scaled[1]
