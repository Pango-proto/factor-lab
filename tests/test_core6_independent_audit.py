import numpy as np
import pytest

from factor_matrix.calculation.services.core6_independent_audit import metric_statistics, residual_moments


def test_independent_metric_detects_wrong_projection_metric():
    # Explicit orthogonal vector for nonuniform W, no production transform.
    w = np.array([1., 2., 3., 4.])
    controls = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    y = np.array([2., -1., 8., -6.])
    y /= np.sqrt(np.average(y*y, weights=w))
    result = metric_statistics(y, controls, w)
    assert abs(result['mean_w']) < 1e-14
    assert result['sd_w'] == pytest.approx(1.)
    assert result['max_abs_control_corr'] < 1e-14
    assert metric_statistics(y, controls, np.ones(4))['max_abs_control_corr'] > .01


@pytest.mark.parametrize('weights', [[1., 0.], [1., -1.], [1., np.nan]])
def test_invalid_metric_never_passes(weights):
    with pytest.raises(ValueError, match='WEIGHT'):
        metric_statistics([1., -1.], [[1.], [0.]], weights)


def test_signed_cross_product_and_variance_identity():
    positive = residual_moments([.02, .02], [.0004, .0004])
    negative = residual_moments([.02, -.02], [.0004, .0004])
    assert positive['diagonal_second_moment'] == pytest.approx(1.)
    assert positive['cross_second_moment'] == pytest.approx(1.)
    assert positive['z_squared'] == pytest.approx(2.)
    assert negative['cross_second_moment'] == pytest.approx(-1.)
    assert negative['z_squared'] == 0.
    assert positive['identity_error'] < 1e-14


def test_missing_outcome_does_not_shrink_or_zero_fill():
    result = residual_moments([.02, np.nan], [.0004, .0004])
    assert result['status'] == 'missing_outcome'
    assert result['assets'] == 2 and result['z'] is None


def test_invalid_forecast_is_not_silently_dropped():
    with pytest.raises(ValueError, match='INVALID_FORECAST'):
        residual_moments([.02, .01], [.0004, 0.])


def test_audit_requires_pinned_config_hash(tmp_path):
    from factor_matrix.calculation.services.core6_independent_audit import run_independent_audit
    config = tmp_path/'config.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='CONFIG_HASH'):
        run_independent_audit(project_root=tmp_path, config_path=config,
                              config_sha256='incorrect', output_root=tmp_path/'outputs')
    assert not (tmp_path/'outputs').exists()
