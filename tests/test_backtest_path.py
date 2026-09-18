from decimal import Decimal as D

import pytest

from factor_matrix.calculation.services.backtest_path import path_functionals, capacity_diagnostics, off_policy_evaluation


def time(day):
    return f"2001-01-{day:02}T16:00:00+00:00"


def path(values, flows=None):
    rows = [{"time": time(i+2), "nav": str(v), "external_flow": str((flows or {}).get(i, 0))} for i,v in enumerate(values)]
    return path_functionals(rows, initial_nav="100", initial_time=time(1))


def test_same_terminal_return_can_have_different_path_functionals():
    smooth, volatile = path([105,110,120]), path([120,80,120])
    assert smooth['total_return'] == volatile['total_return'] == D('.2')
    assert smooth['max_drawdown'] == 0
    assert abs(volatile['max_drawdown'] - D(1)/3) < D('1e-26')
    assert volatile['max_drawdown_peak_time'] == time(2)
    assert volatile['max_drawdown_trough_time'] == time(3)
    assert volatile['max_drawdown_recovery_time'] == time(4)
    assert volatile['log_quadratic_variation'] > smooth['log_quadratic_variation']


def test_initial_capital_is_included_in_first_day_drawdown():
    result = path([90,80])
    assert result['max_drawdown'] == D('.2')
    assert result['max_drawdown_peak_time'] == time(1)
    assert result['max_drawdown_recovery_time'] is None
    assert result['terminal_underwater_observations'] == 2


def test_external_deposit_is_not_a_return_or_drawdown_recovery():
    result = path([90,190,209], {1:100})
    assert result['curve'][1]['period_return'] == 0
    assert result['total_return'] == D('-.01')
    assert result['max_drawdown'] == D('.1')
    assert result['max_drawdown_recovery_time'] is None


@pytest.mark.parametrize('values', [[], [0], [-1], ['NaN']])
def test_invalid_nav_not_silently_clipped(values):
    with pytest.raises(ValueError):path(values)


def test_duplicate_timestamp_is_not_an_extra_observation():
    with pytest.raises(ValueError):
        path_functionals([dict(time=time(2),nav='100'),dict(time=time(2),nav='101')],initial_nav='100',initial_time=time(1))


def liquidity(**kwargs):
    return dict(asset_id='600000.SH',decision_time=time(3),requested_quantity=1000,reference_price='10',
                adv_shares='10000',adv_observations=20,required_observations=20,
                adv_available_at=time(2),adv_last_observation=time(2),**kwargs)


def test_capacity_is_share_based_participation_bound_not_economic_aum():
    result=capacity_diagnostics([liquidity()],participation='.01',lot_size=100)
    row=result['rows'][0]
    assert row['one_session_capacity_shares']==100
    assert row['estimated_sessions_at_constant_adv']==10
    assert row['quantity_shortfall']==900
    assert result['economic_capacity_cny'] is None and result['impact_calibration_status']=='unavailable'


@pytest.mark.parametrize('field,value,status', [('adv_shares',None,'insufficient_liquidity_history'),
    ('adv_observations',19,'insufficient_liquidity_history'),('adv_shares','0','no_observed_liquidity')])
def test_missing_liquidity_is_unavailable_not_free_capacity(field,value,status):
    row=liquidity();row[field]=value
    result=capacity_diagnostics([row],participation='.01',lot_size=100)['rows'][0]
    assert result['status']==status and result['one_session_capacity_shares'] is None


def test_capacity_rejects_future_volume():
    row=liquidity();row['adv_last_observation']=time(3)
    with pytest.raises(ValueError,match='FUTURE_LIQUIDITY'):
        capacity_diagnostics([row],participation='.01',lot_size=100)


def step(action, reward, target=None, behavior=None):
    return dict(action=action,reward=str(reward),target_probabilities=target or ['.75','.25'],
                behavior_probabilities=behavior or ['.5','.5'])


def test_supported_one_step_ope_matches_known_exact_population():
    # Uniform logging covers both actions equally; target value .75*2+.25*4=2.5.
    result=off_policy_evaluation([[step(0,2)],[step(1,4)]],assumptions_verified=True)
    assert result['trajectory_is']==result['per_decision_is']==result['weighted_is']==D('2.5')
    assert result['effective_sample_size']==D('1.6')
    assert not result['policy_selection_authorized']


def test_sequential_is_multiplies_propensities_and_pdis_uses_prefixes():
    result=off_policy_evaluation([[step(0,2),step(1,4)]],assumptions_verified=True)
    assert result['trajectory_is']==D('4.5')  # .75 * (2+4)
    assert result['per_decision_is']==D('6')  # 1.5*2 + .75*4
    assert result['weighted_is']==D('6')


def test_deterministic_logging_cannot_evaluate_unsupported_action():
    result=off_policy_evaluation([[step(0,2,behavior=['1','0'])]],assumptions_verified=True)
    assert result['status']=='not_estimable' and result['reason']=='TARGET_POLICY_OUTSIDE_BEHAVIOR_SUPPORT'


def test_real_backtest_without_propensities_and_verified_assumptions_is_not_ope():
    assert off_policy_evaluation([[dict(reward='1')]])['estimate'] is None
    assert off_policy_evaluation([[dict(reward='1')]],assumptions_verified=True)['reason']=='MISSING_LOGGED_PROPENSITIES'
