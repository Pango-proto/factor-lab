from copy import deepcopy
from decimal import Decimal as D

import pytest

from factor_matrix.calculation.services.accounting import Lot
from factor_matrix.calculation.services.equity_execution import EquityAccountingEngine, clock, fees_for, next_weekdays
from factor_matrix.calculation.services.equity_fixture import build_equity_fixture, audit_engine, submit, execute, mark

DAYS = next_weekdays('2026-09-01T00:00:00+00:00', 12)
ASSET = 'FIXTURE.SH'


def engine(board='MAIN', cash='10000'):
    return EquityAccountingEngine(cash, instruments={ASSET: board}, sessions=DAYS)


@pytest.mark.parametrize('board,qty,expected', [('MAIN', 630, 'quantity_constraint'), ('MAIN', 690, 'quantity_constraint'),
    ('MAIN', 600, 'accepted'), ('CHINEXT', 630, 'quantity_constraint'), ('CHINEXT', 600, 'accepted'),
    ('STAR', 199, 'quantity_constraint'), ('STAR', 200, 'accepted'), ('STAR', 201, 'accepted')])
def test_board_buy_units(board, qty, expected):
    assert submit(engine(board), DAYS[0], 'b', ASSET, 'buy', qty, '10') == expected


@pytest.mark.parametrize('qty,expected', [(99, 'accepted'), (100, 'accepted'), (199, 'accepted'),
    (299, 'accepted'), (298, 'quantity_constraint'), (1, 'quantity_constraint')])
def test_main_odd_lot_remainder_must_be_sold_together(qty, expected):
    e = engine()
    e.lots.append(Lot(ASSET, 299, clock(DAYS[0], 'open'), 'corporate-action-lot'))
    assert submit(e, DAYS[0], 's', ASSET, 'sell', qty, '10') == expected


def test_odd_lot_reserved_cannot_be_sold_twice():
    e = engine()
    e.lots.append(Lot(ASSET, 299, clock(DAYS[0], 'open'), 'lot'))
    assert submit(e, DAYS[0], 's1', ASSET, 'sell', 99, '10') == 'accepted'
    assert submit(e, DAYS[0], 's2', ASSET, 'sell', 99, '10') == 'quantity_constraint'


@pytest.mark.parametrize('board,requested,expected', [('MAIN', 999, 900), ('STAR', 999, 999), ('STAR', 199, 0)])
def test_target_sizing_all_fees(board, requested, expected):
    assert engine(board).affordable_quantity(ASSET, D('10000'), D('10'), requested) == expected
    # Buying 100 at 10 needs 1005.01, not just the old commission-only 1005.
    assert engine(board).affordable_quantity(ASSET, D('1005'), D('10'), 100) == 0


def test_sell_stamp_and_transfer_hand_calculation():
    assert fees_for(D('6900'), 'sell') == {'commission': D('5'), 'stamp_duty': D('3.45'), 'transfer_fee': D('.07')}


def test_partial_fills_charge_minimum_and_round_components_once_per_order():
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 300, '10')
    assert execute(e, DAYS[1], 'b', '10', '10', capacity=151) == 'partial_fill'
    event = deepcopy(e.seen['execute-b-'+DAYS[1]])
    event.update(event_id='fill2', fill_id='fill2', capacity=149)
    assert e.apply(event) == 'filled'
    assert e.cash == D('6994.97')
    assert sum(f['amount'] for f in e.tables['fees'] if f['component'] == 'commission') == 5
    audit_engine(e)


def test_next_session_only_and_no_future_signal_atomic():
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '10')
    event = deepcopy(e.seen['submit-b'])
    event.update(event_id='future', order_id='future', signal_available_at=clock(DAYS[1]))
    before = e.export()
    with pytest.raises(ValueError, match='FUTURE_SIGNAL'):
        e.apply(event)
    assert e.export() == before
    event.update(signal_available_at=clock(DAYS[0]), earliest_execution=DAYS[0]+'T08:00:00+00:00')
    with pytest.raises(ValueError, match='NEXT_SESSION_OPEN'):
        e.apply(event)
    execute(e, DAYS[1], 'b', '10', '10')
    assert e.total(ASSET, clock(DAYS[1])) == 0
    assert e.total(ASSET, clock(DAYS[2], 'open')) == 100


def test_caller_cannot_forge_same_day_settlement():
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '10')
    other = deepcopy(e)
    execute(other, DAYS[1], 'b', '10', '10')
    event = deepcopy(other.seen['execute-b-'+DAYS[1]])
    event['sellable_at'] = event['time']
    with pytest.raises(ValueError, match='T_PLUS_ONE'):
        e.apply(event)
    assert not e.tables['fills']


def test_future_prices_do_not_change_orders_or_open_fills():
    original = engine()
    mark(original, DAYS[0], {ASSET:'10'})
    submit(original, DAYS[0], 'b', ASSET, 'buy', 100, '10.20')
    low, high = deepcopy(original), deepcopy(original)
    execute(low, DAYS[1], 'b', '10.05', '10')
    execute(high, DAYS[1], 'b', '10.05', '10')
    mark(low, DAYS[1], {ASSET:'9.50'}, opens={ASSET:'10.05'})
    mark(high, DAYS[1], {ASSET:'10.50'}, opens={ASSET:'10.05'})
    assert low.export()['orders'] == high.export()['orders']
    assert low.export()['fills'] == high.export()['fills']
    assert low.tables['daily_nav'][-1]['nav'] != high.tables['daily_nav'][-1]['nav']


def test_main_20_percent_drop_invalid_but_chinext_20_percent_valid():
    bar = dict(reference_close='10', normal_session=True, open='8', high='8', low='8', close='8')
    with pytest.raises(ValueError, match='BAR_OUTSIDE_PRICE_LIMIT'):
        engine().validate_bar(ASSET, bar)
    assert engine('CHINEXT').validate_bar(ASSET, bar) == (D('8'), D('12'))


def test_upper_limit_buy_blocked_and_execution_below_lower_rejected():
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '11')
    assert execute(e, DAYS[1], 'b', '11', '10') == 'price_limit_blocked'
    invalid = deepcopy(e.seen['execute-b-'+DAYS[1]])
    invalid.update(event_id='invalid', fill_id='invalid', price='8')
    with pytest.raises(ValueError, match='EXECUTION_OUTSIDE_PRICE_LIMIT'):
        e.apply(invalid)
    assert not e.tables['fills']


def test_missing_held_halt_quote_fails_and_valid_halt_marks_carry():
    e = engine()
    mark(e, DAYS[0], {ASSET:'10'})
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '10')
    execute(e, DAYS[1], 'b', '10', '10')
    mark(e, DAYS[1], {ASSET:'10'})
    with pytest.raises(ValueError, match='MISSING_HELD_ASSET_MARK'):
        mark(e, DAYS[2], {})
    mark(e, DAYS[2], {ASSET:'10'}, halted=(ASSET,))
    assert e.tables['positions'][-1]['stale'] is True
    assert e.tables['daily_nav'][-1]['nav'] == D('9994.99')


def test_external_withdrawal_is_neutral_and_insufficient_withdrawal_logged():
    e = engine()
    e.apply(dict(kind='external_flow', event_id='w', time=clock(DAYS[0]), amount='-1000'))
    mark(e, DAYS[0], {ASSET:'10'})
    assert e.tables['daily_nav'][-1]['external_flow'] == -1000
    from factor_matrix.calculation.services.backtest_path import path_functionals
    p = path_functionals(e.tables['daily_nav'], initial_nav='10000', initial_time=DAYS[0]+'T00:00:00+00:00')
    assert p['total_return'] == p['max_drawdown'] == 0
    assert e.apply(dict(kind='external_flow', event_id='w2', time=clock(DAYS[1]), amount='-10000')) == 'insufficient_cash_for_withdrawal'


def test_hand_calculated_integration_fixture_and_tamper_detection():
    report = build_equity_fixture()
    assert len(report['checks']) == 25
    assert D(report['path_metrics']['max_drawdown']) == D('0.032321')
    assert D(report['path_metrics']['total_return']) == D('0.057679')
    assert report['tables']['daily_nav'][-1]['nav'] == '12576.79'
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '10')
    execute(e, DAYS[1], 'b', '10', '10')
    e.tables['fees'][-1]['amount'] = D('1')
    with pytest.raises(AssertionError):
        audit_engine(e)


def test_next_day_sell_is_allowed_but_cannot_execute_on_purchase_day():
    e = engine()
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '10')
    execute(e, DAYS[1], 'b', '10', '10')
    assert e.total(ASSET, clock(DAYS[1])) == 0
    # A close signal may schedule a sale for tomorrow, when today's lot settles.
    assert submit(e, DAYS[1], 's', ASSET, 'sell', 100, '10') == 'accepted'
    probe = deepcopy(e)
    execute(probe, DAYS[2], 's', '10', '10')
    event = deepcopy(probe.seen['execute-s-'+DAYS[2]])
    event.update(event_id='same-day', fill_id='same-day', time=clock(DAYS[1]), quote_time=clock(DAYS[1]))
    before = e.export()
    with pytest.raises(ValueError, match='OPEN_EXECUTION_REQUIRED'):
        e.apply(event)
    assert e.export() == before
    assert execute(e, DAYS[2], 's', '10', '10') == 'filled'


def test_unsupported_dates_boards_and_special_sessions_fail_closed():
    with pytest.raises(ValueError, match='CALENDAR'):
        EquityAccountingEngine('10000', instruments={ASSET:'MAIN'}, sessions=['2001-01-01','2001-01-02'])
    with pytest.raises(ValueError, match='UNSUPPORTED_BOARD'):
        engine('ST')
    with pytest.raises(ValueError, match='SPECIAL_SESSION'):
        engine().validate_bar(ASSET, dict(normal_session=False))


def test_duration_v2_includes_t0_and_does_not_invent_recovery():
    from factor_matrix.calculation.services.backtest_path import path_functionals_v2
    rows = [{'time': clock(DAYS[1]), 'nav': '9000'}, {'time': clock(DAYS[2]), 'nav': '9500'}]
    metrics = path_functionals_v2(rows, initial_nav='10000', initial_time=clock(DAYS[0]))
    assert metrics['max_drawdown_peak_to_trough_observations'] == 1
    assert metrics['max_drawdown_trough_to_recovery_observations'] is None
    assert metrics['max_drawdown_peak_to_recovery_observations'] is None


def test_blocked_day_order_expires_and_releases_reservation():
    e = engine()
    mark(e, DAYS[0], {ASSET:'10'})
    submit(e, DAYS[0], 'b', ASSET, 'buy', 100, '11')
    assert execute(e, DAYS[1], 'b', '11', '10') == 'price_limit_blocked'
    assert e.reserved_cash > 0
    mark(e, DAYS[1], {ASSET:'11'})
    assert e.orders['b'].status == 'expired'
    assert e.orders['b'].reason == 'unfilled_session_order'
    assert e.reserved_cash == 0
