from copy import deepcopy
from decimal import Decimal as D
import numpy as np
import pytest
from factor_matrix.calculation.services.backtest_preview_v3 import build_preview_data, build_market, target_exposure
from factor_matrix.calculation.services.backtest_path import path_functionals_v3
from factor_matrix.calculation.services.equity_execution import EquityAccountingEngine, clock, next_weekdays
from factor_matrix.calculation.services.equity_fixture import execute, mark, submit
from factor_matrix.calculation.services.execution_invariants import audit_execution_path
from factor_matrix.calculation.services.preview_statistics import describe_returns, price_diagnostics

@pytest.fixture(scope='module')
def preview():
    return build_preview_data()


def test_market_shared_and_pinned(preview):
    for scenario in preview['scenarios']:
        prices=[float(b['close']) for b in scenario['market']]
        assert prices==[float(b['close']) for b in build_market(scenario['id'])]
        assert len(scenario['series'])==4
        for series in scenario['series']:
            assert [r['price'] for r in series['rows']]==prices
            assert series['rows'][0]['cash']==100000
            assert series['rows'][0]['market_value']==series['rows'][0]['shares']==0
            assert all(t['day']>=1 for t in series['trades'])
            assert series['invariants']['status']=='passed'
            assert series['invariants']['sessions']==252
        assert scenario['series'][-1]['id']=='benchmark'
        assert len(scenario['series'][-1]['trades'])==1
        assert scenario['series'][-1]['trades'][0]['day']==1
        assert scenario['invariants']['strategy_sessions']==756


def test_every_path_exact_cash_nav_and_statistics(preview):
    for scenario in preview['scenarios']:
        returns=[]
        for s in scenario['series']:
            cash,shares=D('100000'),0
            trades={t['day']:t for t in s['trades']}
            peak=100000.;terminal=longest=0
            for row in s['rows']:
                if row['day'] in trades:
                    t=trades[row['day']]
                    delta=t['quantity']*(1 if t['side']=='buy' else -1)
                    shares+=delta
                    cash-=delta*D(str(t['price']))+sum(D(str(t[k])) for k in ['commission','stamp_duty','transfer_fee'])
                    assert t['decision_time'][:10]<t['fill_time'][:10]
                assert cash==D(str(row['cash']))>=0
                assert shares==row['shares']
                assert cash+shares*D(str(row['price']))==D(str(row['nav']))
                if row['nav']>=peak:peak=row['nav'];terminal=0
                else:terminal+=1
                longest=max(longest,terminal)
            assert s['summary']['terminal_underwater_sessions']==terminal
            assert s['summary']['underwater_sessions']==longest
            assert s['summary']['right_censored']==(terminal>0)
            r=[row['return'] for row in s['rows'][1:]]
            returns.append(r)
            assert s['summary']['annualized_volatility']==pytest.approx(np.std(r,ddof=1)*np.sqrt(252))
            assert s['summary']['sharpe']==pytest.approx(np.mean(r)/np.std(r,ddof=1)*np.sqrt(252))
            assert sum(p['y'] for p in s['statistics']['histogram'][:-1])*.0025==pytest.approx(1)
        assert np.allclose(np.corrcoef(returns),scenario['correlation'])


def test_unfilled_pressure_is_visible_and_has_no_fill(preview):
    stress=next(s for s in preview['scenarios'] if s['id']=='stress')
    assert sum(len(s['unfilled']) for s in stress['series'])>0
    for s in stress['series']:
        for missing in s['unfilled']:
            assert missing['day'] not in [t['day'] for t in s['trades']]
            assert missing['reason'] in ('halted','outside_order_limit','price_limit_blocked')


def test_statistics_objects_and_lag_bands(preview):
    for s in preview['scenarios']:
        p=[float(b['close']) for b in s['market']]
        stats=s['price_diagnostics']
        returns=np.diff(p)/p[:-1]
        assert np.allclose([r['y'] for r in stats['returns']],returns)
        assert stats['acf_reference']==pytest.approx(1.96/np.sqrt(252))
        assert stats['rolling_volatility'][0]['x']==20
        assert stats['rolling_volatility'][0]['y']==pytest.approx(np.std(returns[:20],ddof=1))
        altered=p[:];altered[-1]*=1.01
        assert price_diagnostics(altered)['rolling_volatility'][:-1]==stats['rolling_volatility'][:-1]


def test_tail_statistics_and_zero_variance():
    r=np.linspace(-.025,.025,101)
    stats=describe_returns(r)
    assert stats['quantile_05']==pytest.approx(-.0225)
    assert stats['cvar_95_loss']==pytest.approx(.02375)
    assert stats['skewness']==pytest.approx(0,abs=1e-12)
    zero=describe_returns([0]*252)
    assert zero['normal']==[] and zero['sharpe'] is None


@pytest.mark.parametrize('timing,expected',[('period_end',D('.057679')),('period_start',D('.042263804556732'))])
def test_two_cash_flow_conventions_hand_calculated(timing,expected):
    days=next_weekdays('2026-09-01T00:00:00+00:00',3)
    metric=path_functionals_v3([{'time':clock(days[1]),'nav':'12576.79','external_flow':'2000'}],initial_nav='9676.79',initial_time=clock(days[0]),flow_timing=timing)
    assert abs(D('.967679')*(1+metric['total_return'])-1-expected)<D('1e-12')
    e=EquityAccountingEngine('10000',instruments={'FIXTURE.SH':'MAIN'},sessions=days,flow_timing=timing)
    at=clock(days[0]) if timing=='period_end' else days[0]+'T01:29:00+00:00'
    e.apply({'kind':'external_flow','event_id':'deposit','time':at,'amount':'2000'})
    mark(e,days[0],{'FIXTURE.SH':'10'})
    assert e.cash==D('12000')
    assert e.tables['daily_nav'][0]['external_flow']==D('2000')
    assert e.tables['cash_ledger'][0]['time']==at


def test_invariant_tampering_fails_closed():
    days=next_weekdays('2026-09-01T00:00:00+00:00',3)
    e=EquityAccountingEngine('10000',instruments={'FIXTURE.SH':'MAIN'},sessions=days)
    mark(e,days[0],{'FIXTURE.SH':'10'})
    submit(e,days[0],'buy','FIXTURE.SH','buy',100,'10')
    execute(e,days[1],'buy','10','10')
    mark(e,days[1],{'FIXTURE.SH':'10'})
    assert audit_execution_path(e)['status']=='passed'
    damaged=deepcopy(e);damaged.seen['execute-buy-'+days[1]]['halted']=True
    with pytest.raises(ValueError,match='NO_HALTED_FILL'):audit_execution_path(damaged)
    damaged=deepcopy(e);damaged.orders['buy'].quantity=101
    with pytest.raises(ValueError,match='BUY_ORDER_UNIT'):audit_execution_path(damaged)
    damaged=deepcopy(e);damaged.tables['fills'][0]['sellable_at']=clock(days[1],'open')
    with pytest.raises(ValueError,match='T1_LOCK'):audit_execution_path(damaged)


def test_declared_strategy_rules():
    assert target_exposure('a',[D('20')])==D('.8')
    assert target_exposure('c',[D('20')])==D('.4')
    assert target_exposure('b',[D('20')]*19)==D('.2')
    assert target_exposure('b',[D('20')]*19+[D('21')])==D('.8')
    assert target_exposure('b',[D('20')]*19+[D('19')])==D('.2')
