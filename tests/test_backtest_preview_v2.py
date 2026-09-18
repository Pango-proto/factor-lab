from decimal import Decimal as D
import pytest
import numpy as np

from factor_matrix.calculation.services.backtest_preview_v2 import build_preview_data

@pytest.fixture(scope='module')
def preview():
    return build_preview_data()


def test_v2_preview_exact_trade_cash_reconciliation(preview):
    assert preview['schema_version'] == 2 and preview['real_market_data'] is False
    assert preview['research_status'] == 'not_eligible'
    for scenario in preview['scenarios']:
        for series in scenario['series']:
            cash, shares = D('100000'), 0
            trades = {t['day']:t for t in series['trades']}
            peak, max_dd = D('100000'), D(0)
            assert series['rows'][0]['day'] == 0
            assert series['rows'][0]['wealth'] == 1
            assert series['rows'][0]['drawdown'] == 0
            for row in series['rows']:
                t = trades.get(row['day'])
                if t:
                    assert t['quantity'] % 100 == 0
                    assert t['decision_time'][:10] < t['fill_time'][:10]
                    delta = t['quantity']*(1 if t['side']=='buy' else -1)
                    shares += delta
                    cash -= delta*D(str(t['price']))+sum(D(str(t[k])) for k in ('commission','stamp_duty','transfer_fee'))
                assert cash == D(str(row['cash']))
                assert shares == row['shares']
                nav = cash + shares*D(str(row['price']))
                assert nav == D(str(row['nav'])) == D(str(row['nav_stack_top']))
                assert D(str(row['market_value'])) + D(str(row['cash'])) + D(str(row['receivables'])) == nav
                peak = max(peak, nav)
                max_dd = max(max_dd, 1-nav/peak)
                assert -row['drawdown'] == pytest.approx(float(1-nav/peak))
            assert series['summary']['max_drawdown'] == pytest.approx(float(max_dd))


def test_statistics_exclude_t0_and_match_the_same_252_returns(preview):
    for scenario in preview['scenarios']:
        returns = [[r['return'] for r in s['rows'][1:]] for s in scenario['series']]
        assert np.allclose(np.corrcoef(returns), scenario['correlation'])
        for series in scenario['series']:
            assert len(series['rows']) == 253
            assert sum(p['y'] for p in series['density'])*(.12/32) == pytest.approx(1)
            absolute = np.abs([r['return'] for r in series['rows'][1:]])
            assert series['acf'][0]['y'] == pytest.approx(np.corrcoef(absolute[:-1],absolute[1:])[0,1])


def test_preview_acceptance_report_is_not_research(preview):
    fixture = preview['acceptance_fixture']
    assert fixture['audit']['status'] == 'passed'
    assert len(fixture['checks']) == 25
    assert D(fixture['path_metrics']['total_return']) == D('.057679')
