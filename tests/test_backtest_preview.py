import json

import pytest

from factor_matrix.calculation.services.backtest_preview import build_preview_data,publish_preview


def test_preview_is_deterministic_and_does_not_claim_research_or_real_data():
    first=build_preview_data()
    assert first==build_preview_data()
    assert first['real_market_data'] is False
    assert first['research_status']=='not_eligible'
    assert first['risk_basis_id'] is None
    assert len(first['scenarios'])==3


def test_preview_cash_holdings_nav_and_commissions_reconcile():
    for scenario in build_preview_data()['scenarios']:
        for series in scenario['series']:
            cash,shares=100000.,0
            trades={t['day']:t for t in series['trades']}
            for row in series['rows']:
                if row['day'] in trades:
                    t=trades[row['day']]
                    delta=t['quantity']*(1 if t['side']=='buy' else -1)
                    shares+=delta
                    cash-=delta*t['price']+t['commission']
                # Display trade prices have four decimal digits.
                assert abs(cash-row['cash'])<1
                assert shares==row['shares']
                assert row['nav']==pytest.approx(row['cash']+shares*row['price'])
                assert row['cash']>=0
            peak=100000
            dd=[]
            for r in series['rows']:
                peak=max(peak,r['nav']);dd.append(1-r['nav']/peak)
                assert -r['drawdown']==pytest.approx(dd[-1])
            assert series['summary']['max_drawdown']==pytest.approx(max(dd))


def test_preview_statistics_share_the_same_paths():
    import numpy as np
    for s in build_preview_data()['scenarios']:
        matrix=np.array(s['correlation'])
        assert np.allclose(matrix,matrix.T)
        assert np.allclose(np.diag(matrix),1)
        for series in s['series']:
            # Fixed support is wide enough for the declared synthetic scenarios.
            assert sum(p['y'] for p in series['density'])*(.12/32)==pytest.approx(1)
            assert len(series['rows'])==252


def test_preview_publication_is_isolated_and_immutable(tmp_path):
    manifest=publish_preview(tmp_path/'preview')
    assert publish_preview(tmp_path/'preview')==manifest
    record=json.loads(manifest.read_text())
    assert record['status']=='visual_demo_only'
    assert record['research_promoted'] is False
    assert not list(tmp_path.rglob('_CURRENT.json'))
