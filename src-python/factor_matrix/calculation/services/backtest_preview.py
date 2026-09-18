"""Fixed synthetic data for a presentation prototype, isolated from research.

No market data, fitted alpha, risk basis, registry, or CURRENT is consumed.
All plotted financial/statistical values are computed here, never in React.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ...storage import DataLake, json_hash, source_tree_hash
from .backtest_path import path_functionals
from .strategy_fixture import plain

CONTRACT = {
    'id': 'backtest_chart_preview_v1', 'seed': 1729, 'sessions': 252,
    'purpose': 'synthetic_visual_demo_only', 'initial_cash': 100000,
    'commission_rate': .0001854, 'minimum_commission': 5,
    'execution': 'deterministic_21_session_rebalance_at_previous_price_no_slippage',
    'calendar': 'D001_to_D252_ordinal_synthetic_sessions_not_market_dates',
    'scenarios': ['steady', 'stress', 'recovery'],
}


def build_preview_data() -> dict:
    scenarios = []
    for scenario_id, name in [('steady', '常态波动'), ('stress', '集中回撤'), ('recovery', '震荡恢复')]:
        rng = random.Random(CONTRACT['seed'])
        shocks = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(CONTRACT['sessions'])]
        paths = []
        for asset, (series_id, series_name, color, base_exposure) in enumerate([
            ('a', '合成路径 A', '#346779', .8), ('b', '合成路径 B', '#d89da6', .95),
            ('c', '合成路径 C', '#8a7199', .45),
        ]):
            cash, shares, price, prior_nav = 100000., 0, 20., 100000.
            rows, trades, path_rows = [], [], []
            commission_total = 0.
            for day, noise in enumerate(shocks, 1):
                if (day-1) % 21 == 0:
                    exposure = max(.15, min(.95, base_exposure + .12 * math.sin(day/23 + asset)))
                    target = int((cash + shares * price) * exposure / price / 100) * 100
                    delta = target-shares
                    if delta:
                        commission = round(max(abs(delta)*price*.0001854, 5), 2)
                        cash -= delta*price + commission
                        shares = target
                        commission_total += commission
                        trades.append({'day': day, 'side': 'buy' if delta > 0 else 'sell',
                                       'quantity': abs(delta), 'price': round(price, 4), 'commission': commission})
                volatility = .007 if not 87 <= day <= 138 else .019
                drift = .00065
                if scenario_id == 'stress' and 95 <= day <= 115:
                    drift -= .010
                if scenario_id == 'recovery':
                    if 60 <= day <= 85: drift -= .009
                    if 135 <= day <= 180: drift += .004
                ret = drift + volatility*(.72*noise[0]+.69*noise[asset+1])
                price *= math.exp(ret)
                nav = cash + shares * price
                time = (datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)).isoformat()
                path_rows.append({'time': time, 'nav': str(nav)})
                rows.append({'day': day, 'nav': nav, 'wealth': nav/100000,
                             'return': nav/prior_nav-1, 'price': price,
                             'shares': shares, 'cash': cash})
                prior_nav = nav
            metrics = plain(path_functionals(path_rows, initial_nav='100000', initial_time='2001-01-01T00:00:00+00:00'))
            for row, metric in zip(rows, metrics['curve']):
                row['drawdown'] = -float(metric['drawdown'])
            returns = np.array([r['return'] for r in rows])
            edges = np.linspace(-.06, .06, 33)
            count, _ = np.histogram(returns, edges)
            density = [{'x': float((edges[i]+edges[i+1])/2), 'y': float(count[i]/len(returns)/(edges[i+1]-edges[i]))} for i in range(len(count))]
            absolute = np.abs(returns)
            acf = [{'x': lag, 'y': float(np.corrcoef(absolute[:-lag], absolute[lag:])[0, 1])} for lag in range(1, 21)]
            paths.append({'id': series_id, 'name': series_name, 'color': color, 'rows': rows,
                          'trades': trades, 'density': density, 'acf': acf,
                          'summary': {'final_nav': rows[-1]['nav'], 'total_return': float(metrics['total_return']),
                                      'max_drawdown': float(metrics['max_drawdown']), 'fills': len(trades),
                                      'commission': commission_total,
                                      'underwater_sessions': metrics['longest_underwater_observations']}})
        correlation = np.corrcoef([[r['return'] for r in p['rows']] for p in paths]).tolist()
        scenarios.append({'id': scenario_id, 'name': name, 'series': paths, 'correlation': correlation})
    return {'schema_version': 1, 'contract': CONTRACT, 'research_status': 'not_eligible',
            'real_market_data': False, 'risk_basis_id': None, 'risk_set_version': None,
            'parent_run_ids': [], 'scenarios': scenarios}


def publish_preview(output_dir: Path) -> Path:
    lake = DataLake(output_dir)
    payload = build_preview_data()
    data_path = lake.write_immutable_json(output_dir/'preview.json', payload)
    return lake.write_immutable_json(output_dir/'_MANIFEST.json', {
        'schema_version': 1, 'run_id': 'backtest_chart_preview_v1_'+json_hash(CONTRACT)[:16],
        'status': 'visual_demo_only', 'purpose': CONTRACT['purpose'],
        'definition': CONTRACT, 'code_hash': source_tree_hash(),
        'risk_basis_id': None, 'risk_set_version': None, 'parent_run_ids': [],
        'sample_axis': 'D001..D252', 'real_market_data': False, 'research_promoted': False,
        'outputs': {'preview': lake.artifact_record(data_path)},
    })
