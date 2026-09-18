"""Chart preview v2 driven by the same strict, Decimal execution ledger as QA."""
import math
import random
from pathlib import Path

import numpy as np

from ...storage import DataLake, json_hash, source_tree_hash
from .accounting import ZERO, Decimal, money
from .backtest_path import path_functionals_v2
from .equity_execution import EquityAccountingEngine, POLICY, clock, next_weekdays
from .equity_fixture import audit_engine, build_equity_fixture, execute, mark, submit
from .strategy_fixture import plain

CONTRACT = {'id': 'backtest_chart_preview_v2', 'seed': 1729, 'sessions': 252,
            'purpose': 'synthetic_visual_demo_only', 'initial_cash': 100000,
            'execution_policy': POLICY, 'calendar': 'ordinal_synthetic_weekdays_not_exchange_calendar',
            'rebalance_sessions': 21, 'daily_price_generation': 'cent_rounded_log_shocks_bounded_inside_daily_limits',
            'scenarios': ['steady', 'stress', 'recovery']}


def build_preview_data():
    scenarios = []
    days = next_weekdays('2026-01-01T00:00:00+00:00', 254)
    for scenario_id, name in [('steady', '常态波动'), ('stress', '集中回撤'), ('recovery', '震荡恢复')]:
        rng = random.Random(CONTRACT['seed'])
        shocks = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(CONTRACT['sessions'])]
        paths = []
        for index, (sid, title, color, exposure_base) in enumerate([
            ('a', '合成路径 A', '#346779', .8), ('b', '合成路径 B', '#d89da6', .95),
            ('c', '合成路径 C', '#8a7199', .45)]):
            asset = 'FIXTURE.SH'
            engine = EquityAccountingEngine('100000', instruments={asset: 'MAIN'}, sessions=days)
            price = Decimal('20.00')
            mark(engine, days[0], {asset: price})
            prices = [price]
            for day, noise in enumerate(shocks, 1):
                prior_price = price
                oid = str(day)
                # Target/limit are frozen using only the previous close and NAV.
                # An adverse opening gap can miss the limit; do not resize using future data.
                if (day-1) % 21 == 0:
                    exposure = Decimal(str(max(.15, min(.95, exposure_base + .12*math.sin(day/23+index)))))
                    target = int(engine.tables['daily_nav'][-1]['nav']*exposure/prior_price/100)*100
                    change = target-engine.total(asset)
                    if change:
                        side = 'buy' if change > 0 else 'sell'
                        limit = money(prior_price * (Decimal('1.02') if side == 'buy' else Decimal('.98')))
                        qty = engine.affordable_quantity(asset, engine.cash-engine.reserved_cash, limit, change) if change > 0 else -change
                        if qty:
                            submit(engine, days[day-1], oid, asset, side, qty, limit)
                volatility = .007 if not 87 <= day <= 138 else .019
                drift = .00065
                if scenario_id == 'stress' and 95 <= day <= 115: drift -= .010
                if scenario_id == 'recovery':
                    if 60 <= day <= 85: drift -= .009
                    if 135 <= day <= 180: drift += .004
                ret = drift+volatility*(.72*noise[0]+.69*noise[index+1])
                # Synthetic generation specification, not window/seed selection.
                lower, upper = money(prior_price*Decimal('.91')), money(prior_price*Decimal('1.09'))
                price = min(upper, max(lower, money(str(float(prior_price)*math.exp(ret)))))
                opening = min(upper, max(lower, money(str(float(prior_price)*math.exp(.002*noise[index+1])))))
                if oid in engine.orders:
                    execute(engine, days[day], oid, opening, prior_price)
                    if engine.orders[oid].status == 'open':
                        engine.apply({'kind': 'cancel', 'event_id': 'cancel-'+oid, 'order_id': oid, 'time': clock(days[day], 'open')})
                mark(engine, days[day], {asset: price}, opens={asset: opening})
                prices.append(price)
            audit_engine(engine)
            navs = engine.tables['daily_nav']
            metrics = path_functionals_v2(navs[1:], initial_nav='100000', initial_time=clock(days[0]))
            rows = []
            for i, (account, close) in enumerate(zip(navs, prices)):
                m = metrics['curve'][i-1] if i else None
                positions = [p for p in engine.tables['positions'] if p['date'] == days[i]]
                shares = sum(p['quantity'] for p in positions)
                rows.append({'day': i, 'nav': float(account['nav']), 'wealth': float(m['unit_wealth']) if m else 1,
                    'return': float(m['period_return']) if m else 0, 'drawdown': -float(m['drawdown']) if m else 0,
                    'price': float(close), 'shares': shares, 'cash': float(account['cash']),
                    'market_value': float(account['market_value']), 'receivables': float(account['receivables']),
                    'cash_stack_top': float(account['market_value']+account['cash']),
                    'nav_stack_top': float(account['nav'])})
            trades = []
            for f in engine.tables['fills']:
                fees = {r['component']: float(r['amount']) for r in engine.tables['fees'] if r['fill_id'] == f['fill_id']}
                o = engine.orders[f['order_id']]
                trades.append({'day': days.index(f['time'][:10]), 'side': f['side'], 'quantity': f['quantity'],
                               'price': float(f['price']), 'decision_time': o.decision_time, 'fill_time': f['time'], **fees})
            returns = np.array([r['return'] for r in rows[1:]])
            edges = np.linspace(-.06, .06, 33)
            count, _ = np.histogram(returns, edges)
            if count.sum() != len(returns):
                raise ValueError('DISPLAY_HISTOGRAM_SUPPORT_EXCEEDED')
            density = [{'x': float((edges[i]+edges[i+1])/2), 'y': float(count[i]/len(returns)/(edges[i+1]-edges[i]))} for i in range(len(count))]
            absolute = np.abs(returns)
            acf = [{'x': lag, 'y': float(np.corrcoef(absolute[:-lag], absolute[lag:])[0,1])} for lag in range(1,21)]
            paths.append({'id': sid, 'name': title, 'color': color, 'rows': rows, 'trades': trades,
                'density': density, 'acf': acf, 'summary': {'final_nav': rows[-1]['nav'],
                'total_return': float(metrics['total_return']), 'max_drawdown': float(metrics['max_drawdown']),
                'fills': len(trades), 'commission': sum(t['commission'] for t in trades),
                'total_fees': float(sum((f['amount'] for f in engine.tables['fees']), ZERO)),
                'underwater_sessions': metrics['longest_underwater_observations']}})
        correlation = np.corrcoef([[r['return'] for r in p['rows'][1:]] for p in paths]).tolist()
        scenarios.append({'id': scenario_id, 'name': name, 'series': paths, 'correlation': correlation})
    fixture = build_equity_fixture()
    return {'schema_version': 2, 'contract': CONTRACT, 'research_status': 'not_eligible',
            'real_market_data': False, 'risk_basis_id': None, 'risk_set_version': None,
            'parent_run_ids': [], 'scenarios': scenarios, 'acceptance_fixture': fixture}


def publish_preview(output_dir: Path) -> Path:
    lake = DataLake(output_dir)
    payload = build_preview_data()
    path = lake.write_immutable_json(output_dir/'preview.json', payload)
    return lake.write_immutable_json(output_dir/'_MANIFEST.json', {
        'schema_version': 2, 'run_id': CONTRACT['id']+'_'+json_hash(CONTRACT)[:16],
        'status': 'visual_demo_only', 'definition': CONTRACT, 'code_hash': source_tree_hash(),
        'risk_basis_id': None, 'risk_set_version': None, 'parent_run_ids': [],
        'sample_axis': 't0,D001..D252', 'real_market_data': False, 'research_promoted': False,
        'outputs': {'preview': lake.artifact_record(path)}})
