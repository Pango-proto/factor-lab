"""Versioned synthetic acceptance cases, exact hand-ledger expectations."""
from pathlib import Path
from decimal import Decimal

from ...storage import DataLake, json_hash, source_tree_hash
from .accounting import ZERO, decimal, money
from .equity_execution import EquityAccountingEngine, POLICY, clock, next_weekdays
from .backtest_path import path_functionals_v2
from .strategy_fixture import plain

DEFINITION = {'id': 'equity_execution_fixture_v2', 'initial_cash': '10000.00',
              'sessions': next_weekdays('2026-09-01T00:00:00+00:00', 12),
              'instruments': {'FIXTURE.SH': 'MAIN', 'FIXTURE.SZ': 'MAIN'},
              'real_market_data': False, 'seed': None, 'policy': POLICY}


def submit(engine, day, oid, asset, side, qty, limit):
    return engine.apply({'kind': 'submit', 'event_id': 'submit-'+oid, 'order_id': oid,
        'time': clock(day), 'decision_time': clock(day), 'signal_available_at': clock(day),
        'earliest_execution': clock(engine._next_day(day), 'open'), 'asset_id': asset,
        'side': side, 'quantity': qty, 'limit_price': str(limit)})


def execute(engine, day, oid, price, ref, *, halted=False, capacity=1000000):
    return engine.apply({'kind': 'execute', 'event_id': 'execute-'+oid+'-'+day,
        'fill_id': 'fill-'+oid+'-'+day, 'order_id': oid, 'time': clock(day, 'open'),
        'quote_time': clock(day, 'open'), 'price': str(price), 'capacity': capacity,
        'halted': halted, 'side_blocked': False, 'sellable_at': clock(engine._next_day(day), 'open'),
        'reference_close': str(ref), 'normal_session': True})


def mark(engine, day, prices, *, refs=None, opens=None, halted=()):
    quotes = {}
    for asset, value in prices.items():
        price = money(value)
        if asset in halted:
            prior = next(r for r in reversed(engine.tables['positions']) if r['asset_id'] == asset)
            quotes[asset] = {'price': str(price), 'observed_at': prior['observed_at'],
                            'carry_reason': 'synthetic_halt_last_mark', 'halted': True}
        else:
            ref = (refs or {}).get(asset, engine.references.get(asset, price))
            op = money((opens or {}).get(asset, price))
            quotes[asset] = {'price': str(price), 'observed_at': clock(day),
                'bar': {'reference_close': str(ref), 'normal_session': True,
                        'open': str(op), 'high': str(max(op, price)), 'low': str(min(op, price)), 'close': str(price)}}
    return engine.apply({'kind': 'mark', 'event_id': 'mark-'+day, 'time': clock(day), 'quotes': quotes})


def _require(condition, reason):
    if not condition:
        raise AssertionError(reason)


def audit_engine(engine):
    """Independently replay signed cash/shares and component totals at each mark."""
    tables = engine.export()
    cash = engine.initial_cash
    for row in tables['cash_ledger']:
        cash += row['delta']
        _require(cash == row['balance'], 'CASH_LEDGER_RECONCILIATION')
    _require(cash == engine.cash, 'LEDGER_RECONCILIATION')
    for order in tables['orders']:
        fills = [f for f in tables['fills'] if f['order_id'] == order['order_id']]
        notional = sum((f['price']*f['quantity'] for f in fills), ZERO)
        _require(notional == order['notional'], 'LEDGER_RECONCILIATION')
        if not fills:
            continue
        expected = {'commission': max(Decimal('5'), money(notional*Decimal('.0001854'))),
                    'stamp_duty': money(notional*Decimal('.0005')) if order['side'] == 'sell' else ZERO,
                    'transfer_fee': money(notional*Decimal('.00001'))}
        for component, value in expected.items():
            actual = sum((f['amount'] for f in tables['fees'] if f['order_id'] == order['order_id'] and f['component'] == component), ZERO)
            _require(actual == value, 'FEE_COMPONENT_RECONCILIATION')
    for fee in tables['fees']:
        entries = [c for c in tables['cash_ledger'] if c['kind'] == fee['component'] and c['reference'] == fee['fill_id']]
        _require(len(entries) == 1 and entries[0]['delta'] == -fee['amount'], 'LEDGER_RECONCILIATION')
    for fill in tables['fills']:
        signed = fill['quantity'] if fill['side'] == 'buy' else -fill['quantity']
        entries = [c for c in tables['cash_ledger'] if c['kind'] == 'trade' and c['reference'] == fill['fill_id']]
        _require(len(entries) == 1 and entries[0]['delta'] == -signed*fill['price'], 'LEDGER_RECONCILIATION')
    for row in tables['daily_nav']:
        ledger = tables['cash_ledger'][:row['cash_ledger_count']]
        _require(row['cash'] == engine.initial_cash+sum((c['delta'] for c in ledger), ZERO), 'LEDGER_RECONCILIATION')
        positions = [p for p in tables['positions'] if p['date'] == row['date']]
        for p in positions:
            qty = sum(c['delta'] for c in tables['share_ledger'][:row['share_ledger_count']] if c['asset_id'] == p['asset_id'])
            _require(p['quantity'] == qty and p['market_value'] == money(p['mark']*qty), 'LEDGER_RECONCILIATION')
        _require(row['market_value'] == sum((p['market_value'] for p in positions), ZERO), 'LEDGER_RECONCILIATION')
        _require(row['nav'] == row['cash']+row['market_value']+row['receivables'], 'LEDGER_RECONCILIATION')
        actions = tables['corporate_actions'][:row['corporate_action_count']]
        paid = {a['action_id'] for a in actions if a['kind'] == 'dividend_pay'}
        _require(row['receivables'] == sum((a['amount'] for a in actions if a['kind'] == 'dividend_record' and a['action_id'] not in paid), ZERO), 'LEDGER_RECONCILIATION')
    return {'status': 'passed', 'checks': ['cash', 'shares', 'fees_by_component', 'NAV', 'receivables']}


def build_equity_fixture():
    days = DEFINITION['sessions']
    engine = EquityAccountingEngine('10000', instruments=DEFINITION['instruments'], sessions=days)
    sh, sz = 'FIXTURE.SH', 'FIXTURE.SZ'
    checks = []
    def expect(name, actual, expected):
        if actual != expected:
            raise AssertionError(f'{name}: {actual} != {expected}')
        checks.append({'name': name, 'actual': actual, 'expected': expected, 'status': 'passed'})
    mark(engine, days[0], {sh: '10', sz: '10'})
    expect('invalid_690_buy', submit(engine, days[0], 'bad-lot', sz, 'buy', 690, '10'), 'quantity_constraint')
    expect('insufficient_cash', submit(engine, days[0], 'bad-cash', sh, 'buy', 10000, '10'), 'insufficient_cash')
    expect('oversell', submit(engine, days[0], 'bad-sell', sh, 'sell', 100, '10'), 'insufficient_sellable')
    expect('buy_SH_submit', submit(engine, days[0], 'buy-sh', sh, 'buy', 300, '10'), 'accepted')
    execute(engine, days[1], 'buy-sh', '10', '10')
    mark(engine, days[1], {sh: '10', sz: '10'})
    expect('day1_cash', engine.cash, Decimal('6994.97'))
    submit(engine, days[1], 'buy-sz', sz, 'buy', 600, '10')
    execute(engine, days[2], 'buy-sz', '10', '10')
    mark(engine, days[2], {sh: '10', sz: '10'})
    expect('day2_nav', engine.tables['daily_nav'][-1]['nav'], Decimal('9989.91'))
    submit(engine, days[2], 'limit-sell', sh, 'sell', 300, '9')
    expect('lower_limit_sell_rejected', execute(engine, days[3], 'limit-sell', '9', '10'), 'price_limit_blocked')
    engine.apply({'kind': 'cancel', 'event_id': 'cancel-limit', 'time': clock(days[3], 'open'), 'order_id': 'limit-sell'})
    mark(engine, days[3], {sh: '9', sz: '10'})
    expect('limit_day_nav', engine.tables['daily_nav'][-1]['nav'], Decimal('9689.91'))
    submit(engine, days[3], 'sell-sz', sz, 'sell', 600, '10')
    execute(engine, days[4], 'sell-sz', '10', '10')
    mark(engine, days[4], {sh: '9', sz: '10'})
    expect('sell_cash_includes_all_fees', engine.cash, Decimal('6981.85'))
    submit(engine, days[4], 'buy-sh2', sh, 'buy', 700, '9')
    execute(engine, days[5], 'buy-sh2', '9', '9')
    mark(engine, days[5], {sh: '9', sz: '10'})
    expect('four_fill_nav', engine.tables['daily_nav'][-1]['nav'], Decimal('9676.79'))
    engine.apply({'kind': 'external_flow', 'event_id': 'deposit', 'time': clock(days[6]), 'amount': '2000'})
    mark(engine, days[6], {sh: '9.90', sz: '10'})
    engine.apply({'kind': 'split', 'event_id': 'split', 'action_id': 'split', 'asset_id': sh,
                  'time': days[7]+'T01:00:00+00:00', 'numerator': 2, 'denominator': 1})
    mark(engine, days[7], {sh: '4.95', sz: '10'})
    expect('split_quantity', engine.total(sh), 2000)
    engine.apply({'kind': 'dividend_record', 'event_id': 'ex-dividend', 'action_id': 'div', 'asset_id': sh,
                  'time': days[8]+'T01:00:00+00:00', 'cash_per_share': '0.10'})
    mark(engine, days[8], {sh: '4.85', sz: '10'})
    expect('dividend_receivable', engine.tables['daily_nav'][-1]['receivables'], Decimal('200'))
    submit(engine, days[8], 'halt-sell', sh, 'sell', 2000, '4.85')
    expect('halt_execution', execute(engine, days[9], 'halt-sell', '4.85', '4.85', halted=True), 'halted')
    mark(engine, days[9], {sh: '4.85', sz: '10'}, halted=(sh,))
    engine.apply({'kind': 'dividend_pay', 'event_id': 'pay', 'action_id': 'div', 'time': days[10]+'T01:00:00+00:00'})
    mark(engine, days[10], {sh: '4.85', sz: '10'})
    expect('final_nav', engine.tables['daily_nav'][-1]['nav'], Decimal('12576.79'))
    expect('final_cash', engine.cash, Decimal('2876.79'))
    expect('commission', sum((r['amount'] for r in engine.tables['fees'] if r['component'] == 'commission'), ZERO), Decimal('20'))
    expect('stamp_duty', sum((r['amount'] for r in engine.tables['fees'] if r['component'] == 'stamp_duty'), ZERO), Decimal('3'))
    expect('transfer_fee', sum((r['amount'] for r in engine.tables['fees'] if r['component'] == 'transfer_fee'), ZERO), Decimal('.21'))
    audit = audit_engine(engine)
    metrics = path_functionals_v2(engine.tables['daily_nav'][1:], initial_nav='10000', initial_time=clock(days[0]))
    expect('max_drawdown', metrics['max_drawdown'], Decimal('.032321'))
    expect('flow_neutral_total_return', metrics['total_return'], Decimal('.057679'))
    expect('recovery_time', metrics['max_drawdown_recovery_time'], clock(days[6]))
    expect('longest_underwater_sessions', metrics['longest_underwater_observations'], 5)
    expect('max_drawdown_peak_to_trough', metrics['max_drawdown_peak_to_trough_observations'], 5)
    expect('max_drawdown_trough_to_recovery', metrics['max_drawdown_trough_to_recovery_observations'], 1)
    expect('max_drawdown_peak_to_recovery', metrics['max_drawdown_peak_to_recovery_observations'], 6)
    prices = [{'time': e['time'], 'asset_id': asset, 'close': q['price']}
              for e in engine.seen.values() if e['kind'] == 'mark' and e['time'] != clock(days[0])
              for asset, q in e['quotes'].items()]
    return plain({'schema_version': 2, 'definition': DEFINITION, 'tables': engine.export(), 'prices': prices,
                  'path_metrics': metrics, 'checks': checks, 'audit': audit,
                  'research_status': 'not_eligible', 'real_market_data': False})


def publish_equity_fixture(output_dir: Path, *, render_charts=False, font_path=None) -> Path:
    payload = build_equity_fixture()
    lake = DataLake(output_dir)
    path = lake.write_immutable_json(output_dir/'fixture.json', payload)
    outputs = {'fixture': lake.artifact_record(path)}
    if render_charts:
        from .backtest_charts import render_backtest_charts
        images = render_backtest_charts(payload['prices'], payload['tables']['fills'], payload['path_metrics'],
            payload['tables']['positions'], directory=output_dir/'charts',
            corporate_actions=payload['tables']['corporate_actions'],
            daily_accounts=payload['tables']['daily_nav'][1:], language='zh' if font_path else 'en',
            font_path=font_path, title='交易规则与账务验收 v2' if font_path else 'Execution acceptance v2',
            data_label='合成数据 · 非真实行情 · 含三项交易费用及外部入金' if font_path else 'Synthetic fixture only · fees and external deposit')
        outputs.update({p.suffix[1:]: lake.artifact_record(p) for p in images})
    return lake.write_immutable_json(output_dir/'_MANIFEST.json', {
        'schema_version': 2, 'run_id': DEFINITION['id']+'_'+json_hash(DEFINITION)[:16],
        'status': 'passed', 'definition': DEFINITION, 'code_hash': source_tree_hash(),
        'real_market_data': False, 'research_promoted': False,
        'parent_run_ids': [], 'risk_basis_id': None, 'risk_set_version': None,
        'outputs': outputs})
