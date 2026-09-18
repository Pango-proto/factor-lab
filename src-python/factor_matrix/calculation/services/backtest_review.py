"""Reproducible analytics and execution counterfactuals for a pinned M2b run."""
from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from .accounting import AccountingEngine, decimal
from .accounting_fixture import audit_tables
from .backtest_path import path_functionals, capacity_diagnostics, off_policy_evaluation
from .strategy_fixture import execute_intents, load_stage, plain


def execution_counterfactual(config, frozen_targets, *, capacity_multiplier):
    """Freeze targets and prices; recompute order deltas from each actual state."""
    engine = AccountingEngine(config['initial_cash'], lot_size=config['strategy']['lot_size'])
    admissions = []
    for original in config['sessions']:
        session = deepcopy(original)
        for side in ('buy', 'sell'):
            for q in session['opportunities'][side].values():
                q['capacity'] = int(Decimal(q['capacity']) * capacity_multiplier)
        intents = []
        for target in frozen_targets:
            if target['session'] != session['session']:
                continue
            delta = target['target_quantity'] - engine.total(target['asset_id'])
            if delta:
                intents.append({'intent_id': target['target_id'], 'target_id': target['target_id'],
                    'decision_time': target['decision_time'], 'asset_id': target['asset_id'],
                    'side': 'buy' if delta > 0 else 'sell', 'requested_quantity': abs(delta),
                    'limit_price': target['reference_price']})
        rows, _ = execute_intents(engine, intents, session)
        admissions.extend(rows)
    ledger = plain(engine.export())
    audit = audit_tables(ledger, config['initial_cash'])
    if audit['status'] != 'passed':
        raise ValueError('COUNTERFACTUAL_LEDGER_FAILED')
    initial_time = config['sessions'][0]['decision_time']
    return {'scenario': f'capacity_x{capacity_multiplier}', 'capacity_multiplier': capacity_multiplier,
            'interpretation': 'fixed-target execution-model counterfactual; not identified causal OPE',
            'path': path_functionals(ledger['daily_nav'], initial_nav=config['initial_cash'], initial_time=initial_time),
            'commission_total': sum((decimal(r['amount']) for r in ledger['fees']), Decimal('0.00')),
            'admissions': admissions, 'ledger': ledger, 'reconciliation': audit}


def review_backtest(lake: DataLake, *, config_path: Path) -> Path:
    cfg = json.loads(config_path.read_text())
    if (cfg['contract_id'] != 'backtest_analytics_v1' or cfg['capacity_multipliers'] != ['1', '0.5']
        or cfg['participation'] != '0.01' or cfg['adv_observations'] != 20):
        raise ValueError('ANALYTICS_CONTRACT_UNSUPPORTED')
    source_path = lake.root / cfg['source']['path']
    if file_sha256(source_path) != cfg['source']['sha256']:
        raise ValueError('ANALYTICS_SOURCE_CHECKSUM_MISMATCH')
    source = json.loads(source_path.read_text())
    if source['status'] != 'engineering_passed' or source['purpose'] != 'engineering_fixture_only':
        raise ValueError('ANALYTICS_REQUIRES_REGISTERED_ENGINEERING_PARENT')
    stages = [load_stage(lake, ref) for ref in source['stages']]
    parent_config = stages[0]['input_config']
    ledger = load_stage(lake, source['final_backtest'])
    target_record = source['outputs']['targets']
    target_path = lake.root / target_record['path']
    if file_sha256(target_path) != target_record['sha256']:
        raise ValueError('ANALYTICS_TARGET_CHECKSUM_MISMATCH')
    targets = json.loads(target_path.read_text())
    functionals = path_functionals(ledger['daily_nav'], initial_nav=parent_config['initial_cash'],
                                  initial_time=parent_config['sessions'][0]['decision_time'])
    scenarios = [execution_counterfactual(parent_config, targets, capacity_multiplier=decimal(v)) for v in cfg['capacity_multipliers']]
    for key in ('orders', 'fills', 'fees', 'cash_ledger', 'positions', 'daily_nav'):
        if scenarios[0]['ledger'][key] != ledger[key]:
            raise ValueError('BASELINE_REPLAY_MISMATCH ' + key)
    # M2b's five synthetic days do not supply a 20-day ADV history. Never use
    # same-day simulated fill capacity as if it were prior market volume.
    capacity_rows = [{'asset_id': t['asset_id'], 'decision_time': t['decision_time'],
        'requested_quantity': abs(t['target_quantity'] - t['opening_quantity']),
        'reference_price': t['reference_price'], 'adv_shares': None,
        'adv_observations': 0, 'required_observations': cfg['adv_observations']} for t in targets]
    capacity = capacity_diagnostics(capacity_rows, participation=cfg['participation'], lot_size=10)
    ope = off_policy_evaluation([], assumptions_verified=False)
    definition = {'config_sha256': file_sha256(config_path), 'code_hash': source_tree_hash(), 'parent': cfg['source']}
    run_id = 'backtest_analytics_v1_' + json_hash(definition)[:16]
    directory = lake.root / 'diagnostics' / 'backtest_analytics_v1' / f'run_id={run_id}'
    outputs = {}
    for name, rows in {'path_functionals': functionals, 'capacity': capacity, 'strict_ope': ope,
                       'counterfactuals': scenarios, 'contract': cfg}.items():
        outputs[name] = lake.artifact_record(lake.write_immutable_json(directory / f'{name}.json', plain(rows)))
    return lake.write_immutable_json(directory / '_MANIFEST.json', {
        'schema_version': 1, 'run_id': run_id, 'status': 'engineering_passed', 'definition': definition,
        'parent_run_ids': [source['run_id']], 'outputs': outputs, 'purpose': 'engineering_fixture_only',
        'research_status': 'not_eligible', 'parameter_selection': False, 'holdout_touched': False,
        'capacity_status': 'insufficient_liquidity_history', 'strict_ope_status': 'not_estimable',
    })
