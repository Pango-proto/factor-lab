"""M3 engineering acceptance: synthetic PIT risk snapshots -> targets -> ledger.

Six synthetic style columns plus industry are plumbing fixtures, not an adopted
Barra factor list. Existing real candidate factor set stays unchanged.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json
import math
from pathlib import Path
import random
import numpy as np
import polars as pl
from ...storage import DataLake, json_hash, file_sha256, source_tree_hash
from .risk_snapshot import exposure_snapshot, forecast_snapshot, portfolio_risk, admit_target, calibration_report
from .equity_execution import EquityAccountingEngine, next_weekdays, clock
from .equity_fixture import mark, submit, execute
from .execution_invariants import audit_execution_path
from .backtest_path import path_functionals_v3
from .accounting import money
from .strategy_fixture import plain

ASSETS=['FIXTURE.SH','FIXTURE.SZ']
FACTORS=['risk_industry_A','risk_industry_B','risk_size','risk_beta','risk_residual_volatility','risk_liquidity','risk_nonlinear_size','risk_listing_age']
X=[[1,0,1,.8,.4,.3,.2,.1],[0,1,-1,1.2,-.4,-.3,-.2,-.1]]
ESTIMATOR={'id':'registered_ewma_group_shrinkage_v1','factor_half_life':34.65620375188814,
    'specific_half_life':10.393355749081836,'minimum_history':60}
BIAS_POLICY={'lower_bound':.95,'upper_bound':1.05,'minimum_observations':60}
CONTRACT={'id':'m3_risk_acceptance_v1','seed':1729,'warmup':80,'sessions':84,
    'purpose':'engineering_fixture_only','risk_basis_id':'synthetic_m3_basis_v1','risk_set_version':1,
    'estimator':ESTIMATOR,'bias_policy':BIAS_POLICY,'horizon_sessions':1,
    'strategy':'same_lagged_5_day_return_rank_tie_asset_id;rebalance_21_sessions',
    'candidate_targets':'top_ranked_asset_weights_0.8_to_0_in_steps_0.1;unused_cash',
    'cash_admission':'reserve_buys_from_decision_cash_only_no_anticipated_sell_proceeds',
    'risk_versions':{'baseline':'no_risk_constraint','exposure':'abs_size<=0.4;industry_each<=0.6',
        'forecast':'exposure_constraints_and_daily_variance<=0.000025'},
    'limits_scope':'synthetic_acceptance_only_not_research_preregistration',
    'availability':'synthetic_close_atomic_phase_return_then_risk_then_decision_zero_latency',
    'current_and_holdout':'no_mutation_no_evaluation'}


def history_fixture(n=80):
    rng=random.Random(1729)
    days=next_weekdays('2026-01-01T00:00:00+00:00',n+2)
    rows=[]
    for i in range(n):
        realized=clock(days[i]);available=realized
        rows.append({'realized_at':realized,'available_at':available,'risk_basis_id':CONTRACT['risk_basis_id'],
            'risk_set_version':1,'factor_returns':{f:rng.gauss(0,.0025) for f in FACTORS},
            'specific_returns':{a:rng.gauss(0,.006) for a in ASSETS}})
    return rows


def snapshot_pair(rows):
    cutoff=rows[-1]['realized_at'];available=rows[-1]['available_at']
    decision=available
    x=exposure_snapshot(asset_ids=ASSETS,factor_ids=FACTORS,values=X,risk_basis_id=CONTRACT['risk_basis_id'],risk_set_version=1,
        risk_factor_set_status='candidate',valid_at=cutoff,available_at=available,decision_time=decision,
        expected_valid_at=cutoff,parent_run_ids=['synthetic_m3_history_v1'],purpose='engineering_fixture_only')
    f=forecast_snapshot(x,rows,decision_time=decision,estimation_cutoff=cutoff,expected_cutoff=cutoff,
        estimator=ESTIMATOR,group_by_asset={a:'fixture_group' for a in ASSETS})
    return x,f,decision


def calibration_acceptance(model_id):
    # Known standardized sample variance exactly one; a doubled sample must fail.
    days=next_weekdays('2026-01-01T00:00:00+00:00',62)
    scale=math.sqrt(59/60)
    pairs=[]
    for i in range(60):
        pairs.append({'prediction_id':str(i),'model_id':model_id,'horizon_sessions':1,
            'forecast_available_at':clock(days[i]),'decision_time':clock(days[i]),
            'outcome_start':clock(days[i+1],'01:30:00'),'outcome_end':clock(days[i+1]),
            'outcome_available_at':clock(days[i+1]),'realized_return':(-1 if i%2 else 1)*.01*scale,'predicted_variance':.0001})
    published=clock(days[61]);passed=calibration_report(pairs,model_id=model_id,policy=BIAS_POLICY,purpose='engineering_fixture_only',published_at=published)
    failed=calibration_report([{**p,'realized_return':p['realized_return']*2} for p in pairs],model_id=model_id,policy=BIAS_POLICY,purpose='engineering_fixture_only',published_at=published)
    assert passed['status']=='passed' and failed['status']=='failed'
    return {'known_unit_std':passed,'known_double_std':failed}


def comparison_fixture():
    history=history_fixture(164)
    days=[r['realized_at'][:10] for r in history]
    prices={a:[D('20')] for a in ASSETS}
    for row in history:
        factors=np.array([row['factor_returns'][f] for f in FACTORS])
        for j,a in enumerate(ASSETS):
            ret=float(np.asarray(X[j])@factors)+row['specific_returns'][a]
            prices[a].append(money(str(float(prices[a][-1])*(1+ret))))
    limits={'risk_size':[-.4,.4],'risk_industry_A':[0,.6],'risk_industry_B':[0,.6]}
    versions=[];snapshot_cache={}
    # Reuse identical PIT snapshots across versions, never change estimator settings.
    for i in range(80,164):snapshot_cache[i]=snapshot_pair(history[:i])
    for version in ('baseline','exposure','forecast'):
        sessions=days[79:]+next_weekdays(days[-1]+'T00:00:00+00:00',3)[1:]
        engine=EquityAccountingEngine('100000',instruments={a:'MAIN' for a in ASSETS},sessions=sessions)
        mark(engine,days[79],{a:prices[a][80] for a in ASSETS})
        reports=[];decisions=[]
        for i in range(80,164):
            x,f,decision=snapshot_cache[i]
            if (i-80)%21==0:
                scores={a:float(prices[a][i]/prices[a][i-5]-1) for a in ASSETS}
                winner=sorted(ASSETS,key=lambda a:(-scores[a],a))[0]
                trials=[]
                for tenths in range(8,-1,-1):
                    weights={a:tenths/10 if a==winner else 0. for a in ASSETS}
                    # Both risk versions consume exactly the same alpha and candidate grid.
                    gate=admit_target(x,f,weights,decision_time=decision,exposure_limits=limits,
                        max_variance=.000025 if version=='forecast' else None,engineering_only=True)
                    trials.append({'weights':weights,'accepted':gate['accepted'],'reasons':gate['reasons']})
                    if version=='baseline' or gate['accepted']:break
                nav=engine.tables['daily_nav'][-1]['nav']
                target={a:int(nav*D(str(weights[a]))/prices[a][i]/100)*100 for a in ASSETS}
                decisions.append({'day':i-79,'scores':scores,'winner':winner,'weights':weights,'trials':trials,
                    'risk_snapshot_id':f['snapshot_id'],'decision_time':decision})
                pending=[]
                for side in ('sell','buy'):
                    for a in ASSETS:
                        delta=target[a]-engine.total(a)
                        if (side=='sell' and delta<0) or (side=='buy' and delta>0):
                            opening=prices[a][i]
                            qty=abs(delta)
                            if side=='buy':qty=engine.affordable_quantity(a,engine.cash-engine.reserved_cash,opening,qty)
                            if qty:
                                oid=f'{i}:{side}:{a}'
                                submit(engine,days[i-1],oid,a,side,qty,opening)
                                pending.append((oid,a,opening))
                for oid,a,opening in pending:
                    execute(engine,days[i],oid,opening,prices[a][i])
            mark(engine,days[i],{a:prices[a][i+1] for a in ASSETS},opens={a:prices[a][i] for a in ASSETS})
            nav=engine.tables['daily_nav'][-1]['nav']
            # Decision-price holdings report before trading next day; match X/F clock.
            current_weights={a:float(engine.total(a)*prices[a][i+1]/nav) for a in ASSETS}
            reports.append({'day':i-79,'weights_at_close':current_weights,
                'report_kind':'closing_holdings_under_prior_close_risk_forecast_not_new_forecast',
                'risk':portfolio_risk(x,f,current_weights,decision_time=decision)})
        invariant=audit_execution_path(engine)
        metrics=path_functionals_v3(engine.tables['daily_nav'][1:],initial_nav='100000',initial_time=clock(days[79]),flow_timing='period_end')
        versions.append({'id':version,'summary':{'total_return':float(metrics['total_return']),
            'max_drawdown':float(metrics['max_drawdown']),'total_fees':float(sum(f['amount'] for f in engine.tables['fees'])),
            'fills':len(engine.tables['fills']),'mean_predicted_daily_volatility':float(np.mean([r['risk']['volatility'] for r in reports])),
            'turnover':sum(float(f['quantity']*f['price'])/float(engine.tables['daily_nav'][sessions.index(f['time'][:10])-1]['nav']) for f in engine.tables['fills']),
            'mean_concentration':float(np.mean([sum(w*w for w in r['weights_at_close'].values()) for r in reports]))},
            'decisions':decisions,'daily_risk':reports,'nav':[{'x':i,'y':float(r['nav']/D('100000'))} for i,r in enumerate(engine.tables['daily_nav'])],
            'tables':plain(engine.export()),'invariants':invariant})
    x,f,decision=snapshot_cache[80]
    production=admit_target(x,f,{ASSETS[0]:.2,ASSETS[1]:0.},decision_time=decision,exposure_limits=limits)
    if production['accepted']:raise AssertionError('RISK_PRODUCTION_GATE_BYPASSED')
    return {'contract':CONTRACT,'versions':versions,'snapshot_example':{'X':x,'F_Delta':f},
        'calibration_acceptance':calibration_acceptance(f['model_id']),'production_gate':production,
        'risk_basis_id':CONTRACT['risk_basis_id'],'real_market_data':False,'research_status':'not_eligible'}


def real_readiness(data_root:Path, config_root:Path):
    """Inspect immutable lineage and schemas only. Never fit or open holdout outcomes."""
    inputs={};parents=[];basis=[];latest={};missing=[]
    for name,table_names in [('risk_exposure_matrix',['risk_exposure_matrix_v1']),('l2b_risk_only',['factor_returns_v1','specific_returns_v1'])]:
        pointer=json.loads((data_root/'gold'/name/'_CURRENT.json').read_text())
        manifest_path=data_root/pointer['manifest'];manifest=json.loads(manifest_path.read_text())
        if manifest['run_id']!=pointer['run_id'] or manifest['risk_basis_id']!=pointer['risk_basis_id']:raise ValueError('M3_REAL_LINEAGE_MISMATCH')
        parents.append(manifest['run_id']);basis.append(manifest['risk_basis_id']);latest[name]=manifest['end']
        inputs[name]={'manifest':str(manifest_path),'sha256':file_sha256(manifest_path),'created_at':manifest.get('created_at'),'tables':{}}
        for table in table_names:
            record=manifest['outputs'][table];path=data_root/record['path']
            if file_sha256(path)!=record['sha256']:raise ValueError('M3_REAL_TABLE_CHECKSUM')
            schema=pl.scan_parquet(path).collect_schema()
            inputs[name]['tables'][table]={'path':str(path),'sha256':record['sha256'],'fields':list(schema)}
            if 'available_at' not in schema:missing.append(table)
    if len(set(basis))!=1:raise ValueError('M3_REAL_BASIS_MISMATCH')
    policy=json.loads((config_root/'evaluation_framework_v2.json').read_text())['sample']
    reasons=['RISK_SET_CANDIDATE_NOT_FROZEN','CURRENT_BASIS_CALIBRATION_PENDING',
        'RISK_SNAPSHOTS_BEHIND_MARKET_DATA','HISTORICAL_AVAILABLE_AT_EVIDENCE_REQUIRED']
    return {'status':'blocked','fit_performed':False,'holdout_opened':False,'risk_basis_id':basis[0],
        'parent_run_ids':parents,'latest_dates':latest,'inputs':inputs,'missing_available_at_tables':missing,
        'holdout_start':policy['holdout_start'],'reasons':reasons,
        'next_action':'approve_explicit_historical_availability_policy_and_development_calibration_plan_before_fitting',
        'risk_set_version':json.loads((config_root/'risk_set_expansion_v1.json').read_text())['risk_set_version']}


def publish_m3(output_dir:Path,data_root:Path,config_root:Path):
    payload=comparison_fixture();readiness=real_readiness(data_root,config_root)
    lake=DataLake(output_dir)
    full=lake.write_immutable_json(output_dir/'acceptance.json',payload)
    ready=lake.write_immutable_json(output_dir/'real-readiness.json',readiness)
    # Compact presentation artifact; authoritative ledgers stay separately inspectable.
    view={**payload,'versions':[{k:v for k,v in version.items() if k not in ('tables','daily_risk','decisions')} for version in payload['versions']],
        'real_readiness':{k:v for k,v in readiness.items() if k!='inputs'}}
    page=lake.write_immutable_json(output_dir/'risk.json',view)
    config_hashes={name:file_sha256(config_root/name) for name in ['l2_risk_validation_v1.json','risk_set_expansion_v1.json','evaluation_framework_v2.json','l2c_ragged_factor_panel_readiness_v1.json']}
    return lake.write_immutable_json(output_dir/'_MANIFEST.json',{'schema_version':1,'run_id':CONTRACT['id']+'_'+json_hash(CONTRACT)[:16],
        'definition':CONTRACT,'code_hash':source_tree_hash(),'config_hashes':config_hashes,
        'status':'engineering_passed_real_risk_blocked','research_promoted':False,'parent_run_ids':readiness['parent_run_ids'],
        'risk_basis_id':CONTRACT['risk_basis_id'],'real_risk_basis_id':readiness['risk_basis_id'],'risk_set_version':1,
        'outputs':{k:lake.artifact_record(p) for k,p in [('acceptance',full),('readiness',ready),('preview',page)]}})
