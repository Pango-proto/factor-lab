"""Approved retrospective diagnostic. Observed clocks never become replay clocks."""
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os

import numpy as np
import polars as pl

from ...storage import DataLake,file_sha256,json_hash,utc_now,source_tree_hash
from ...factor_engine.registry import FactorRegistry
from ...weight_metric_contract import load_weight_metric_contract
from ..l1.momentum import load_core6_design
from ..l1.builder import build_l1_day
from ..l2.config import load_return_decomposition_config
from ..l2.label_policy import load_return_label_policy
from .core6_inputs import load_inputs
from .core6_pipeline import audit_universe
from .core6_model import build_core6_day,fit_core6_day,estimate_core6_risk


def momentum_panel(returns,calendar,definition,knowledge_time):
    """Calendar-dense compiled equivalent of the reviewed observed product."""
    if returns['known_at'].null_count() or returns['known_at'].max()>knowledge_time:
        raise ValueError('CORE6_MOMENTUM_PANEL_FUTURE_SOURCE')
    if returns.unique(['trade_date','asset_id']).height != returns.height:
        raise ValueError('CORE6_MOMENTUM_PANEL_DUPLICATE')
    assets=returns.select('asset_id').unique()
    panel=assets.join(pl.DataFrame({'trade_date':calendar}),how='cross').join(
        returns.select('trade_date','asset_id','total_return','return_source'),
        on=['trade_date','asset_id'],how='left',validate='1:1').sort(['asset_id','trade_date'])
    usable=(pl.col('total_return').is_finite() & (pl.col('total_return')>=-1)
        & pl.col('return_source').is_not_null() & (pl.col('return_source')!='')
        & (pl.col('return_source')!='resumption')).fill_null(False)
    invalid=(pl.col('total_return').is_not_null() & (pl.col('return_source')!='resumption').fill_null(True)
        & ~usable).fill_null(False)
    panel=panel.with_columns(usable.cast(pl.Int64).alias('ok'),invalid.cast(pl.Int64).alias('bad'),
        (usable & (pl.col('total_return')==-1)).cast(pl.Int64).alias('zero'),
        pl.when(usable & (pl.col('total_return')>-1)).then(pl.col('total_return').log1p()).otherwise(0.).alias('log_gross'))
    panel=panel.with_columns(*[pl.col(c).rolling_sum(definition.window_sessions,min_samples=1)
        .shift(definition.nearest_lag).over('asset_id').alias(c+'_window') for c in ('ok','bad','zero','log_gross')])
    panel=panel.with_columns(pl.when((pl.col('ok_window')>=definition.minimum_observations)&(pl.col('bad_window')==0))
        .then(pl.when(pl.col('zero_window')>0).then(-1.).otherwise(pl.col('log_gross_window').exp()-1.))
        .otherwise(None).alias('momentum_raw'),pl.col('ok_window').fill_null(0).alias('n_momentum_obs'))
    return panel.select('trade_date','asset_id','momentum_raw','n_momentum_obs')


def liquidity_groups(exposure):
    ordered=exposure.sort(['risk_liquidity','asset_id'])['asset_id'].to_list()
    # SQL ntile(10): earliest buckets get one extra row, deterministic tie order.
    q,r=divmod(len(ordered),10);result={};position=0
    for bucket in range(10):
        n=q+(bucket<r)
        for asset in ordered[position:position+n]:result[asset]=str(bucket+1)
        position+=n
    return result


_WORK={}


def _initialize(cache_dir,project_root):
    root=Path(project_root);cache=Path(cache_dir)
    spec,momentum=load_core6_design(root/'config/risk_core6_design_v2.json',project_root=root)
    meta=json.loads((cache/'meta.json').read_text())
    _WORK.update(root=root,cache=cache,spec=spec,momentum=momentum,meta=meta,
        calendar=[date.fromisoformat(d) for d in meta['calendar']],
        knowledge=datetime.fromisoformat(meta['knowledge_time']),registry=FactorRegistry.discover(),
        l2config=load_return_decomposition_config(root/'config/l2_return_decomposition_v1.json',derived_params=meta['parameters']),
        label_policy=load_return_label_policy(root/'config/return_label_policy_v1.json'))
    for name in ['returns_daily','valuation_daily','security_daily_state','prices_daily','universe','board','membership','momentum','weights']:
        frame=pl.read_parquet(cache/(name+'.parquet'))
        _WORK[name]=frame
    _WORK['universe_dates']=_WORK['universe'].partition_by('trade_date',as_dict=True)
    _WORK['momentum_dates']=_WORK['momentum'].partition_by('trade_date',as_dict=True)


def _one_day(pair):
    exposure_date,return_date=map(date.fromisoformat,pair)
    w=_WORK;spec=w['spec'];meta=w['meta'];root=w['root']
    tag=return_date.isoformat(); directory=w['cache'].parent/'daily'/tag
    directory.mkdir(parents=True,exist_ok=True)
    status={'exposure_date':str(exposure_date),'return_date':tag,'status':'failed','errors':{},
            'temporal_mode':'reconstructed_diagnostic_only','historical_pit_verified':False}
    calendar=w['calendar'];i=calendar.index(exposure_date);first=calendar[max(0,i-251)]
    frames={n:w[n].filter(pl.col('trade_date').is_between(first,exposure_date)) for n in ('returns_daily','valuation_daily','security_daily_state')}
    board=w['board'].filter(pl.col('trade_date').is_between(first,exposure_date))
    universe=audit_universe(w['universe_dates'][(exposure_date,)],policy=meta['universe_policy'],d0=meta['d0'])
    universe.write_parquet(directory/'coverage.parquet')
    assets=universe.filter('risk_eligible').with_columns(pl.lit(True).alias('base_tradable_without_listing_age'))
    weights=dict(w['weights'].filter(pl.col('exposure_date')==exposure_date).select('asset_id','candidate_weight').iter_rows())
    source_available=max(frames['returns_daily']['known_at'].max(),frames['valuation_daily']['known_at'].max(),
        frames['security_daily_state']['known_at'].max(),board['source_available_at'].max(),
        universe['industry_available_at'].drop_nulls().max())
    replay=datetime.combine(return_date,time(9,25),ZoneInfo('Asia/Shanghai'))
    outcome_start=datetime.combine(return_date,time(9,30),ZoneInfo('Asia/Shanghai'))
    labels=w['returns_daily'].filter(pl.col('trade_date')==return_date)
    prices=w['prices_daily'].filter(pl.col('trade_date')==return_date)
    models={}
    try:
        mom=w['momentum_dates'][(exposure_date,)].join(assets.select('asset_id'),on='asset_id',how='inner')
        new=build_core6_day(as_of=exposure_date,knowledge_time=w['knowledge'],assets=assets,
            universe_history=frames['security_daily_state'],returns=frames['returns_daily'],valuation=frames['valuation_daily'],
            board=board,calendar=calendar,spec=spec,momentum_definition=w['momentum'],parameters=meta['parameters'],
            descriptor_parameters=meta['descriptor_parameters'],metric_weights=weights,source_available_at=source_available,
            momentum_override=mom)
        models['core6']=new['exposure'];new['provenance'].write_parquet(directory/'cell_provenance.parquet')
        new['quality'].write_parquet(directory/'style_quality.parquet')
    except (ValueError,RuntimeError) as exc:status['errors']['core6_L1']=str(exc)
    try:
        old=build_l1_day(trade_date=exposure_date,universe_day=assets,universe_history=frames['security_daily_state'],
            valuation=frames['valuation_daily'],returns=frames['returns_daily'],board_benchmarks=board,
            benchmark_membership=w['membership'],registry=w['registry'],derived_params=meta['parameters'],
            new_listing_policy_path=root/'config/new_listing_policy_v1.json',risk_factor_set_id='old44_reconstructed_control',
            metric_weights_by_asset=weights,metric_weight_scheme_id='structural_pit_v1')
        if not old.exposure['is_valid'].all():raise ValueError('OLD44_L1_INVALID')
        oldx=old.exposure.join(assets.select('asset_id','float_mkt_cap','exchange_list_date'),on='asset_id',validate='1:1')
        from ...canonical_definitions import exposure_orthogonalization_weight
        oldx=oldx.with_columns(pl.Series('candidate_weight',[weights.get(a,exposure_orthogonalization_weight(c)) for a,c in zip(oldx['asset_id'],oldx['float_mkt_cap'])]))
        models['old44']=oldx
    except (ValueError,RuntimeError) as exc:status['errors']['old44_L1']=str(exc)
    if len(models)==2 and models['core6']['asset_id'].to_list()!=models['old44']['asset_id'].to_list():
        raise ValueError('CORE6_PAIRED_ASSET_AXIS_MISMATCH')
    for name,x in models.items():
        columns=spec['expanded_columns'] if name=='core6' else meta['old_columns']
        if not set(columns)<=set(x.columns):
            status['errors'][name+'_axis']='fixed_columns_missing';continue
        x=x.with_columns(pl.lit(source_available).alias('partial_source_available_at'),
            pl.lit(None,dtype=pl.Datetime('us','UTC')).alias('observed_available_at'),
            pl.lit('unknown_retained_weight_publication').alias('availability_status'),
            pl.lit(utc_now()).alias('computed_at'),pl.lit(replay).alias('replay_decision_at'))
        x.write_parquet(directory/(name+'_X.parquet'))
        try:
            model_spec={**spec,'expanded_columns':columns}
            blocks={'industry':spec['industry_columns']}
            if name=='old44':blocks['board']=[c for c in columns if c.startswith('risk_board_')]
            result=fit_core6_day(exposure=x,labels=labels,prices=prices,spec=model_spec,config=w['l2config'],
                label_policy=w['label_policy'],next_session=return_date,outcome_start=outcome_start,
                exposure_available_at=replay,categorical_blocks=blocks)
            pl.DataFrame({'factor_id':columns,'factor_return':result.factor_returns}).write_parquet(directory/(name+'_factor.parquet'))
            pl.DataFrame({'asset_id':result.included_assets,'specific_return':result.specific_returns,
                'in_estimation_domain':result.in_estimation_domain,'effective_weight':result.estimation_weights,
                'huber_multiplier':result.huber_weight_multipliers}).write_parquet(directory/(name+'_specific.parquet'))
            status[name]={'sample_count':result.sample_count,'label_sample_count':result.label_sample_count,
                'condition_number':result.condition_number,'constraint_error':result.constraint_error,
                'identity_error':result.regression_identity_error,'r_squared':result.r_squared,
                'weight_fallback_assets':sum(a not in weights for a in x['asset_id'])}
        except (ValueError,RuntimeError) as exc:status['errors'][name+'_G3']=str(exc)
    status['status']='passed' if not status['errors'] and len(models)==2 else 'failed'
    (directory/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    return status


def run_reconstruction(*,project_root,gate_path,gate_sha256,approval_path,output_root,workers=2):
    approval=json.loads(approval_path.read_text())
    if approval['status']!='approved' or approval['historical_pit_verified'] is not False:
        raise ValueError('CORE6_RECONSTRUCTION_APPROVAL_REQUIRED')
    if file_sha256(project_root/approval['proposal_path'])!=approval['proposal_sha256']:
        raise ValueError('CORE6_RECONSTRUCTION_PROPOSAL_CHANGED')
    design=project_root/'config/risk_core6_design_v2.json';spec,momentum=load_core6_design(design,project_root=project_root)
    window=(spec['paired_validation']['start'],spec['paired_validation']['end'])
    if (approval['start'],approval['end'])!=window or not 1<=workers<=4:
        raise ValueError('CORE6_RECONSTRUCTION_WINDOW_OR_WORKERS')
    lake=DataLake(project_root/'data')
    inputs=load_inputs(lake=lake,gate_path=gate_path,gate_sha256=gate_sha256,spec=spec,historical_window=window)
    oldpath=project_root/'data/gold/risk_exposure_matrix/run_id=l1_risk_exposure_history_20260814_e3b192fa61031760/_MANIFEST.json'
    params=json.loads(oldpath.read_text())['derived_params']
    contract=load_weight_metric_contract(project_root/'config/weight_metric_contract_v1.json');weights=contract.resolve_artifact(lake.root)
    expansion=json.loads((project_root/'config/risk_set_expansion_v1.json').read_text())
    oldcols=[c for m in expansion['logical_members'] for c in m['generated_columns']]
    identity={'approval_sha256':file_sha256(approval_path),'design_sha256':file_sha256(design),
        'data_gate_sha256':gate_sha256,'window':window,'source_records':inputs['source_records'],
        'weight_contract':contract.as_identity(),'parameters':params,'source_tree_sha256':source_tree_hash(project_root),
        'code':{str(p.relative_to(project_root)):file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('core6_model.py'),Path(__file__).with_name('core6_inputs.py'),Path(__file__).with_name('core6_pipeline.py')]}}
    run_id='core6_reconstructed_'+json_hash(identity)[:16];directory=output_root/f'run_id={run_id}'
    manifest_path=directory/'_MANIFEST.json'
    if manifest_path.exists():
        for relative,record in json.loads(manifest_path.read_text())['outputs'].items():
            if file_sha256(directory/relative)!=record['sha256']:raise ValueError('CORE6_RECONSTRUCTION_OUTPUT_CHANGED')
        return manifest_path
    cache=directory/'inputs';cache.mkdir(parents=True,exist_ok=True)
    minimal={'returns_daily':['trade_date','asset_id','total_return','return_source','known_at','available_at','availability_evidence'],
        'valuation_daily':['trade_date','asset_id','float_mkt_cap','turnover_rate','known_at'],
        'security_daily_state':['trade_date','asset_id','board_id','known_at'],
        'prices_daily':['trade_date','asset_id','raw_open','raw_high','raw_low','raw_close','limit_up','limit_down','known_at']}
    for name,cols in minimal.items(): inputs['frames'][name].select(cols).write_parquet(cache/(name+'.parquet'))
    for name,frame in {'universe':inputs['universe_history'],'board':inputs['board'],'membership':inputs['membership'],
        'weights':pl.read_parquet(weights),'industry':inputs['industry'],'availability':inputs['availability']}.items():
        frame.write_parquet(cache/(name+'.parquet'))
    calendar=inputs['calendar'];first=inputs['frames']['returns_daily']['trade_date'].min()
    mom=momentum_panel(inputs['frames']['returns_daily'],[d for d in calendar if d>=first],momentum,inputs['knowledge_time'])
    mom.write_parquet(cache/'momentum.parquet')
    weight_notice=json.loads((weights.parent/'RETAINED_INPUT_NOTICE.json').read_text())
    if weight_notice['retained_sha256']!=file_sha256(weights):raise ValueError('CORE6_RETAINED_WEIGHT_HASH')
    meta={'calendar':[str(d) for d in calendar],'knowledge_time':inputs['knowledge_time'].isoformat(),
        'parameters':params,'old_columns':oldcols,'weight_available_at':None,
        'weight_availability_status':'unknown_retained_input_without_original_publication_manifest',
        'descriptor_parameters':json.loads((project_root/'config/risk_descriptor_parameters_v1.json').read_text())['definitions'],
        'universe_policy':json.loads((project_root/'config/tradable_universe_v1.json').read_text()),
        'd0':json.loads((project_root/'config/new_listing_policy_v1.json').read_text())['d0_model_exclusion']['value']}
    (cache/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    pairs=[(str(calendar[i-1]),str(d)) for i,d in enumerate(calendar) if window[0]<=str(d)<=window[1]]
    del inputs,mom
    statuses=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(str(cache),str(project_root))) as pool:
        for index,status in enumerate(pool.map(_one_day,pairs,chunksize=1)):
            statuses.append(status)
            if (index+1)%20==0:print(f'core6 paired G3 {index+1}/{len(pairs)}; failed={sum(s["status"]!="passed" for s in statuses)}',flush=True)
    (directory/'daily_status.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2)+'\n')
    # Common successful dates, with every excluded date retained above, never a new window.
    common=[s for s in statuses if s['status']=='passed']
    computed=utc_now();forecasts={}
    for name,columns in [('core6',spec['expanded_columns']),('old44',oldcols)]:
        observations=[]
        basis=name+'_'+json_hash(identity)[:16]
        for status in common:
            day=directory/'daily'/status['return_date']
            observations.append({'trade_date':date.fromisoformat(status['return_date']),'available_at':computed,
                'basis_id':basis,'factor_returns':dict(pl.read_parquet(day/(name+'_factor.parquet')).iter_rows()),
                'specific_returns':dict(pl.read_parquet(day/(name+'_specific.parquet')).select('asset_id','specific_return').iter_rows())})
        if not common:forecasts[name]={'status':'unavailable','reason':'no_common_G3_dates'};continue
        latest=directory/'daily'/common[-1]['return_date']
        x=pl.read_parquet(latest/(name+'_X.parquet'))
        forecasts[name]=estimate_core6_risk(observations=observations,asset_groups=liquidity_groups(x),factor_ids=columns,
            basis_id=basis,decision_time=computed,parameters=params)
        forecasts[name].update(temporal_mode='reconstructed_diagnostic_only',historical_pit_verified=False,
            estimation_cutoff=common[-1]['return_date'],exposure_date=common[-1]['exposure_date'],computed_at=computed.isoformat())
        (directory/(name+'_F_Delta.json')).write_text(json.dumps(forecasts[name],ensure_ascii=False,indent=2)+'\n')
    output_records={str(p.relative_to(directory)):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
        for p in directory.rglob('*') if p.is_file()}
    manifest={'schema_version':1,'run_id':run_id,'status':'completed_reconstructed_diagnostic',
        'temporal_mode':'reconstructed_diagnostic_only','historical_pit_verified':False,'production_eligible':False,
        'parent_run_ids':[json.loads(gate_path.read_text())['run_id']],'identity':identity,
        'created_at':computed.isoformat(),'start':window[0],'end':window[1],
        'paired_dates':len(pairs),'common_passed_dates':len(common),'failed_dates':len(pairs)-len(common),
        'minimum_warmup':60,'calibration_status':'not_calibrated','registry_status':'standalone_diagnostic_basis',
        'forecasts':{k:{'status':v['status'],'factor_observations':v.get('factor_observations',0),
            'missing_Delta':sum(x is None for x in v.get('Delta',{}).values())} for k,v in forecasts.items()},
        'current_changed':False,'holdout_evaluated':False,'outputs':output_records}
    lake.write_immutable_json(manifest_path,manifest)
    return manifest_path
