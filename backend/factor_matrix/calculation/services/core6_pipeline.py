"""Independent candidate publication. Never moves production/research CURRENT."""
from datetime import date
from pathlib import Path
import json

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, utc_now
from ...weight_metric_contract import load_weight_metric_contract
from ..l1.momentum import load_core6_design
from .core6_inputs import load_inputs
from .core6_model import build_core6_day, estimate_core6_risk


def audit_universe(latest, *, policy, d0):
    flags = {
        'excluded_st': pl.col('is_st').fill_null(True) if policy['exclude_st'] else pl.lit(False),
        'excluded_suspended': pl.col('is_suspended').fill_null(True) if policy['exclude_suspended'] else pl.lit(False),
        'excluded_bse': pl.col('board_id') == 'BSE' if not policy['include_bse'] else pl.lit(False),
        'excluded_new_listing': (pl.col('days_since_exchange_list') < d0).fill_null(True),
        'excluded_price': (~pl.col('raw_close').is_finite() | (pl.col('raw_close') <= 0)).fill_null(True),
        'excluded_cap': (~pl.col('float_mkt_cap').is_finite() | (pl.col('float_mkt_cap') <= 0)).fill_null(True),
        'excluded_industry': pl.col('sw_l1_code').is_null(),
        'excluded_board': ~pl.col('board_id').is_in(['MAIN','CHINEXT','STAR','BSE']).fill_null(False),
    }
    audit = latest.with_columns(*[v.alias(k) for k,v in flags.items()])
    return audit.with_columns((~pl.any_horizontal(*flags)).alias('risk_eligible'))


def run_core6_current(*, project_root: Path, gate_path: Path, gate_sha256: str, output_root: Path):
    design_path = project_root/'config/risk_core6_design_v2.json'
    spec, momentum = load_core6_design(design_path, project_root=project_root)
    lake = DataLake(project_root/'data')
    inputs = load_inputs(lake=lake, gate_path=gate_path, gate_sha256=gate_sha256, spec=spec)
    inherited_path = project_root/'data/gold/risk_exposure_matrix/run_id=l1_risk_exposure_history_20260814_e3b192fa61031760/_MANIFEST.json'
    parameters = json.loads(inherited_path.read_text())['derived_params']
    contract = load_weight_metric_contract(project_root/'config/weight_metric_contract_v1.json')
    weight_path = contract.resolve_artifact(lake.root)
    descriptor_path = project_root/'config/risk_descriptor_parameters_v1.json'
    params = json.loads(descriptor_path.read_text())['definitions']
    policy_path = project_root/'config/tradable_universe_v1.json'
    policy = json.loads(policy_path.read_text())
    listing_path = project_root/'config/new_listing_policy_v1.json'
    d0 = json.loads(listing_path.read_text())['d0_model_exclusion']['value']
    files = [Path(__file__), Path(__file__).with_name('core6_inputs.py'), Path(__file__).with_name('core6_model.py'),
        project_root/'backend/factor_matrix/calculation/l1/momentum.py',
        project_root/'backend/factor_matrix/calculation/l1/descriptors.py',
        project_root/'backend/factor_matrix/calculation/l1/builder.py',
        project_root/'backend/factor_matrix/calculation/l1/style_math.py',
        project_root/'backend/factor_matrix/calculation/l2/risk_modeling.py']
    identity = {'design_sha256':file_sha256(design_path),'data_gate_sha256':gate_sha256,
        'source_records': inputs['source_records'], 'inherited_parameters_sha256':file_sha256(inherited_path),
        'weight_contract':contract.as_identity(),'universe_policy_sha256':file_sha256(policy_path),
        'listing_policy_sha256':file_sha256(listing_path),'descriptor_parameters_sha256':file_sha256(descriptor_path),
        'code':{str(p.relative_to(project_root)):file_sha256(p) for p in files}}
    run_id = 'core6_current_'+json_hash(identity)[:16]
    directory = output_root/f'run_id={run_id}'
    manifest_path = directory/'_MANIFEST.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest['outputs'].values():
            if file_sha256(directory/record['path']) != record['sha256']:
                raise ValueError('CORE6_EXISTING_OUTPUT_CHANGED')
        return manifest_path
    directory.mkdir(parents=True, exist_ok=True)
    latest_date = date.fromisoformat(inputs['gate']['data_as_of'])
    universe = audit_universe(inputs['latest'], policy=policy, d0=d0)
    assets = universe.filter('risk_eligible')
    metric_weights = dict(pl.scan_parquet(weight_path).filter(pl.col('exposure_date') == latest_date)
                          .select('asset_id','candidate_weight').collect().iter_rows())
    source_available = max(inputs['calendar_available_at'], inputs['industry']['source_available_at'].max(),
        inputs['board']['source_available_at'].max(), *[f['known_at'].max() for f in inputs['frames'].values()])
    result = build_core6_day(as_of=latest_date,knowledge_time=inputs['knowledge_time'],assets=assets,
        universe_history=inputs['frames']['security_daily_state'],returns=inputs['frames']['returns_daily'],
        valuation=inputs['frames']['valuation_daily'],board=inputs['board'],calendar=inputs['calendar'],
        spec=spec,momentum_definition=momentum,parameters=parameters,descriptor_parameters=params,
        metric_weights=metric_weights,source_available_at=source_available)
    basis_id = 'risk_core6_candidate_'+json_hash({'design':identity['design_sha256'],
        'weight_contract':contract.as_identity(),'parameters':parameters,'code':identity['code']})[:16]
    published_at = utc_now()
    # Actual publication is now; no historical X publication is fabricated.
    x = result['exposure'].with_columns(pl.lit(basis_id).alias('candidate_basis_id'),
        pl.lit(2).alias('candidate_version'),pl.lit(published_at).alias('available_at'))
    groups = dict(x.select('asset_id','sw_l1_code').iter_rows())
    forecast = estimate_core6_risk(observations=[],asset_groups=groups,
        factor_ids=spec['expanded_columns'],basis_id=basis_id,decision_time=published_at,parameters=parameters)
    audit = inputs['availability']
    summary = {'execution_status':'passed','candidate_status':'not_calibrated',
        'data_as_of':str(latest_date),'knowledge_time':inputs['knowledge_time'].isoformat(),
        'source_available_at':source_available.isoformat(),'computed_at':published_at.isoformat(),
        'universe':{'listed_a_share_assets':universe.height,'risk_eligible':assets.height,
            'excluded':universe.height-assets.height,
            'exclusion_counts_overlap':{c:int(universe[c].sum()) for c in universe.columns if c.startswith('excluded_')}},
        'industry':{'intervals':inputs['industry'].height,'eligible_l1_coverage':float(assets['sw_l1_code'].is_not_null().mean()),
                    'historical_pit_verified':False},
        'board':{'start':str(inputs['board']['trade_date'].min()),'end':str(inputs['board']['trade_date'].max()),
                 'rows':inputs['board'].height,'latest':inputs['board'].filter(pl.col('trade_date')==latest_date).to_dicts()},
        'styles':result['quality'].to_dicts(),
        'momentum':{'raw_valid':result['momentum'].filter(~pl.col('raw_missing')).height,
            'unavailable':result['momentum'].filter('raw_missing').height,
            'partial_observed_products':result['momentum'].filter('partial_observed_product').height},
        'weight_fallback_assets':sum(a not in metric_weights for a in assets['asset_id']),
        'paired_window_availability':audit.group_by('table').agg(pl.len().alias('dates'),pl.col('known_by_event_date').sum()).to_dicts(),
        'G3':{'status':'unavailable','historical_published_core6_exposures':0,
              'reason':'new X is published now; no previous six-style X available before historical outcomes'},
        'F_Delta':{'status':forecast['status'],'factor_observations':forecast['factor_observations'],
                   'minimum_history':60,'reason':forecast['reason']},
        'registry_status':'standalone_candidate_identity_only_not_registered_in_active_sqlite',
        'production_eligible':False,'current_changed':False,'holdout_evaluated':False}
    outputs = {}
    for name,frame in {'industry_intervals':inputs['industry'],'board_benchmarks':inputs['board'],
        'asset_coverage':universe,'source_availability':audit,'risk_exposure':x,
        'style_quality':result['quality'],'cell_provenance':result['provenance'],'momentum_coverage':result['momentum']}.items():
        path=directory/(name+'.parquet'); frame.write_parquet(path,compression='zstd')
        outputs[name]={'path':path.name,'sha256':file_sha256(path),'rows':frame.height}
    for name,value in {'summary':summary,'risk_forecast':forecast}.items():
        path=directory/(name+'.json');path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
        outputs[name]={'path':path.name,'sha256':file_sha256(path)}
    manifest={'schema_version':1,'run_id':run_id,'status':'passed_engineering_partial_risk',
        'design_id':spec['design_id'],'candidate_version':2,'candidate_basis_id':basis_id,
        'risk_set_version':None,'risk_basis_id':None,'registry_status':summary['registry_status'],
        'purpose':'current_candidate_exposure_and_availability_audit',
        'parent_run_ids':[inputs['gate']['run_id']], 'silver_version_id':inputs['gate']['silver_version_id'],
        'start':str(inputs['board']['trade_date'].min()),'end':str(latest_date),
        'knowledge_time':inputs['knowledge_time'].isoformat(),'created_at':published_at.isoformat(),
        'identity':identity,'outputs':outputs,'promotion_allowed':False,
        'remaining':['registered_basis_identity','PIT_eligible_G3_history','F_Delta_estimation','real_stratified_calibration']}
    lake.write_immutable_json(manifest_path, manifest)
    return manifest_path
