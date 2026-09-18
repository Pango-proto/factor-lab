"""Registered synthetic Monte Carlo calibration. Never promotes real research."""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from ..components.probe_signal import MaterializeProbeSignal
from ..components.labels import BuildForwardLabels
from ..core.contracts import ArtifactKey, ArtifactRef, LoadedArtifact, CalculationRequest, QualityStatus
from ..l2.estimation_panel import _residualize, _spearman
from ..publishing.local import _write_immutable_parquet

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT/'config/synthetic_temporal_calibration_v1.json'
EVAL = np.arange(20,148)
DAYS = [date(2000,1,1)+timedelta(days=i) for i in range(153)]
ASSETS = [f'SIM{i:03d}' for i in range(128)]


def standardize(values):
    sd = np.std(values)
    return (values-np.mean(values))/sd if sd>1e-12 else np.zeros_like(values)


class SyntheticDesign:
    """Deterministic exposure/support cache; reference and production solve separately."""
    def __init__(self, config, scenario):
        self.columns = config['risk_columns']
        self.ipo = np.zeros(128,dtype=int)
        if scenario in (3,4):
            self.ipo[96:112]=48;self.ipo[112:]=80
        t=np.arange(153)[:,None];i=np.arange(128)
        self.listed=t>=self.ipo
        self.finite=self.listed.copy()
        if scenario in (3,4):
            self.finite &= ~((i>=16)&((t+7*i)%97==0))
        self.exposures=np.full((153,128,len(self.columns)),np.nan)
        industries=sorted(c for c in self.columns if c.startswith('risk_industry_'))
        boards=sorted(c for c in self.columns if c.startswith('risk_board_'))
        u=(i+.5)/128
        self.fit_masks=[];self.prod_designs=[];self.reference=[]
        for d in range(153):
            active=self.finite[d]
            size=standardize(u[active])
            cubic=size**3
            nuisance=np.column_stack((np.ones(len(size)),size))
            nonlinear=cubic-nuisance@np.linalg.lstsq(nuisance,cubic,rcond=1e-10)[0]
            styles={'risk_size':size,'risk_beta':standardize(np.cos(2*np.pi*u[active])),
                'risk_residual_volatility':standardize(np.sin(4*np.pi*u[active])),
                'risk_liquidity':standardize(np.cos(6*np.pi*u[active])),
                'risk_nonlinear_size':standardize(nonlinear),
                'risk_listing_age':standardize(np.log1p(np.maximum(1,d-self.ipo[active]+1)))}
            for k,col in enumerate(self.columns):
                if col=='risk_country': value=np.ones(active.sum())
                elif col in industries: value=((i[active]%31)==industries.index(col)).astype(float)
                elif col in boards: value=(((i[active]//31)%4)==boards.index(col)).astype(float)
                else: value=styles[col]
                self.exposures[d,active,k]=value
            signal_finite=self.finite[max(0,d-19):d+1].sum(axis=0)>=15
            fit=active&signal_finite
            self.fit_masks.append(fit)
            x=self.exposures[d,fit].copy()
            self.prod_designs.append(x)
            # Full-column least-squares linear map. Only deterministic design
            # is cached; no residuals or production solver cache are shared.
            if fit.sum()>=max(3,len(self.columns)+2):
                inverse=np.linalg.lstsq(x,np.eye(len(x)),rcond=1e-10)[0]
                self.reference.append((x.copy(),inverse))
            else: self.reference.append(None)
        self.expected_n=np.array([np.sum(self.fit_masks[d]&self.finite[d+1:d+6].all(axis=0)) for d in EVAL])
        if np.any(self.expected_n<3) or any(self.reference[d] is None for d in EVAL):
            raise ValueError('SYNTHETIC_REGISTERED_DESIGN_UNSUPPORTED')


def stream(scenario,role,outer,panel,component):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [20260818,777,scenario,role,outer,panel,component])))


def ar_path(noise,phi,scale):
    result=np.empty_like(noise);previous=np.zeros(noise.shape[0:1]+noise.shape[2:])
    for d in range(noise.shape[1]):
        previous=phi*previous+scale*noise[:,d];result[:,d]=previous
    return result


def generate_returns(design,scenario,role,outer,panel_ids):
    """Seed-addressed arrays; reference panels always follow the null DGP."""
    def normal(component,shape):
        return np.stack([stream(scenario,role,outer,p,component).standard_normal(shape) for p in panel_ids])
    market=ar_path(normal(0,(153,)),.3,np.sqrt(1-.3**2))
    industry=ar_path(normal(1,(153,31)),.2,np.sqrt(1-.2**2))
    size=ar_path(normal(2,(153,)),.1,np.sqrt(1-.1**2))
    i=np.arange(128);u=(i+.5)/128;q=np.sqrt(12)*(u-.5)
    background=.0001*q+.004*market[:,:,None]+.004*industry[:,:,i%31]+.002*q*size[:,:,None]
    epsilon=normal(4,(153,128))
    if scenario in (1,4): epsilon=(-1.)**i*(np.exp(epsilon-.5)-1)/np.sqrt(np.e-1)
    if scenario in (2,4):
        h=ar_path(normal(3,(153,128)),.9,.15)
        sigma=(.008+.016*u)*np.exp(.3*np.sin(2*np.pi*np.arange(153)[None,:,None]/40)+h/2)
    else: sigma=.015
    raw=background+sigma*epsilon
    returns=np.clip(raw,-.4,.4)
    returns[:,~design.finite]=np.nan
    if role==1 and 0 in panel_ids:
        a=list(panel_ids).index(0)
        for d in range(20,153):
            prior=d-1;valid=design.fit_masks[prior]
            extra=np.zeros(128)
            if design.reference[prior] is not None:
                history=returns[a,max(0,prior-19):prior+1]
                # Feedback is causal and uses the production OLS solver.
                signal=1-np.exp(np.nansum(np.log1p(history),axis=0))
                residual=_residualize(signal[valid],design.prod_designs[prior])[0]
                extra[valid]=.003*standardize(residual)
            returns[a,d]=np.clip(raw[a,d]+extra,-.4,.4)
            returns[a,d,~design.finite[d]]=np.nan
    return returns


def reference_signal_labels(returns):
    """Independent product/count prefixes and explicit five future factors."""
    finite=np.isfinite(returns)
    prefix=np.concatenate((np.ones((len(returns),1,128)),np.cumprod(np.where(finite,1+returns,1.),axis=1)),axis=1)
    count=np.concatenate((np.zeros((len(returns),1,128),dtype=int),np.cumsum(finite,axis=1)),axis=1)
    signal=np.full_like(returns,np.nan);labels=np.full_like(returns,np.nan)
    for d in range(153):
        first=max(0,d-19)
        value=1-prefix[:,d+1]/prefix[:,first]
        signal[:,d]=np.where(count[:,d+1]-count[:,first]>=15,value,np.nan)
        if d+5<153:
            growth=np.ones((len(returns),128));valid=np.ones((len(returns),128),dtype=bool)
            for offset in (1,2,3,4,5):
                growth*=1+returns[:,d+offset];valid &=finite[:,d+offset]
            labels[:,d]=np.where(valid,growth-1,np.nan)
    return signal,labels


def reference_statistics(returns,design):
    signal,labels=reference_signal_labels(returns)
    daily=np.empty((len(returns),128));counts=np.empty((len(returns),128),dtype=int)
    for j,d in enumerate(EVAL):
        fit=design.fit_masks[d];x,inverse=design.reference[d]
        values=signal[:,d,fit].T
        residual=values-x@(inverse@values)
        valid=np.isfinite(labels[0,d,fit])
        if not np.all(np.isfinite(labels[:,d,fit])==valid[None,:]):
            raise ValueError('SYNTHETIC_REFERENCE_MASK_CHANGED')
        def rank_columns(matrix):
            ranked=pl.DataFrame(matrix).select(pl.all().rank(method='average')).to_numpy()
            ranked=ranked-ranked.mean(axis=0)
            norm=np.sqrt(np.sum(ranked*ranked,axis=0))
            if np.any(norm==0): raise ValueError('SYNTHETIC_REFERENCE_RANK_DEGENERATE')
            return ranked/norm
        sr=rank_columns(residual[valid]);yr=rank_columns(labels[:,d,fit][:,valid].T)
        daily[:,j]=np.sum(sr*yr,axis=0);counts[:,j]=valid.sum()
    if not np.all(counts==design.expected_n): raise ValueError('SYNTHETIC_REFERENCE_SUPPORT_MISMATCH')
    return daily.mean(axis=1),counts


def loaded(kind,frame):
    ref=ArtifactRef(key=ArtifactKey(artifact_type=kind,version='1',run_id='synthetic_'+kind,
        as_of_date=DAYS[-1],scope_key='synthetic:preregistered'),uri='synthetic',checksum='in_memory',
        quality_status=QualityStatus.PASSED,tags=frozenset())
    return LoadedArtifact(reference=ref,tables={kind+'_v1':frame})


def production_signal_labels(returns,design):
    """Use the actual registered component implementations on synthetic rows."""
    keys=pl.DataFrame({'trade_date':[day for day in DAYS for _ in ASSETS],'asset_id':ASSETS*153})
    frame=keys.with_columns(pl.Series('realized_return',returns.reshape(-1)),
        pl.lit('TRADED').alias('observation_state'),pl.lit(False).alias('is_resumption_day'))
    # Keep missing-day placeholder rows inside the listed domain so rolling
    # windows remain market-day windows. Synthetic gaps are floating NaNs.
    universe=loaded('tradable_universe',keys.filter(pl.Series(design.listed.reshape(-1))))
    realized=loaded('realized_returns',frame)
    signal_component=MaterializeProbeSignal(None)
    request=CalculationRequest(operation_id=signal_component.spec.operation_id,operation_version='1',
        as_of_date=DAYS[-1],scope_key='synthetic:preregistered',parameters={},inputs=())
    s=signal_component.calculate(request,(realized,universe)).outputs[0].tables['probe_signal_panel_v1']
    label_component=BuildForwardLabels(DataLake(PROJECT/'data'))
    label_request=CalculationRequest(operation_id=label_component.spec.operation_id,operation_version='1',
        as_of_date=DAYS[-1],scope_key='synthetic:preregistered',parameters={'horizons':[5]},inputs=())
    y=label_component.calculate(label_request,(realized,universe)).outputs[0].tables['forward_labels_v1']
    merged=keys.join(s.select('trade_date','asset_id','signal_value'),on=['trade_date','asset_id'],how='left',validate='1:1')
    merged=merged.join(y.select('trade_date','asset_id','target_return'),on=['trade_date','asset_id'],how='left',validate='1:1')
    return merged['signal_value'].to_numpy().reshape(153,128),merged['target_return'].to_numpy().reshape(153,128)


def production_statistics(returns,design,role):
    signal,labels=production_signal_labels(returns,design)
    daily=[];counts=[]
    for d in EVAL:
        fit=design.fit_masks[d]
        if not np.all(np.isfinite(signal[d,fit])): raise ValueError('SYNTHETIC_PRODUCTION_SIGNAL_SUPPORT')
        values=signal[d,fit].copy()
        if role==2:
            y=labels[d,fit];valid=np.isfinite(y)
            values[valid]+=.5*np.std(values)*standardize(y[valid])
        residual=_residualize(values,design.prod_designs[d])[0]
        joint=np.isfinite(labels[d,fit])
        ic=_spearman(residual[joint],labels[d,fit][joint])
        if ic is None: raise ValueError('SYNTHETIC_PRODUCTION_RANK_DEGENERATE')
        daily.append(ic);counts.append(int(joint.sum()))
    if not np.array_equal(counts,design.expected_n): raise ValueError('SYNTHETIC_PRODUCTION_SUPPORT_MISMATCH')
    return float(np.mean(daily)),counts


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def verify_inputs(config,path):
    if config['experiment_id']!='synthetic_temporal_calibration_v1': raise ValueError('SYNTHETIC_VERSION_UNSUPPORTED')
    fixed={'seed':20260818,'namespace':777,'null_repetitions':400,'power_repetitions':200,'references':99,
        'assets':128,'market_dates':153,'evaluation_first':20,'evaluation_last':147,'horizon':5,
        'lookback':20,'minimum_signal_observations':15,'rcond':1e-10,'alpha':.05,
        'null_failure_count':35,'power_wilson_lower':.8,'wilson_z':1.959963984540054,
        'scenario_order':[[i,0] for i in range(5)]+[[4,1],[4,2]],'research_promotion_permitted':False}
    if any(config.get(key)!=value for key,value in fixed.items()):
        raise ValueError('SYNTHETIC_PREREGISTERED_PARAMETERS_CHANGED')
    protected={PROJECT/config['preregistration']:config['preregistration_sha256'],
        **{PROJECT/p:sha for p,sha in config['protected_files'].items()}}
    for p,sha in protected.items():
        if file_sha256(p)!=sha: raise ValueError(f'SYNTHETIC_INPUT_PIN_MISMATCH {p}')
    metadata=config['risk_columns_metadata'];mp=PROJECT/'data'/metadata['path']
    if file_sha256(mp)!=metadata['sha256']: raise ValueError('SYNTHETIC_RISK_METADATA_CHANGED')
    if json.loads(mp.read_text())['metadata']['risk_columns']!=config['risk_columns']:
        raise ValueError('SYNTHETIC_RISK_COLUMNS_CHANGED')
    return {str(p):sha for p,sha in protected.items()}


def verify_receipt(lake,row,scenario,role,outer,expected_n):
    record=row['table']
    if file_sha256(lake.root/record['path'])!=record['sha256']:
        raise ValueError('SYNTHETIC_RESUME_CHECKSUM_MISMATCH')
    frame=pl.read_parquet(lake.root/record['path'])
    values=frame['statistic'].to_numpy()
    valid=(frame.height==100 and frame['panel_id'].to_list()==list(range(100))
        and frame['scenario_id'].unique().to_list()==[scenario]
        and frame['role_id'].unique().to_list()==[role] and frame['outer_id'].unique().to_list()==[outer]
        and frame['seed_prefix'].to_list()==[[20260818,777,scenario,role,outer,p] for p in range(100)]
        and np.isfinite(values).all() and np.all(np.asarray(frame['n_valid_by_date'].to_list())==expected_n))
    p=(1+int(np.sum(values[1:]>=values[0])))/100
    if (not valid or row['observed_statistic']!=values[0] or row['p_value']!=p
        or row['reject']!=(p<=.05) or row['scenario']!=scenario or row['role']!=role or row['outer']!=outer):
        raise ValueError('SYNTHETIC_RESUME_RECEIPT_MISMATCH')


def run_synthetic_calibration(lake: DataLake,*,config_path=DEFAULT_CONFIG,progress=print):
    config=json.loads(config_path.read_text());protected=verify_inputs(config,config_path)
    definition={'experiment_id':config['experiment_id'],'config_sha256':file_sha256(config_path),
        'preregistration_sha256':config['preregistration_sha256'],'code_hash':source_tree_hash(),
        'uv_lock_sha256':file_sha256(PROJECT/'uv.lock'),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'polars':pl.__version__}}
    run_id='synthetic_calibration_v1_'+json_hash(definition)[:16]
    directory=lake.root/'diagnostics/synthetic_temporal_calibration_v1'/f'run_id={run_id}'
    lake.write_immutable_json(directory/'_LOCK.json',{'definition':definition,'protected_files':protected})
    final_manifest=directory/'_MANIFEST.json'
    if final_manifest.exists():
        complete=json.loads(final_manifest.read_text())
        for name,record in complete['outputs'].items():
            path=lake.root/record['path']
            if file_sha256(path)!=record['sha256']: raise ValueError('SYNTHETIC_COMPLETED_OUTPUT_CHANGED')
            if name.startswith('scenario_') and '_outer_' in name:
                receipt=json.loads(path.read_text());table=receipt['table']
                if file_sha256(lake.root/table['path'])!=table['sha256']:
                    raise ValueError('SYNTHETIC_COMPLETED_TABLE_CHANGED')
        progress(f'已核验全部合成校准产物，复用 {run_id}')
        return final_manifest
    records=[];outputs={};run_start=time.monotonic()
    for scenario,role in config['scenario_order']:
        design=SyntheticDesign(config,scenario)
        support_name=f'support_scenario_{scenario}'
        support_path=directory/f'{support_name}.json'
        lake.write_immutable_json(support_path,{'scenario':scenario,'risk_columns':design.columns,
            'exposure_sha256':hashlib.sha256(design.exposures.tobytes()).hexdigest(),
            'finite_return_mask_sha256':hashlib.sha256(design.finite.tobytes()).hexdigest(),
            'n_valid_by_date':design.expected_n.tolist(),'finite_return_cells':int(design.finite.sum()),
            'listed_cells':int(design.listed.sum()),'IPO_indices':design.ipo.tolist()})
        outputs[support_name]=lake.artifact_record(support_path)
        repetitions=config['null_repetitions'] if role==0 else config['power_repetitions']
        for outer in range(repetitions):
            name=f'scenario_{scenario}_role_{role}_outer_{outer:04d}'
            receipt=directory/'repetitions'/f'{name}.json'
            if receipt.exists():
                row=json.loads(receipt.read_text())
                verify_receipt(lake,row,scenario,role,outer,design.expected_n)
            else:
                started=time.monotonic()
                observation=generate_returns(design,scenario,role,outer,[0])[0]
                observed,n_valid=production_statistics(observation,design,role)
                references=generate_returns(design,scenario,role,outer,list(range(1,100)))
                reference_values,reference_n=reference_statistics(references,design)
                p=(1+int(np.sum(reference_values>=observed)))/100
                table=pl.DataFrame({'scenario_id':[scenario]*100,'role_id':[role]*100,'outer_id':[outer]*100,
                    'panel_id':list(range(100)),'statistic':[observed,*reference_values.tolist()],
                    'n_valid_by_date':[n_valid,*reference_n.tolist()],
                    'seed_prefix':[[20260818,777,scenario,role,outer,panel] for panel in range(100)]})
                table_path=directory/'repetitions'/f'{name}.parquet'
                _write_immutable_parquet(table_path,table)
                row={'scenario':scenario,'role':role,'outer':outer,'observed_statistic':observed,
                    'p_value':p,'reject':p<=.05,'reference_count':99,'table':lake.artifact_record(table_path),
                    'return_path_sha256':hashlib.sha256(observation.tobytes()+references.tobytes()).hexdigest(),
                    'seconds':time.monotonic()-started,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'peak_rss_unit':'bytes' if platform.system()=='Darwin' else 'KiB'}
                lake.write_immutable_json(receipt,row)
            records.append(row);outputs[name]=lake.artifact_record(receipt)
            if outer==0 or (outer+1)%20==0:
                progress(f'情景 {scenario} / 角色 {role}：{outer+1}/{repetitions}；本次 {row["seconds"]:.2f}s；累计完成 {len(records)}/2400')
    results=[]
    for scenario,role in config['scenario_order']:
        selected=[r for r in records if r['scenario']==scenario and r['role']==role]
        n=len(selected);k=sum(r['reject'] for r in selected);interval=wilson(k,n)
        results.append({'scenario':scenario,'role':role,'repetitions':n,'reject_count':k,
            'reject_rate':k/n,'wilson_95_interval':interval,
            'passed':k<35 if role==0 else interval[0]>=.8})
    verify_inputs(config,config_path)
    if (source_tree_hash()!=definition['code_hash'] or file_sha256(config_path)!=definition['config_sha256']
        or file_sha256(PROJECT/'uv.lock')!=definition['uv_lock_sha256']):
        raise ValueError('SYNTHETIC_IMPLEMENTATION_CHANGED_DURING_RUN')
    report={'run_id':run_id,'execution_status':'completed','status':'synthetic_framework_calibration_passed'
        if all(r['passed'] for r in results) else 'synthetic_framework_calibration_failed',
        'results':results,'completed_outer_repetitions':len(records),'completed_panels':len(records)*100,
        'seconds_this_invocation':time.monotonic()-run_start,'holdout_touched':False,
        'real_research_status':'blocked','temporal_validation_status':'not_calibrated',
        'interpretation':'Known synthetic DGP only; no real-market calibration or research promotion.',
        'support':{'assets':ASSETS,'market_day_indices':[0,152],'evaluation_indices':[20,147],
            'IPO_cohorts':[[0,95,0],[96,111,48],[112,127,80]],
            'missing_rule':'i>=16 and (t+7*i)%97==0 in scenarios 3,4'}}
    # Timing belongs in per-repetition receipts. Keep aggregate reuse immutable.
    report.pop('seconds_this_invocation')
    rp=lake.write_immutable_json(directory/'report.json',report);outputs['report']=lake.artifact_record(rp)
    manifest=lake.write_immutable_json(directory/'_MANIFEST.json',{'schema_version':1,'run_id':run_id,
        'status':report['status'],'definition':definition,'protected_files':protected,'outputs':outputs,
        'research_promotion_permitted':False,'holdout_touched':False})
    return manifest
