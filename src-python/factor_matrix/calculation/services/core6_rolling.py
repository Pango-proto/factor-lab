"""Expanding one-session reconstructed forecasts; no outcome-based selection."""
from pathlib import Path
import json
import math

import numpy as np
import polars as pl

from ...storage import DataLake,file_sha256,json_hash,utc_now
from .core6_reconstruction import liquidity_groups


class RollingRiskState:
    """Streaming equivalent of registered EWMA and prior-only MAD clipping.

    F advances on complete G3 dates; each Delta advances on its own valid rows.
    The 252-place ring stores observations, not calendar days.
    """
    def __init__(self,assets,factors,parameters):
        self.assets=list(assets);self.factors=list(factors);self.positions={a:i for i,a in enumerate(assets)}
        n=len(assets);k=len(factors)
        self.fd=math.exp(math.log(.5)/parameters['ewma_factor_covariance_half_life_days'])
        self.sd=math.exp(math.log(.5)/parameters['ewma_specific_variance_half_life_days'])
        self.fw=0.;self.fm=np.zeros(k);self.fq=np.zeros((k,k));self.dates=0
        self.count=np.zeros(n,dtype=int);self.sums=np.zeros(n);self.weights=np.zeros(n);self.weight_squares=np.zeros(n)
        self.ring=np.full((252,n),np.nan);self.last=np.full(n,-1,dtype=int)

    def update(self,factor_returns,specific_returns):
        if set(factor_returns)!=set(self.factors):raise ValueError('CORE6_ROLLING_FACTOR_AXIS')
        v=np.array([factor_returns[f] for f in self.factors])
        if not np.isfinite(v).all():raise ValueError('CORE6_ROLLING_NONFINITE_FACTOR')
        pairs=[(self.positions[a],value) for a,value in specific_returns.items() if value is not None]
        ix=np.array([i for i,_ in pairs],dtype=int);values=np.array([value for _,value in pairs],dtype=float)
        if not np.isfinite(values).all():raise ValueError('CORE6_ROLLING_NONFINITE_SPECIFIC')
        squared=values.copy()
        eligible=self.count[ix]>=60
        if eligible.any():
            prior=self.ring[:,ix[eligible]]
            center=np.nanmedian(prior,axis=0)
            scale=1.482602218505602*np.nanmedian(np.abs(prior-center),axis=0)
            clipped=np.clip(values[eligible],center-5*scale,center+5*scale)
            squared[eligible]=np.where(scale>0,clipped,values[eligible])
        self.fw=self.fd*self.fw+1.;self.fm=self.fd*self.fm+v;self.fq=self.fd*self.fq+np.outer(v,v)
        self.weights[ix]=self.sd*self.weights[ix]+1.
        self.weight_squares[ix]=self.sd**2*self.weight_squares[ix]+1.
        self.sums[ix]=self.sd*self.sums[ix]+squared**2
        self.ring[self.count[ix]%252,ix]=values
        self.count[ix]+=1;self.last[ix]=self.dates;self.dates+=1

    def forecast(self,groups):
        if self.dates<60:raise ValueError('CORE6_ROLLING_WARMUP')
        mean=self.fm/self.fw;F=self.fq/self.fw-np.outer(mean,mean)
        if not np.isfinite(F).all() or np.linalg.eigvalsh(F).min() < -1e-12:
            raise ValueError('CORE6_ROLLING_NOT_PSD')
        eligible={a:self.positions[a] for a in groups if self.count[self.positions[a]]>=60 and self.last[self.positions[a]]==self.dates-1}
        raw={a:self.sums[i]/self.weights[i] for a,i in eligible.items()}
        targets={g:float(np.mean([raw[a] for a in raw if groups[a]==g])) for g in set(groups[a] for a in raw)}
        delta={a:None for a in groups}
        for a,i in eligible.items():
            neff=self.weights[i]**2/self.weight_squares[i];shrink=120/(120+neff)
            value=(1-shrink)*raw[a]+shrink*targets[groups[a]]
            if math.isfinite(value) and value>0:delta[a]=value
        return F,delta


def run_rolling(*,parent_manifest:Path,parent_sha256:str,output_root:Path):
    if file_sha256(parent_manifest)!=parent_sha256:raise ValueError('CORE6_ROLLING_PARENT_HASH')
    parent=json.loads(parent_manifest.read_text());root=parent_manifest.parent
    if parent['temporal_mode']!='reconstructed_diagnostic_only' or parent['production_eligible']:
        raise ValueError('CORE6_ROLLING_RECONSTRUCTED_PARENT_REQUIRED')
    for path,record in parent['outputs'].items():
        if file_sha256(root/path)!=record['sha256']:raise ValueError('CORE6_ROLLING_PARENT_OUTPUT_CHANGED:'+path)
    statuses=json.loads((root/'daily_status.json').read_text())
    common=[s for s in statuses if s['status']=='passed']
    # Missing whole dates stay missing; pairs must remain calendar-adjacent.
    meta=json.loads((root/'inputs/meta.json').read_text());calendar=meta['calendar']
    columns={'old44':meta['old_columns'],'core6':[c for c in meta['old_columns']
        if not c.startswith(('risk_board_','risk_index_')) and c!='risk_listing_age']+['risk_momentum']}
    identity={'parent_sha256':parent_sha256,'code_sha256':file_sha256(Path(__file__)),
        'parameters':meta['parameters'],'minimum_warmup':60,'horizon_sessions':1}
    run_id='core6_rolling_'+json_hash(identity)[:16];directory=output_root/f'run_id={run_id}'
    manifest_path=directory/'_MANIFEST.json'
    if manifest_path.exists():
        for relative,record in json.loads(manifest_path.read_text())['outputs'].items():
            if file_sha256(directory/relative)!=record['sha256']:raise ValueError('CORE6_ROLLING_EXISTING_OUTPUT_CHANGED')
        return manifest_path
    directory.mkdir(parents=True,exist_ok=True)
    summaries={}
    for model,factors in columns.items():
        assets=pl.scan_parquet(str(root/'daily'/'*'/(model+'_X.parquet'))).select('asset_id').unique().collect()['asset_id'].sort().to_list()
        state=RollingRiskState(assets,factors,meta['parameters']);covariances=[];daily=[]
        out=directory/model;out.mkdir(exist_ok=True)
        for i,status in enumerate(common):
            date=status['return_date'];day=root/'daily'/date
            state.update(dict(pl.read_parquet(day/(model+'_factor.parquet')).iter_rows()),
                dict(pl.read_parquet(day/(model+'_specific.parquet')).select('asset_id','specific_return').iter_rows()))
            if state.dates<60 or i+1>=len(common):continue
            nxt=common[i+1]
            if nxt['exposure_date']!=date or calendar.index(nxt['return_date'])!=calendar.index(date)+1:continue
            # Only X at the cutoff is consumed here. No next-day return is read.
            x=pl.read_parquet(root/'daily'/nxt['return_date']/(model+'_X.parquet'))
            F,delta=state.forecast(liquidity_groups(x))
            ids=x['asset_id'].to_list();X=x.select(factors).to_numpy()
            factor_var=np.einsum('ij,jk,ik->i',X,F,X)
            rows=[]
            for a,fv in zip(ids,factor_var):
                j=state.positions[a];d=delta[a]
                rows.append({'asset_id':a,'Delta':d,'factor_variance':float(fv),
                    'total_variance':float(fv+d) if d is not None else None,
                    'specific_observations':int(state.count[j]),
                    'last_valid_g3_index':int(state.last[j]),'risk_available':d is not None})
            table=pl.DataFrame(rows).with_columns(pl.lit(date).alias('cutoff'),pl.lit(nxt['return_date']).alias('outcome_date'))
            table.write_parquet(out/(date+'.parquet'),compression='zstd')
            covariances.append({'cutoff':date,'outcome_date':nxt['return_date'],'observations':state.dates,
                'F':F.tolist(),'min_eigenvalue':float(np.linalg.eigvalsh(F).min())})
            daily.append({'cutoff':date,'outcome_date':nxt['return_date'],'observations':state.dates,
                'assets':len(ids),'risk_available':sum(d is not None for d in delta.values()),
                'missing_Delta':sum(d is None for d in delta.values())})
            if len(daily)%100==0:print(f'core6 rolling {model}: {len(daily)} forecast dates',flush=True)
        (directory/(model+'_covariance.json')).write_text(json.dumps({'factor_ids':factors,'rows':covariances},separators=(',',':'))+'\n')
        (directory/(model+'_coverage.json')).write_text(json.dumps(daily,indent=2)+'\n')
        summaries[model]={'forecast_dates':len(daily),'first_cutoff':daily[0]['cutoff'] if daily else None,
            'last_outcome_date':daily[-1]['outcome_date'] if daily else None,
            'asset_forecasts':sum(r['assets'] for r in daily),'unavailable_asset_forecasts':sum(r['missing_Delta'] for r in daily)}
    outputs={str(p.relative_to(directory)):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in directory.rglob('*') if p.is_file()}
    manifest={'schema_version':1,'run_id':run_id,'parent_run_ids':[parent['run_id']],'identity':identity,
        'status':'completed_reconstructed_rolling_forecasts','temporal_mode':'reconstructed_diagnostic_only',
        'historical_pit_verified':False,'production_eligible':False,'calibration_status':'not_calibrated',
        'horizon_sessions':1,'created_at':utc_now().isoformat(),'summaries':summaries,'outputs':outputs,
        'holdout_evaluated':False,'current_changed':False}
    # A manifest commits the complete directory; there is deliberately no CURRENT.
    DataLake(output_root).write_immutable_json(manifest_path,manifest)
    return manifest_path
