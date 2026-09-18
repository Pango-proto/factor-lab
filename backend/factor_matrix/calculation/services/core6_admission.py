"""Approved native-strata retrospective reassessment, never automatic release."""
from pathlib import Path
import json
import math
import numpy as np
import polars as pl
from ...storage import DataLake,file_sha256,json_hash,utc_now
from .core6_calibration import (_verified_parent,_read_frame,_write_json,deciles,
                                portfolio_pair,standardized_summary,MODELS)


def evaluate_criteria(z_by_group, dates, policy):
    required={('overall',0)}|{(a,g) for a in policy['stratified']['axes'] if a!='overall' for g in range(1,11)}
    if set(z_by_group)!=required or len(dates)!=len(set(dates)) or dates!=sorted(dates):
        raise ValueError('ADMISSION_GROUP_OR_DATE_AXIS')
    if any(len(z)!=len(dates) for z in z_by_group.values()):raise ValueError('ADMISSION_WINDOW_AXIS')
    def score(values, lower, upper, minimum):
        if any(v is not None and not math.isfinite(v) for v in values):raise ValueError('ADMISSION_NONFINITE')
        summary=standardized_summary([v for v in values if v is not None],minimum)
        state='unavailable' if summary['std'] is None else ('passed' if lower<=summary['std']<=upper else 'failed')
        return {**summary,'criterion_status':state,'lower':lower,'upper':upper}
    generic=policy['generic'];strat=policy['stratified'];window=strat['window_sessions']
    overall=score(z_by_group['overall',0],generic['lower'],generic['upper'],generic['minimum_dates'])
    rows=[]
    for (axis,group),z in sorted(z_by_group.items()):
        for end in range(window-1,len(dates)):
            result=score(z[end-window+1:end+1],strat['lower'],strat['upper'],strat['minimum_dates'])
            hard=axis=='predicted_specific_risk_decile' and group==strat['high_risk_group'] and result['std'] is not None and result['std']>strat['hard_stop_above']
            rows.append({'axis':axis,'group':group,'window_start':dates[end-window+1],
                         'window_end':dates[end],'hard_stop':hard,**result})
    failed=sum(r['criterion_status']=='failed' for r in rows);missing=sum(r['criterion_status']=='unavailable' for r in rows)
    status='failed' if overall['criterion_status']=='failed' or failed else (
        'unavailable' if not rows or missing or overall['criterion_status']=='unavailable' else 'passed')
    return {'numeric_criteria_status':status,'generic':overall,'required_window_count':len(rows),
            'failed_windows':failed,'unavailable_windows':missing,
            'passed_windows':sum(r['criterion_status']=='passed' for r in rows),
            'high_risk_hard_stop_windows':sum(r['hard_stop'] for r in rows),
            'all_required_windows_present':bool(rows) and not missing},rows


def run_admission(*, project_root, policy_path, policy_sha256, output_root):
    if file_sha256(policy_path)!=policy_sha256:raise ValueError('ADMISSION_POLICY_HASH')
    policy=json.loads(policy_path.read_text())
    if policy['protocol_id']!='core6_risk_admission_v2' or policy['status']!='approved':raise ValueError('ADMISSION_POLICY')
    approval_path=project_root/policy['approval']['path']
    if file_sha256(approval_path)!=policy['approval']['sha256']:raise ValueError('ADMISSION_APPROVAL_HASH')
    approval=json.loads(approval_path.read_text())
    if approval['status']!='approved' or file_sha256(project_root/approval['proposal_path'])!=approval['proposal_sha256']:
        raise ValueError('ADMISSION_APPROVAL_SCOPE')
    root,parent=_verified_parent(project_root,policy['parent_calibration'])
    original=json.loads((root/'contract.json').read_text())
    rec,rm=_verified_parent(project_root,original['parents']['reconstruction'])
    roll,fm=_verified_parent(project_root,original['parents']['rolling'])
    if parent['identity']['parents']!=original['parents'] or fm['identity']['parent_sha256']!=original['parents']['reconstruction']['sha256']:
        raise ValueError('ADMISSION_PARENT_LINK')
    identity={'policy_sha256':policy_sha256,'parent_sha256':policy['parent_calibration']['sha256'],
              'code':{p.name:file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('core6_calibration.py')]}}
    run_id='core6_admission_v2_'+json_hash(identity)[:16];out=output_root/f'run_id={run_id}';manifest=out/'_MANIFEST.json'
    if manifest.exists():
        for rel,v in json.loads(manifest.read_text())['outputs'].items():
            if file_sha256(out/rel)!=v['sha256']:raise ValueError('ADMISSION_EXISTING_CHANGED')
        return manifest
    if out.exists():raise ValueError('ADMISSION_INCOMPLETE_RETAINED')
    out.mkdir(parents=True);_write_json(out/'policy.json',policy)
    summaries={};rows=[];windows=[]
    for model in MODELS:
        cov=json.loads((roll/(model+'_covariance.json')).read_text());columns=cov['factor_ids']
        zs={('overall',0):[]}|{(axis,g):[] for axis in policy['stratified']['axes'] if axis!='overall' for g in range(1,11)}
        dates=[]
        for index,c in enumerate(cov['rows']):
            cutoff,outcome=c['cutoff'],c['outcome_date']
            if not cutoff<outcome<policy['holdout_start']:raise ValueError('ADMISSION_TIMING')
            dates.append(outcome)
            pair=_read_frame(root,parent,f'pairs/{cutoff}.parquet')
            x=_read_frame(rec,rm,f'daily/{outcome}/{model}_X.parquet')
            t=x.join(pair,on='asset_id',how='inner',validate='1:1',maintain_order='left')
            if t.height!=pair.height or [str(v) for v in t['trade_date'].unique()]!=[cutoff]:raise ValueError('ADMISSION_X_AXIS')
            t=t.filter(pl.col(model+'_Delta').is_not_null()&(pl.col(model+'_Delta')>0)&pl.col(model+'_total_variance').is_not_null()&(pl.col(model+'_total_variance')>0)).sort('asset_id')
            ids=t['asset_id'].to_list();D=t[model+'_Delta'].to_numpy();F=np.asarray(c['F']);X=t.select(columns).to_numpy()
            groups={'liquidity_decile':deciles(ids,t['risk_liquidity'].to_numpy()),'predicted_specific_risk_decile':deciles(ids,D)}
            for axis,g in zs:
                mask=np.ones(len(ids),dtype=bool) if axis=='overall' else groups[axis]==g
                result=portfolio_pair(X[mask],F,D[mask],t[model+'_realized_return'].to_numpy()[mask],t[model+'_specific_return'].to_numpy()[mask])
                zs[axis,g].append(result.get('total_z'))
                rows.append({'model':model,'cutoff':cutoff,'outcome_date':outcome,'axis':axis,'group':g,
                             'membership_sha256':json_hash([a for a,keep in zip(ids,mask) if keep]),**result})
            if (index+1)%100==0:print(f'admission {model}: {index+1}/{len(cov["rows"])} dates',flush=True)
        if len(dates)!=444 or dates[-1]!='2025-02-28':raise ValueError('ADMISSION_SAMPLE_BOUNDARY')
        summary,model_windows=evaluate_criteria(zs,dates,policy)
        summaries[model]={**summary,'basis_acceptance_status':'unknown',
            'covariance_acceptance_status':'failed' if summary['numeric_criteria_status']=='failed' else 'unavailable',
            'pit_acceptance_status':'unavailable','risk_optimization_eligible':False,
            'interpretation':policy['reassessment_mode'],'full_residual_gate':'not_run'}
        windows.extend({'model':model,**w} for w in model_windows)
    pl.DataFrame(rows).write_parquet(out/'native_portfolio_pairs.parquet',compression='zstd')
    pl.DataFrame(windows).write_parquet(out/'native_250_session_tests.parquet',compression='zstd')
    _write_json(out/'summary.json',summaries)
    outputs={str(p.relative_to(out)):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
    return DataLake(output_root).write_immutable_json(manifest,{'schema_version':2,'run_id':run_id,'identity':identity,
        'created_at':utc_now().isoformat(),'status':'completed_retrospective_reassessment',
        'parent_run_ids':[parent['run_id'],rm['run_id'],fm['run_id']],
        'historical_pit_verified':False,'production_eligible':False,'current_changed':False,'holdout_evaluated':False,
        'temporal_mode':'reconstructed_diagnostic_only','outputs':outputs})
