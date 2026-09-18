"""M3 v1: PIT X/F/Delta, portfolio forecasts and fail-closed target admission.

The service never changes the factor basis, fills missing columns with zero,
repairs a covariance matrix, or implicitly rescales a prediction horizon.
"""
from datetime import datetime
from dataclasses import asdict
import math
from pathlib import Path
import json
import numpy as np
from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from ..l2.risk_modeling import ewma_factor_covariance, ewma_specific_variance, bias_test
from ...risk_model.acceptance import acceptance_from_mapping

VERSION='risk_snapshot_v1'


def time(value):
    dt=datetime.fromisoformat(value)
    if dt.tzinfo is None:raise ValueError('RISK_TIMEZONE_REQUIRED')
    return dt


def axis(values):
    if not values or len(values)!=len(set(values)) or any(not isinstance(x,str) or not x for x in values):
        raise ValueError('RISK_AXIS_INVALID')
    return list(values)


def finite(values, shape=None):
    a=np.asarray(values,dtype=float)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():raise ValueError('RISK_NONFINITE_OR_SHAPE')
    return a


def identity(x, estimator):
    return {'risk_basis_id':x['risk_basis_id'],'risk_set_version':x['risk_set_version'],
        'risk_factor_set_status':x['risk_factor_set_status'],'factor_ids':x['factor_ids'],
        'asset_ids':x['asset_ids'],'estimator':estimator,'horizon_sessions':1,
        'acceptance':x.get('acceptance',{})}


def exposure_snapshot(*, asset_ids, factor_ids, values, risk_basis_id, risk_set_version,
                      risk_factor_set_status, valid_at, available_at, decision_time,
                      expected_valid_at, parent_run_ids, purpose, acceptance=None):
    assets,factors=axis(asset_ids),axis(factor_ids)
    if purpose not in ('engineering_fixture_only','research_diagnostic','production'):
        raise ValueError('RISK_PURPOSE_INVALID')
    if not risk_basis_id or not risk_set_version or not parent_run_ids:raise ValueError('RISK_LINEAGE_REQUIRED')
    if risk_factor_set_status not in ('candidate','frozen'):raise ValueError('RISK_SET_STATUS')
    if time(valid_at)!=time(expected_valid_at):raise ValueError('RISK_STALE_EXPOSURE')
    if not time(valid_at)<=time(available_at)<=time(decision_time):raise ValueError('RISK_EXPOSURE_NOT_AVAILABLE')
    X=finite(values,(len(assets),len(factors)))
    admission=acceptance_from_mapping(acceptance or {'risk_set_version':risk_set_version,'risk_basis_id':risk_basis_id})
    if admission.risk_set_version!=risk_set_version or admission.risk_basis_id!=risk_basis_id:
        raise ValueError('RISK_ACCEPTANCE_EXPOSURE_IDENTITY')
    if 'risk_size' not in factors or not any(f.startswith('risk_industry_') for f in factors):
        raise ValueError('RISK_SIZE_AND_INDUSTRY_REQUIRED')
    result={'schema_version':1,'kind':'RiskExposureSnapshot','asset_ids':assets,'factor_ids':factors,'X':X.tolist(),
        'risk_basis_id':risk_basis_id,'risk_set_version':risk_set_version,'risk_factor_set_status':risk_factor_set_status,
        'valid_at':valid_at,'available_at':available_at,'decision_time':decision_time,
        'parent_run_ids':parent_run_ids,'purpose':purpose,'quality':{'status':'passed','coverage':1.},
        'acceptance':asdict(admission),'basis_consumption_allowed':admission.basis_consumption_allowed}
    result['snapshot_id']='risk_x_'+json_hash(result)[:16]
    return result


def forecast_snapshot(exposure, observations, *, decision_time, estimation_cutoff,
                      expected_cutoff, estimator, group_by_asset, horizon_sessions=1):
    if horizon_sessions!=1:raise ValueError('RISK_HORIZON_CONVERSION_REQUIRED')
    if time(estimation_cutoff)!=time(expected_cutoff):raise ValueError('RISK_STALE_FORECAST')
    if time(estimation_cutoff)>time(decision_time) or time(exposure['available_at'])>time(decision_time):
        raise ValueError('RISK_FORECAST_NOT_AVAILABLE')
    if time(exposure['decision_time'])!=time(decision_time):raise ValueError('RISK_DECISION_MISMATCH')
    # Unavailable observations are excluded before reading their values.
    rows=[r for r in observations if time(r['realized_at'])<=time(estimation_cutoff) and time(r['available_at'])<=time(decision_time)]
    rows.sort(key=lambda r:time(r['realized_at']))
    if len({time(r['realized_at']) for r in rows})!=len(rows):raise ValueError('RISK_DUPLICATE_OBSERVATION')
    if not rows or time(rows[-1]['realized_at'])!=time(estimation_cutoff):raise ValueError('RISK_CUTOFF_OBSERVATION_MISSING')
    nmin=estimator['minimum_history']
    if type(nmin) is not int or nmin<60 or len(rows)<nmin:raise ValueError('RISK_HISTORY_INSUFFICIENT')
    for key in ('factor_half_life','specific_half_life'):
        if not math.isfinite(estimator[key]) or estimator[key]<=0:raise ValueError('RISK_ESTIMATOR_PARAMETER')
    if estimator.get('id')!='registered_ewma_group_shrinkage_v1':raise ValueError('RISK_ESTIMATOR_UNSUPPORTED')
    assets,factors=exposure['asset_ids'],exposure['factor_ids']
    if set(group_by_asset)!=set(assets) or any(not v for v in group_by_asset.values()):raise ValueError('RISK_GROUP_AXIS')
    for r in rows:
        if time(r['available_at'])<time(r['realized_at']):raise ValueError('RISK_RETURN_AVAILABILITY_INVALID')
        if r['risk_basis_id']!=exposure['risk_basis_id'] or r['risk_set_version']!=exposure['risk_set_version']:
            raise ValueError('RISK_BASIS_MISMATCH')
        if set(r['factor_returns'])!=set(factors) or set(r['specific_returns'])!=set(assets):
            raise ValueError('RISK_RAGGED_PANEL_UNSUPPORTED')
        finite(list(r['factor_returns'].values()));finite(list(r['specific_returns'].values()))
    fr={f:[r['factor_returns'][f] for r in rows] for f in factors}
    sr={a:[r['specific_returns'][a] for r in rows] for a in assets}
    cov=ewma_factor_covariance(fr,estimator['factor_half_life'])
    F=np.array([[cov[a,b] for b in factors] for a in factors])
    delta=ewma_specific_variance(sr,estimator['specific_half_life'],group_by_asset=group_by_asset,
        lookback_days=252,minimum_observations=60,clip_multiple=5.,prior_effective_observations=120.)
    D=np.array([delta[a] for a in assets]); validate_matrices(F,D,len(factors),len(assets))
    model=identity(exposure,estimator)
    result={'schema_version':1,'kind':'RiskForecastSnapshot','exposure_snapshot_id':exposure['snapshot_id'],
        **model,'model_id':'risk_model_'+json_hash(model)[:16],'F':F.tolist(),'Delta':D.tolist(),
        'estimation_cutoff':estimation_cutoff,'decision_time':decision_time,
        'available_at':max([exposure['available_at'],*[r['available_at'] for r in rows]],key=time),
        'estimator_version':estimator['id'],'group_by_asset':group_by_asset,
        'coverage':{'observations':len(rows),'first_realized_at':rows[0]['realized_at'],'last_realized_at':rows[-1]['realized_at'],
            'factor_fraction':1.,'asset_fraction':1.,'input_observations_sha256':json_hash(rows)},
        'quality':{'status':'passed','min_eigenvalue':float(np.linalg.eigvalsh(F).min()),'minimum_specific_variance':float(D.min())},
        'purpose':exposure['purpose'],'parent_run_ids':[exposure['snapshot_id']],
        'calibration_status':'not_calibrated','risk_driven_strategy_allowed':False}
    result['snapshot_id']='risk_fd_'+json_hash(result)[:16]
    return result


def validate_matrices(F,D,nf,na):
    F=finite(F,(nf,nf));D=finite(D,(na,))
    if not np.allclose(F,F.T,rtol=0,atol=1e-12):raise ValueError('RISK_COVARIANCE_ASYMMETRIC')
    # Floating-point eigensolver tolerance only, not a statistical acceptance threshold.
    if np.linalg.eigvalsh(F).min() < -1e-12:raise ValueError('RISK_COVARIANCE_NOT_PSD')
    if (D<=0).any():raise ValueError('RISK_SPECIFIC_VARIANCE_NOT_POSITIVE')


def portfolio_risk(exposure, forecast, weights, *, decision_time, horizon_sessions=1):
    assets,factors=exposure['asset_ids'],exposure['factor_ids']
    for key in ('risk_basis_id','risk_set_version','risk_factor_set_status','asset_ids','factor_ids'):
        if forecast[key]!=exposure[key]:raise ValueError('RISK_SNAPSHOT_AXIS_OR_BASIS_MISMATCH')
    if forecast['exposure_snapshot_id']!=exposure['snapshot_id']:raise ValueError('RISK_SNAPSHOT_PAIR_MISMATCH')
    if forecast.get('acceptance',{})!=exposure.get('acceptance',{}):raise ValueError('RISK_ACCEPTANCE_PAIR_MISMATCH')
    if forecast['horizon_sessions']!=horizon_sessions:raise ValueError('RISK_HORIZON_MISMATCH')
    if any(time(s['available_at'])>time(decision_time) for s in (exposure,forecast)):
        raise ValueError('RISK_SNAPSHOT_NOT_AVAILABLE')
    if time(forecast['decision_time'])!=time(decision_time):raise ValueError('RISK_STALE_DECISION')
    if set(weights)!=set(assets):raise ValueError('RISK_PORTFOLIO_COVERAGE')
    w=finite([weights[a] for a in assets],(len(assets),))
    if (w<0).any() or w.sum()>1+1e-12:raise ValueError('RISK_LONG_ONLY_BUDGET')
    X=finite(exposure['X'],(len(assets),len(factors)));F=np.asarray(forecast['F']);D=np.asarray(forecast['Delta'])
    validate_matrices(F,D,len(factors),len(assets))
    b=X.T@w;factor=float(b@F@b);specific=float((w*w)@D);variance=factor+specific
    if variance<0:raise ValueError('RISK_NEGATIVE_PORTFOLIO_VARIANCE')
    asset_contributions=w*(X@(F@b)+D*w)
    return {'model_id':forecast['model_id'],'forecast_snapshot_id':forecast['snapshot_id'],'horizon_sessions':horizon_sessions,
        'exposures':dict(zip(factors,b.tolist())),'factor_variance':factor,'specific_variance':specific,
        'variance':variance,'volatility':math.sqrt(variance),'cash_weight':float(1-w.sum()),
        'asset_variance_contributions':dict(zip(assets,asset_contributions.tolist())),
        'factor_variance_contributions':dict(zip(factors,(b*(F@b)).tolist())),
        'calibration_status':forecast['calibration_status'],'purpose':forecast['purpose']}


def calibration_report(pairs, *, model_id, policy, purpose, published_at):
    """Realized outcomes must be strictly subsequent to each prediction decision.

    Caller supplies an explicit sealed evaluation sample, never auto-picks dates.
    Missing/invalid pairs fail; the lower-level estimator's filtering is not used.
    """
    if purpose not in ('engineering_fixture_only','research_diagnostic'):raise ValueError('RISK_CALIBRATION_PURPOSE')
    if len({p['prediction_id'] for p in pairs})!=len(pairs):raise ValueError('RISK_CALIBRATION_DUPLICATES')
    for p in pairs:
        if p['model_id']!=model_id or p['horizon_sessions']!=1:raise ValueError('RISK_CALIBRATION_MODEL_OR_HORIZON')
        if not time(p['forecast_available_at'])<=time(p['decision_time'])<time(p['outcome_start'])<=time(p['outcome_end'])<=time(p['outcome_available_at'])<=time(published_at):
            raise ValueError('RISK_CALIBRATION_TIME_LEAK')
        if p['outcome_start'][:10]!=p['outcome_end'][:10]:raise ValueError('RISK_CALIBRATION_HORIZON')
        finite([p['realized_return'],p['predicted_variance']])
        if p['predicted_variance']<=0:raise ValueError('RISK_CALIBRATION_VARIANCE')
    if len({p['outcome_end'] for p in pairs})!=len(pairs):raise ValueError('RISK_CALIBRATION_OVERLAPPING_DAYS')
    if policy!={'lower_bound':.95,'upper_bound':1.05,'minimum_observations':60}:
        raise ValueError('RISK_CALIBRATION_POLICY_NOT_REGISTERED')
    result=asdict(bias_test([p['realized_return'] for p in pairs],[p['predicted_variance'] for p in pairs],**policy))
    result.update({'model_id':model_id,'policy':policy,'purpose':purpose,'available_at':published_at,
        'evaluation_end':max((p['outcome_end'] for p in pairs),key=time,default=None),
        'pairs_sha256':json_hash(pairs),'status':'passed' if result['passed'] else 'failed',
        'production_eligible':False,'required_remaining':'independent_risk_and_liquidity_deciles_and_freeze_evidence'})
    return result


def admit_target(exposure, forecast, weights, *, decision_time, exposure_limits,
                 max_variance=None, horizon_sessions=1, engineering_only=False):
    """No fallback target. Violating targets are returned as rejected, with reasons."""
    report=portfolio_risk(exposure,forecast,weights,decision_time=decision_time,horizon_sessions=horizon_sessions)
    reasons=[]
    if engineering_only:
        if forecast['purpose']!='engineering_fixture_only':raise ValueError('RISK_ENGINEERING_BYPASS_FORBIDDEN')
    else:
        # v1 cannot issue a production permit: current basis lacks freeze evidence.
        reasons.append('RISK_PRODUCTION_RELEASE_NOT_APPROVED')
        state=acceptance_from_mapping(forecast.get('acceptance') or {
            'risk_set_version':forecast['risk_set_version'],'risk_basis_id':forecast['risk_basis_id']})
        if not state.basis_consumption_allowed:reasons.append('RISK_BASIS_NOT_ACCEPTED')
        if state.covariance_acceptance_status!='passed':reasons.append('RISK_COVARIANCE_NOT_ACCEPTED')
        if state.pit_acceptance_status!='passed':reasons.append('RISK_REAL_PIT_NOT_VERIFIED')
        if forecast['calibration_status']!='passed':reasons.append('RISK_NOT_CALIBRATED')
        if forecast['risk_factor_set_status']!='frozen':reasons.append('RISK_SET_NOT_FROZEN')
    if not exposure_limits:raise ValueError('RISK_EXPLICIT_LIMITS_REQUIRED')
    for factor,bounds in exposure_limits.items():
        if factor not in report['exposures']:raise ValueError('RISK_LIMIT_UNKNOWN_FACTOR')
        bounds=finite(bounds,(2,))
        if bounds[0]>bounds[1]:raise ValueError('RISK_LIMIT_INVALID')
        if not bounds[0]-1e-12<=report['exposures'][factor]<=bounds[1]+1e-12:reasons.append('EXPOSURE_LIMIT:'+factor)
    if max_variance is not None:
        if not math.isfinite(max_variance) or max_variance<=0:raise ValueError('RISK_VARIANCE_LIMIT_INVALID')
        if report['variance']>max_variance:reasons.append('VARIANCE_LIMIT')
    return {'accepted':not reasons,'reasons':reasons,'report':report,'target_weights':weights if not reasons else None,
        'status':'engineering_only' if engineering_only else 'blocked'}


def publish_request(input_path:Path, expected_sha256:str, output_dir:Path):
    if file_sha256(input_path)!=expected_sha256:raise ValueError('RISK_INPUT_CHECKSUM')
    request=json.loads(input_path.read_text())
    x=exposure_snapshot(**request['exposure'])
    f=forecast_snapshot(x,request['observations'],**request['forecast'])
    reports=[admit_target(x,f,**target) for target in request['targets']]
    lake=DataLake(output_dir)
    outputs={name:lake.artifact_record(lake.write_immutable_json(output_dir/(name+'.json'),data))
        for name,data in [('exposure',x),('forecast',f),('targets',{'targets':reports})]}
    return lake.write_immutable_json(output_dir/'_MANIFEST.json',{'schema_version':1,
        'run_id':VERSION+'_'+json_hash(request)[:16],'risk_basis_id':x['risk_basis_id'],'risk_set_version':x['risk_set_version'],
        'status':'engineering_passed_production_blocked','code_hash':source_tree_hash(),
        'input':{'path':str(input_path),'sha256':expected_sha256},'parent_run_ids':x['parent_run_ids'],
        'purpose':x['purpose'],'outputs':outputs,'research_promoted':False})
