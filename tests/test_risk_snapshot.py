from copy import deepcopy
from datetime import datetime,timedelta
import math
import numpy as np
import pytest
from factor_matrix.calculation.services.m3_fixture import history_fixture,snapshot_pair,ASSETS,FACTORS,ESTIMATOR,calibration_acceptance,comparison_fixture,BIAS_POLICY
from factor_matrix.calculation.services.risk_snapshot import exposure_snapshot,forecast_snapshot,portfolio_risk,admit_target,calibration_report,validate_matrices

@pytest.fixture
def bundle():return snapshot_pair(history_fixture())

def forecast(x,rows,decision,**kwargs):
    return forecast_snapshot(x,rows,decision_time=decision,estimation_cutoff=x['valid_at'],expected_cutoff=x['valid_at'],estimator=ESTIMATOR,group_by_asset={a:'fixture_group' for a in ASSETS},**kwargs)

def test_portfolio_matches_dense_and_euler_contributions(bundle):
    x,f,t=bundle;w={ASSETS[0]:.3,ASSETS[1]:.4}
    r=portfolio_risk(x,f,w,decision_time=t)
    X,F,D=np.array(x['X']),np.array(f['F']),np.array(f['Delta']);v=np.array(list(w.values()))
    assert r['variance']==pytest.approx(v@(X@F@X.T+np.diag(D))@v,abs=1e-15)
    assert sum(r['asset_variance_contributions'].values())==pytest.approx(r['variance'])
    assert sum(r['factor_variance_contributions'].values())==pytest.approx(r['factor_variance'])
    assert r['cash_weight']==pytest.approx(.3)

def test_future_observations_and_unavailable_values_do_not_affect_forecast(bundle):
    x,f,t=bundle;rows=history_fixture();bad=deepcopy(rows[-1]);bad['realized_at']='2027-01-01T07:00:00+00:00';bad['available_at']=bad['realized_at'];bad['factor_returns']={'bad':math.nan}
    assert forecast(x,rows+[bad],t)==f
    late=deepcopy(bad);late['realized_at']=rows[0]['realized_at']
    assert forecast(x,rows+[late],t)==f

@pytest.mark.parametrize('change,message',[
 ('factor','RAGGED'),('asset','RAGGED'),('nan','NONFINITE'),('basis','BASIS'),('duplicate','DUPLICATE'),('cutoff','CUTOFF')])
def test_invalid_inputs_fail_closed(bundle,change,message):
    x,f,t=bundle;rows=history_fixture()
    if change=='factor':rows[10]['factor_returns'].pop(FACTORS[0])
    if change=='asset':rows[10]['specific_returns'].pop(ASSETS[0])
    if change=='nan':rows[10]['specific_returns'][ASSETS[0]]=math.nan
    if change=='basis':rows[10]['risk_basis_id']='different'
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    if change=='cutoff':rows.pop()
    with pytest.raises(ValueError,match=message):forecast(x,rows,t)

def test_insufficient_history_and_stale_x():
    with pytest.raises(ValueError,match='HISTORY'):snapshot_pair(history_fixture(59))
    x,f,t=snapshot_pair(history_fixture())
    args={k:v for k,v in x.items() if k in ('asset_ids','factor_ids','risk_basis_id','risk_set_version','risk_factor_set_status','valid_at','available_at','decision_time','parent_run_ids','purpose')}
    with pytest.raises(ValueError,match='STALE'):exposure_snapshot(**args,values=x['X'],expected_valid_at='2027-01-01T00:00:00+00:00')
    args['available_at']='2027-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='NOT_AVAILABLE'):exposure_snapshot(**args,values=x['X'],expected_valid_at=x['valid_at'])

@pytest.mark.parametrize('F,D,reason',[([[1,2],[0,1]],[1,1],'ASYMMETRIC'),([[1,2],[2,1]],[1,1],'NOT_PSD'),([[1,0],[0,1]],[0,1],'NOT_POSITIVE'),([[math.nan,0],[0,1]],[1,1],'NONFINITE')])
def test_covariance_checks(F,D,reason):
    with pytest.raises(ValueError,match=reason):validate_matrices(F,D,2,2)

def test_axis_permutation_by_id_and_horizon_rejection(bundle):
    x,f,t=bundle;rows=history_fixture()
    # Dict insertion order is irrelevant; explicit output axes stay authoritative.
    for row in rows:
        row['factor_returns']=dict(reversed(list(row['factor_returns'].items())))
        row['specific_returns']=dict(reversed(list(row['specific_returns'].items())))
    assert forecast(x,rows,t)==f
    weights={ASSETS[0]:.2,ASSETS[1]:.3}
    with pytest.raises(ValueError,match='HORIZON'):portfolio_risk(x,f,weights,decision_time=t,horizon_sessions=5)
    with pytest.raises(ValueError,match='HORIZON'):forecast(x,rows,t,horizon_sessions=5)
    f['factor_ids']=f['factor_ids'][::-1]
    with pytest.raises(ValueError,match='AXIS_OR_BASIS'):portfolio_risk(x,f,weights,decision_time=t)

def test_admission_limits_and_production_no_bypass(bundle):
    x,f,t=bundle;w={ASSETS[0]:.8,ASSETS[1]:0.}
    limits={'risk_size':[-.4,.4],'risk_industry_A':[0,.6]}
    result=admit_target(x,f,w,decision_time=t,exposure_limits=limits,engineering_only=True,max_variance=1e-10)
    assert not result['accepted'] and result['target_weights'] is None
    assert {'EXPOSURE_LIMIT:risk_size','EXPOSURE_LIMIT:risk_industry_A','VARIANCE_LIMIT'}<=set(result['reasons'])
    w={ASSETS[0]:.2,ASSETS[1]:0.}
    assert admit_target(x,f,w,decision_time=t,exposure_limits=limits,engineering_only=True)['accepted']
    gate=admit_target(x,f,w,decision_time=t,exposure_limits=limits)
    assert not gate['accepted'] and 'RISK_NOT_CALIBRATED' in gate['reasons']
    f['purpose']='research_diagnostic'
    with pytest.raises(ValueError,match='BYPASS'):admit_target(x,f,w,decision_time=t,exposure_limits=limits,engineering_only=True)

def test_calibration_known_pass_failure_and_never_promotion():
    a=calibration_acceptance('fixture')
    assert a['known_unit_std']['standardized_return_std']==pytest.approx(1)
    assert a['known_double_std']['standardized_return_std']==pytest.approx(2)
    assert not a['known_unit_std']['production_eligible']

def test_calibration_does_not_filter_invalid_or_leaked_pairs():
    p={'prediction_id':'1','model_id':'model','horizon_sessions':1,'forecast_available_at':'2026-01-01T07:00:00+00:00',
       'decision_time':'2026-01-01T07:00:00+00:00','outcome_start':'2026-01-02T01:30:00+00:00',
       'outcome_end':'2026-01-02T07:00:00+00:00','outcome_available_at':'2026-01-02T07:00:00+00:00',
       'realized_return':float('nan'),'predicted_variance':.0001}
    with pytest.raises(ValueError,match='NONFINITE'):calibration_report([p],model_id='model',policy=BIAS_POLICY,purpose='engineering_fixture_only',published_at=p['outcome_end'])
    p['realized_return']=.001;p['forecast_available_at']=p['outcome_end']
    with pytest.raises(ValueError,match='TIME_LEAK'):calibration_report([p],model_id='model',policy=BIAS_POLICY,purpose='engineering_fixture_only',published_at=p['outcome_end'])

def test_full_same_signal_cost_comparison():
    result=comparison_fixture()
    versions=result['versions']
    assert all(v['invariants']['status']=='passed' for v in versions)
    assert all(len(v['nav'])==85 for v in versions)
    assert all([d['scores'] for d in v['decisions']]==[d['scores'] for d in versions[0]['decisions']] for v in versions)
    assert versions[0]['decisions'][0]['weights']!=versions[1]['decisions'][0]['weights']
    # Current fixture's variance cap is nonbinding after exposure constraints.
    assert versions[1]['nav']==versions[2]['nav']
    assert not result['production_gate']['accepted']

def test_versioned_request_publication_checks_input_and_outputs(tmp_path):
    import json
    from pathlib import Path
    from factor_matrix.storage import file_sha256
    from factor_matrix.calculation.services.risk_snapshot import publish_request
    rows=history_fixture();x,f,t=snapshot_pair(rows)
    exposure={k:x[k] for k in ('asset_ids','factor_ids','risk_basis_id','risk_set_version','risk_factor_set_status','valid_at','available_at','decision_time','parent_run_ids','purpose')}
    exposure.update(values=x['X'],expected_valid_at=x['valid_at'])
    request={'exposure':exposure,'observations':rows,'forecast':{'decision_time':t,'estimation_cutoff':x['valid_at'],'expected_cutoff':x['valid_at'],'estimator':ESTIMATOR,'group_by_asset':{a:'fixture_group' for a in ASSETS}},
       'targets':[{'weights':{ASSETS[0]:.2,ASSETS[1]:0.},'decision_time':t,'exposure_limits':{'risk_size':[-.4,.4]},'engineering_only':True}]}
    inp=tmp_path/'request.json';inp.write_text(json.dumps(request))
    with pytest.raises(ValueError,match='CHECKSUM'):publish_request(inp,'invalid',tmp_path/'bad')
    output=tmp_path/'risk';manifest=publish_request(inp,file_sha256(inp),output)
    assert publish_request(inp,file_sha256(inp),output)==manifest
    data=json.loads(manifest.read_text())
    for ref in data['outputs'].values():assert file_sha256(output/ref['path'])==ref['sha256']
    assert not data['research_promoted']
    assert json.loads((output/'targets.json').read_text())['targets'][0]['accepted']
