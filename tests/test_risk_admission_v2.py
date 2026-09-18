from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pytest

from factor_matrix.risk_model.acceptance import RiskAcceptanceState, REQUIRED_CHECKS, registry_acceptance
from factor_matrix.risk_model import load_risk_factor_set
from factor_matrix.factor_engine import FactorRegistry,FactorRegistryStore
from factor_matrix.calculation.services.core6_admission import evaluate_criteria


def evidence(tmp_path,stage,basis='test_basis',version=1,passed=True):
    p=tmp_path/(stage+'.json')
    value={'schema':'risk_acceptance_evidence_v2','run_id':'test_'+stage,'stage':stage,
           'risk_set_version':version,'risk_basis_id':basis,'status':'passed' if passed else 'failed',
           'checks':dict.fromkeys(REQUIRED_CHECKS[stage],True),'holdout_evaluated':False,
           'historical_pit_verified':stage=='pit'}
    p.write_text(json.dumps(value))
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def test_separate_basis_covariance_and_pit():
    assert not RiskAcceptanceState(1).basis_consumption_allowed
    assert not RiskAcceptanceState(1).risk_optimization_eligible


def test_basis_only_does_not_require_covariance_and_never_permits_optimization(tmp_path):
    ref=evidence(tmp_path,'basis')
    s=RiskAcceptanceState(1,'test_basis',basis_acceptance_status='passed',basis_evidence=ref)
    assert s.basis_consumption_allowed and not s.risk_optimization_eligible
    c=evidence(tmp_path,'covariance')
    s=replace(s,covariance_acceptance_status='passed',covariance_evidence=c)
    assert not s.risk_optimization_eligible
    s=replace(s,pit_acceptance_status='passed',pit_evidence=evidence(tmp_path,'pit'))
    assert s.risk_optimization_eligible


def test_failed_diagnostic_or_wrong_basis_cannot_certify(tmp_path):
    for ref in [evidence(tmp_path,'basis',passed=False),evidence(tmp_path,'basis',basis='wrong')]:
        with pytest.raises(ValueError,match='EVIDENCE'):
            RiskAcceptanceState(1,'test_basis',basis_acceptance_status='passed',basis_evidence=ref)
    with pytest.raises(ValueError,match='REQUIRES_BASIS'):
        RiskAcceptanceState(1,'test_basis',covariance_acceptance_status='passed',covariance_evidence=evidence(tmp_path,'covariance'))


def test_candidate_basis_certification_allows_research_and_locks_members(tmp_path):
    registry=FactorRegistry.discover();store=FactorRegistryStore(tmp_path/'registry.sqlite')
    spec=load_risk_factor_set(Path('config/risk_factor_set_candidate_v1.json'),registry)
    spec=replace(spec,risk_basis_id='test_basis')
    store.sync_definitions(registry);store.sync_risk_factor_set(spec)
    ref=evidence(tmp_path,'basis')
    s=RiskAcceptanceState(1,'test_basis',basis_acceptance_status='passed',basis_evidence=ref)
    store.record_risk_acceptance(s)
    store.sync_risk_factor_set(spec)  # Legacy defaults preserve certified stages.
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT status FROM risk_set').fetchone()[0]=='candidate'
        assert registry_acceptance(c,1).basis_consumption_allowed
        with pytest.raises(sqlite3.IntegrityError,match='IMMUTABLE'):
            c.execute("UPDATE risk_set_member SET standardize='changed' WHERE risk_set_version=1")
    attempt=store.issue_research_attempt(family_root_id='size',feature_id='size',feature_version=1,
        feature_spec_hash='a'*64,risk_set_version=1,horizon_days=1,code_sha='b'*64,config_sha='c'*64)
    assert attempt
    with pytest.raises(ValueError,match='IMMUTABLE'):
        store.record_risk_acceptance(RiskAcceptanceState(1,'test_basis'))
    Path(ref['path']).write_text('{}')
    with sqlite3.connect(store.path) as c:
        with pytest.raises(ValueError,match='HASH'):registry_acceptance(c,1)


def test_legacy_frozen_defaults_unknown_after_migration(tmp_path):
    store=FactorRegistryStore(tmp_path/'r.sqlite');store.initialize()
    with sqlite3.connect(store.path) as c:
        c.execute("INSERT INTO risk_set(risk_set_version,status,purpose,rationale,created_at) VALUES(1,'frozen','research','old','2000-01-01')")
    store.initialize()
    with sqlite3.connect(store.path) as c:
        assert not registry_acceptance(c,1).basis_consumption_allowed
        assert c.execute('SELECT basis_acceptance_status,covariance_acceptance_status,pit_acceptance_status FROM risk_set').fetchone()==('unknown','unknown','unknown')


def policy():return json.loads(Path('config/core6_risk_admission_v2.json').read_text())


def panel(scale=1.):
    z=np.tile([-1.,1.],125)*np.sqrt(249/250)*scale
    return {('overall',0):z.tolist()}|{(axis,g):z.tolist() for axis in policy()['stratified']['axes'] if axis!='overall' for g in range(1,11)}


def test_parallel_thresholds_and_missing_window_no_compression():
    dates=[f'{i:03d}' for i in range(250)]
    result,_=evaluate_criteria(panel(),dates,policy())
    assert result['numeric_criteria_status']=='passed' and result['passed_windows']==21
    result,_=evaluate_criteria(panel(1.07),dates,policy())
    assert result['numeric_criteria_status']=='failed' and result['passed_windows']==21
    p=panel();p['liquidity_decile',1][100]=None
    result,rows=evaluate_criteria(p,dates,policy())
    assert result['numeric_criteria_status']=='unavailable' and result['unavailable_windows']==1
    assert len(rows)==21


def test_high_risk_hard_stop_and_equality_boundary():
    p=panel();p['predicted_specific_risk_decile',10]=panel(1.201)['overall',0]
    result,_=evaluate_criteria(p,[f'{i:03d}' for i in range(250)],policy())
    assert result['high_risk_hard_stop_windows']==1
    p['predicted_specific_risk_decile',10]=panel(1.19)['overall',0]
    result,_=evaluate_criteria(p,[f'{i:03d}' for i in range(250)],policy())
    assert result['high_risk_hard_stop_windows']==0 and result['numeric_criteria_status']=='failed'


def test_missing_axis_and_nonfinite_reject():
    dates=[f'{i:03d}' for i in range(250)];p=panel();p.pop(('liquidity_decile',1))
    with pytest.raises(ValueError,match='AXIS'):evaluate_criteria(p,dates,policy())
    p=panel();p['overall',0][0]=float('nan')
    with pytest.raises(ValueError,match='NONFINITE'):evaluate_criteria(p,dates,policy())


def test_migration_preserves_old_columns_and_has_backup(tmp_path):
    from factor_matrix.factor_engine.store import SCHEMA
    from factor_matrix.calculation.services.risk_acceptance_migration import migrate_acceptance
    db=tmp_path/'old.sqlite'
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
        c.execute("INSERT INTO risk_set(risk_set_version,status,purpose,rationale,created_at) VALUES(1,'frozen','research','legacy','2000-01-01')")
        assert 'basis_acceptance_status' not in [r[1] for r in c.execute('PRAGMA table_info(risk_set)')]
    root=Path(__file__).resolve().parents[1];ap=root/'config/core6_admission_v2_approval.json'
    manifest=migrate_acceptance(registry_path=db,project_root=root,approval_path=ap,
        approval_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),output_root=tmp_path/'migration')
    report=json.loads(manifest.read_text())
    assert report['status']=='additive_migration_passed'
    assert report['states']==[[1,'unknown','unknown','unknown']]
    assert (manifest.parent/'registry_before.sqlite').exists()
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT status,rationale FROM risk_set').fetchone()==('frozen','legacy')


def test_acceptance_events_append_only_and_covariance_update(tmp_path):
    registry=FactorRegistry.discover();store=FactorRegistryStore(tmp_path/'r.sqlite')
    spec=replace(load_risk_factor_set(Path('config/risk_factor_set_candidate_v1.json'),registry),risk_basis_id='test_basis')
    store.sync_definitions(registry);store.sync_risk_factor_set(spec)
    s=RiskAcceptanceState(1,'test_basis',basis_acceptance_status='passed',basis_evidence=evidence(tmp_path,'basis'))
    store.record_risk_acceptance(s)
    store.record_risk_acceptance(replace(s,covariance_acceptance_status='passed',covariance_evidence=evidence(tmp_path,'covariance')))
    with sqlite3.connect(store.path) as c:
        assert c.execute('SELECT count(*) FROM risk_acceptance_event').fetchone()[0]==3
        with pytest.raises(sqlite3.IntegrityError,match='APPEND_ONLY'):c.execute('DELETE FROM risk_acceptance_event')


def test_readonly_api_never_promotes_legacy_frozen(tmp_path):
    from factor_matrix.api import risk_acceptance_response
    path=tmp_path/'registry.sqlite'
    assert risk_acceptance_response(path)['status']=='unavailable' and not path.exists()
    FactorRegistryStore(path).initialize()
    with sqlite3.connect(path) as c:
        c.execute("INSERT INTO risk_set(risk_set_version,status,purpose,rationale,created_at) VALUES(1,'frozen','research','legacy','2000-01-01')")
    before=path.read_bytes();result=risk_acceptance_response(path)
    assert result['rows'][0]['basis_acceptance_status']=='unknown'
    assert not result['rows'][0]['basis_consumption_allowed'] and not result['production_allowed']
    assert path.read_bytes()==before


def test_snapshot_propagates_basis_only_and_blocks_risk_target(tmp_path):
    from dataclasses import asdict
    from factor_matrix.calculation.services.m3_fixture import snapshot_pair,history_fixture,ASSETS,ESTIMATOR
    from factor_matrix.calculation.services.risk_snapshot import exposure_snapshot,forecast_snapshot,admit_target
    observations=history_fixture();x,_,decision=snapshot_pair(observations)
    state=RiskAcceptanceState(x['risk_set_version'],x['risk_basis_id'],basis_acceptance_status='passed',
        basis_evidence=evidence(tmp_path,'basis',basis=x['risk_basis_id'],version=x['risk_set_version']))
    args={k:x[k] for k in ('asset_ids','factor_ids','risk_basis_id','risk_set_version','risk_factor_set_status','valid_at','available_at','decision_time','parent_run_ids','purpose')}
    x=exposure_snapshot(**args,values=x['X'],expected_valid_at=x['valid_at'],acceptance=asdict(state))
    f=forecast_snapshot(x,observations,decision_time=decision,estimation_cutoff=x['valid_at'],expected_cutoff=x['valid_at'],estimator=ESTIMATOR,group_by_asset=dict.fromkeys(ASSETS,'group'))
    assert x['basis_consumption_allowed'] and f['acceptance']['basis_acceptance_status']=='passed'
    result=admit_target(x,f,dict.fromkeys(ASSETS,.1),decision_time=decision,exposure_limits={'risk_size':[-100.,100.]})
    assert not result['accepted']
    assert 'RISK_COVARIANCE_NOT_ACCEPTED' in result['reasons'] and 'RISK_REAL_PIT_NOT_VERIFIED' in result['reasons']
