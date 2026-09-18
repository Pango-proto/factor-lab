import json
from pathlib import Path

import pytest

from factor_matrix.calculation.services import evaluation_review_v2 as service
from factor_matrix.calculation.services.evaluation_summary import evaluation_summary
from factor_matrix.storage import DataLake, file_sha256


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    config = json.loads(service.DEFAULT_CONFIG.read_text())
    path = tmp_path/'evaluation_framework_v2.json'
    path.write_text(json.dumps(config))
    source = {'status':'blocked','negative_controls':{'variants':{
        'within_date_cross_sectional_return_shuffle':{'passed':True},
        'label_window_shift':{'passed':True},
        **{name:{'passed':False,'permutation_mean_ics':[-.003]*100} for name in service.VARIANTS}}}}
    report = {'run_id':'diagnostic_fixture','source_run_id':'original_fixture','status':'completed',
        'holdout_touched':False,'independent_neutralization':{'max_residual_difference':1e-14,'max_daily_ic_difference':1e-17},
        'variants':{name:{'replay_matches_saved':True,'max_replay_absolute_difference':1e-18,
            'decomposition_max_absolute_error':1e-16,'lag_decomposition_max_absolute_error':1e-16} for name in service.VARIANTS}}
    monkeypatch.setattr(service,'load_evidence',lambda lake,p:(json.loads(p.read_text()),source,report,{'outputs':{}}))
    lake = DataLake(tmp_path/'data')
    return lake,path,config,source,report


def test_all_implemented_checks_pass_but_uncalibrated_remains_blocked(evidence):
    lake,path,config,source,report = evidence
    state = service.review_states(source,report,config)
    assert state['implementation_status']=='passed'
    assert all(state['required_gates'].values())
    assert state['temporal_validation_status']=='not_calibrated'
    assert state['research_status']=='blocked'
    assert state['research_current_permitted'] is False
    assert state['alpha_registration_permitted'] is False
    for value in state['structured_diagnostics'].values():
        assert value['legacy_zero_interval_passed'] is False
        assert 'passed' not in value


@pytest.mark.parametrize('failure',['replay','required_gate','holdout'])
def test_missing_or_failed_evidence_adds_blocking_reason(evidence,failure):
    _,_,config,source,report = evidence
    if failure=='replay': report['variants'][service.VARIANTS[0]]['max_replay_absolute_difference']=.1
    elif failure=='required_gate': source['negative_controls']['variants']['label_window_shift']['passed']=False
    else: report['holdout_touched']=True
    state=service.review_states(source,report,config)
    assert state['research_status']=='blocked'
    assert len(state['blocking_reasons'])>1


def test_review_publication_idempotent_separate_and_no_alpha_store(evidence):
    lake,path,_,_,_=evidence
    old=lake.metadata/'research_pipeline_latest_attempt.json';old.write_text('legacy-v1')
    manifest=service.publish_evaluation_review(lake,config_path=path)
    sha=file_sha256(manifest)
    assert service.publish_evaluation_review(lake,config_path=path)==manifest
    assert file_sha256(manifest)==sha
    assert old.read_text()=='legacy-v1'
    assert not list(lake.root.rglob('_CURRENT.json'))
    assert not (lake.metadata/'factor_registry.sqlite').exists()
    response=evaluation_summary(lake,config_path=path)
    assert response['status']=='blocked' and response['latest_run'] is None
    assert response['review']['implementation_status']=='passed'
    assert 'permutation_mean_ics' not in json.dumps(response)


@pytest.mark.parametrize('tamper',['checksum','status_with_updated_checksum','missing_pointer','config'])
def test_v2_api_rejects_tampering_without_fallback(evidence,tamper):
    lake,path,_,_,_=evidence
    mp=service.publish_evaluation_review(lake,config_path=path)
    m=json.loads(mp.read_text());sp=lake.root/m['outputs']['summary']['path']
    if tamper=='checksum': sp.write_text('{}')
    elif tamper=='missing_pointer': (lake.metadata/service.POINTER).unlink()
    elif tamper=='config':
        c=json.loads(path.read_text());c['scope']='changed';path.write_text(json.dumps(c))
    else:
        payload=json.loads(sp.read_text());payload['research_status']='passed';sp.write_text(json.dumps(payload))
        m['outputs']['summary']=lake.artifact_record(sp);mp.write_text(json.dumps(m))
        publication=json.loads((lake.metadata/service.POINTER).read_text())
        publication['manifest_sha256']=file_sha256(mp);publication['summary']=lake.artifact_record(sp)
        (lake.metadata/service.POINTER).write_text(json.dumps(publication))
    response=evaluation_summary(lake,config_path=path)
    assert response['status']=='blocked' and response['latest_run'] is None
    assert response['review'] is None


def test_unimplemented_calibration_cannot_be_enabled_by_config_string(tmp_path):
    config=json.loads(service.DEFAULT_CONFIG.read_text())
    config['temporal_null_validation']['status']='validated'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='TEMPORAL_VALIDATOR_NOT_IMPLEMENTED'):
        service.load_evidence(DataLake(tmp_path/'data'),path)


def test_unchanged_research_definitions_and_single_required_gate_list():
    v2=json.loads(service.DEFAULT_CONFIG.read_text())
    v1=json.loads((service.DEFAULT_CONFIG.parent/'evaluation_framework_v1.json').read_text())
    for key in ['sample','signal','inference']:
        assert v2[key]==v1[key]
    for key in ['ic_neutralization_sides','neutralization_weight_metric','risk_exposure_column_set_id',
        'risk_set_version','singular_value_rcond','signal_lookback_trading_days','declared_direction','ic_statistic','time_shuffle_repetitions']:
        assert v2['evaluation_channel'][key]==v1['evaluation_channel'][key]
    assert v2['negative_controls']['seed']==v1['negative_controls']['seed']
    assert set(v2['negative_controls']['required'])=={k for k,v in v2['evaluation_channel']['variant_dispositions'].items() if v=='gate'}


def test_default_api_never_falls_back_to_ready_v1_when_v2_pointer_is_invalid(tmp_path):
    lake=DataLake(tmp_path)
    config=service.DEFAULT_CONFIG.parent/'evaluation_framework_v1.json'
    v1=json.loads(config.read_text())
    (lake.metadata/'evaluation-framework-summary.json').write_text(json.dumps({
        'framework_id':v1['framework_id'],'config_sha256':file_sha256(config),'status':'completed',
        'negative_controls':{'passed':True},'scope':{},'ic':{},'data_quality':{}}))
    assert evaluation_summary(lake)['status']=='ready'
    (lake.metadata/service.POINTER).write_text('{}')
    response=evaluation_summary(lake)
    assert response['framework_id']=='estimation_evaluation_v2'
    assert response['status']=='blocked' and response['latest_run'] is None
