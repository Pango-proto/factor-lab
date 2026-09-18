"""Versioned review of existing evidence; temporal inference stays fail-closed."""
from __future__ import annotations

import json
from pathlib import Path

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from .permutation_diagnostics import verify_fixed_inputs, VARIANTS
from .research_pipeline import _atomic_json

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT / 'config/evaluation_framework_v2.json'
POINTER = 'research_pipeline_v2_latest_review.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_json(path, sha):
    if file_sha256(path) != sha:
        raise ValueError(f'REVIEW_V2_CHECKSUM_MISMATCH {path}')
    return read_json(path)


def load_evidence(lake, config_path):
    config = read_json(config_path)
    if config.get('framework_id') != 'estimation_evaluation_v2' or config.get('status') != 'active':
        raise ValueError('REVIEW_V2_CONTRACT_UNSUPPORTED')
    if config.get('publication_policy', {}).get('mode') != 'review_only':
        raise ValueError('REVIEW_V2_PUBLICATION_NOT_IMPLEMENTED')
    # A status string is never accepted as proof of a calibrated new method.
    temporal = config.get('temporal_null_validation', {})
    if temporal.get('status') != 'not_calibrated' or temporal.get('calibration_evidence') is not None:
        raise ValueError('REVIEW_V2_TEMPORAL_VALIDATOR_NOT_IMPLEMENTED')
    parent = config['parent_contract']
    checked_json(config_path.parent / parent['path'], parent['sha256'])
    binding = config['diagnostic_binding']
    diagnostic_config = checked_json(config_path.parent / binding['config'], binding['config_sha256'])
    _, source, _ = verify_fixed_inputs(lake, diagnostic_config)
    manifest = checked_json(lake.root / binding['manifest'], binding['manifest_sha256'])
    if (manifest.get('parent_run_ids') != [diagnostic_config['source_run_id']]
        or manifest.get('definition', {}).get('config_sha256') != binding['config_sha256']
        or manifest.get('holdout_touched') is not False):
        raise ValueError('REVIEW_V2_DIAGNOSTIC_LINEAGE_MISMATCH')
    for record in manifest['outputs'].values():
        if file_sha256(lake.root / record['path']) != record['sha256']:
            raise ValueError('REVIEW_V2_DIAGNOSTIC_OUTPUT_MISMATCH')
    report = read_json(lake.root / manifest['outputs']['report']['path'])
    if report['source_run_id'] != diagnostic_config['source_run_id'] or report['run_id'] != manifest['run_id']:
        raise ValueError('REVIEW_V2_REPORT_LINEAGE_MISMATCH')
    return config, source, report, manifest


def review_states(source, report, config):
    tolerance = config['implementation_audit']['absolute_tolerance']
    numerical = report['independent_neutralization']
    checked = report.get('status') == 'completed' and report.get('holdout_touched') is False
    checked &= numerical['max_residual_difference'] <= tolerance and numerical['max_daily_ic_difference'] <= tolerance
    for name in VARIANTS:
        result = report['variants'][name]
        checked &= result.get('replay_matches_saved') is True and result['max_replay_absolute_difference'] <= tolerance
        checked &= result['decomposition_max_absolute_error'] <= tolerance and result['lag_decomposition_max_absolute_error'] <= tolerance
    variants = source['negative_controls']['variants']
    required = config['negative_controls']['required']
    # Do not allow a shortened config list to silently remove the remaining gates.
    if set(required) != {'within_date_cross_sectional_return_shuffle', 'label_window_shift'}:
        raise ValueError('REVIEW_V2_REQUIRED_GATES_CHANGED')
    gates = {name: variants[name].get('passed') is True for name in required}
    structured = {name: {'disposition':'structured_diagnostic',
        'legacy_zero_interval_passed':variants[name]['passed'],
        'legacy_distribution':variants[name], 'diagnostic':report['variants'][name]} for name in VARIANTS}
    reasons = ['TEMPORAL_NULL_NOT_CALIBRATED']
    if not checked:
        reasons.append('IMPLEMENTATION_AUDIT_FAILED')
    reasons.extend('REQUIRED_GATE_FAILED:'+name for name, passed in gates.items() if not passed)
    return {'execution_status':'completed','implementation_status':'passed' if checked else 'failed',
        'temporal_validation_status':'not_calibrated','research_status':'blocked',
        'required_gates':gates,'structured_diagnostics':structured,'blocking_reasons':reasons,
        'holdout_touched':False,'alpha_registration_permitted':False,'research_current_permitted':False}


def publish_evaluation_review(lake: DataLake, *, config_path=DEFAULT_CONFIG):
    config, source, report, diagnostic_manifest = load_evidence(lake, config_path)
    definition = {'framework_id':config['framework_id'],'config_sha256':file_sha256(config_path),
        'source_run_id':report['source_run_id'],'diagnostic_manifest_sha256':config['diagnostic_binding']['manifest_sha256'],
        'code_hash':source_tree_hash()}
    run_id = 'evaluation_review_v2_' + json_hash(definition)[:16]
    directory = lake.root/'gold/evaluation_review_v2'/f'run_id={run_id}'
    payload = {'schema_version':2,'run_id':run_id,'framework_id':config['framework_id'],
        'config_sha256':definition['config_sha256'],'parent_run_ids':[report['source_run_id'],report['run_id']],
        'source_status':source['status'], **review_states(source,report,config)}
    # Recheck all bound evidence and implementation before publication.
    load_evidence(lake,config_path)
    if file_sha256(config_path)!=definition['config_sha256'] or source_tree_hash()!=definition['code_hash']:
        raise ValueError('REVIEW_V2_INPUT_CHANGED_DURING_RUN')
    summary = lake.write_immutable_json(directory/'summary.json',payload)
    manifest = lake.write_immutable_json(directory/'_MANIFEST.json',{
        'schema_version':2,'run_id':run_id,'status':'blocked','definition':definition,
        'parent_run_ids':payload['parent_run_ids'],'outputs':{'summary':lake.artifact_record(summary)},
        'diagnostic_outputs':diagnostic_manifest['outputs'],'holdout_touched':False})
    publication = {'run_id':run_id,'manifest':str(manifest.relative_to(lake.root)),
        'manifest_sha256':file_sha256(manifest),'summary':lake.artifact_record(summary)}
    # This is a review pointer, never a research CURRENT or a research attempt.
    _atomic_json(lake.metadata/POINTER,publication)
    return manifest


def evaluation_review_summary(lake, *, config_path=DEFAULT_CONFIG):
    config = read_json(config_path)
    reason = '工程审计与结构诊断已记录；时序零假设尚未校准，研究继续阻断。'
    states = None
    try:
        config, source, report, _ = load_evidence(lake,config_path)
        expected = review_states(source,report,config)
        publication = read_json(lake.metadata/POINTER)
        manifest = checked_json(lake.root/publication['manifest'],publication['manifest_sha256'])
        if (manifest['definition']['config_sha256'] != file_sha256(config_path)
            or manifest['run_id'] != publication['run_id'] or manifest['outputs']['summary'] != publication['summary']
            or manifest['definition']['diagnostic_manifest_sha256'] != config['diagnostic_binding']['manifest_sha256']):
            raise ValueError('REVIEW_V2_PUBLICATION_LINEAGE_MISMATCH')
        record = publication['summary']
        payload = checked_json(lake.root/record['path'],record['sha256'])
        if (payload['run_id'] != publication['run_id'] or payload['framework_id'] != config['framework_id']
            or payload['config_sha256'] != file_sha256(config_path)
            or any(payload.get(k) != value for k,value in expected.items())):
            raise ValueError('REVIEW_V2_STATE_MISMATCH')
        states = {key:payload[key] for key in ('execution_status','implementation_status',
            'temporal_validation_status','research_status','required_gates','blocking_reasons')}
        states['structured_diagnostics'] = {name:{'disposition':'structured_diagnostic',
            'legacy_zero_interval_passed':value['legacy_zero_interval_passed']}
            for name,value in payload['structured_diagnostics'].items()}
        if states['implementation_status'] != 'passed' or not all(states['required_gates'].values()):
            reason = '工程审计或必需负对照未通过；时序零假设尚未校准，研究继续阻断。'
    except (OSError, ValueError, KeyError, TypeError):
        reason = 'v2 审计证据、配置或来源校验失败，研究结果已阻断。'
    return {'schema_version':2,'framework_id':config['framework_id'],'config_sha256':file_sha256(config_path),
        'status':'blocked','latest_run':None,'review':states,'contract':{k:config.get(k,{}) for k in (
            'status','component_declaration','sample','signal','negative_controls','evaluation_channel',
            'construction_gates','inference','output_contract','holdout_capacity')},
        'excluded_run':{'status':'blocked','reason':reason,'source':'metadata/'+POINTER,
            'superseded_by':None,'supersession_reason':None},
        'boundary':{'daily_ic_sent':False,'security_returns_sent':False,'factor_matrix_sent':False,'holdout_opened':False}}
