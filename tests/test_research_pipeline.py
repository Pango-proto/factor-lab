from dataclasses import replace
from datetime import date, timedelta
import json
import sqlite3

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factor_matrix.calculation.components.probe_signal import MaterializeProbeSignal
from factor_matrix.calculation.components.probe_evaluation import alignment_curve, permutation_means, EvaluateProbe
from factor_matrix.calculation.components.market import MaterializeRealizedReturns
from factor_matrix.calculation.core.contracts import (ArtifactKey, ArtifactRef, LoadedArtifact,
    CalculationRequest, QualityStatus)
from factor_matrix.calculation.core.policy import architecture_boundary_policy
from factor_matrix.calculation.l2.estimation_evaluation import EvaluationContract
from factor_matrix.calculation.services.research_pipeline import resolve_sample, register_probe, PROJECT
from factor_matrix.calculation.services.evaluation_summary import evaluation_summary
from factor_matrix.storage import DataLake, file_sha256
from tests.test_calculation_chain import _build, _materialize_request


def loaded(kind, frame):
    ref = ArtifactRef(key=ArtifactKey(artifact_type=kind,version="1",run_id=kind,
        as_of_date=date(2020,1,1),scope_key="research:test"),uri=kind,checksum="fixture",
        quality_status=QualityStatus.PASSED,tags=frozenset())
    return LoadedArtifact(reference=ref,tables={kind+"_v1":frame})


def test_research_sample_purge_uses_sessions_and_rejects_holdout():
    c = EvaluationContract.from_json(PROJECT/'config/evaluation_framework_v1.json')
    dates = [date(2024,10,1)+timedelta(days=i) for i in range(151)]
    dates = [d for d in dates if d.weekday()<5]
    end, label_end, safe = resolve_sample(dates,c)
    assert end == safe == dates[-26]
    assert label_end == dates[-6]
    for invalid in (dates[-25], c.holdout_start):
        with pytest.raises(ValueError,match="SAMPLE_BOUNDARY"):
            resolve_sample(dates,c,invalid)


def test_probe_is_causal_and_respects_registered_formation():
    plugin = MaterializeProbeSignal(None)
    architecture_boundary_policy().validate_spec(plugin.spec)
    assert 'forward_labels' not in plugin.spec.input_artifact_types
    dates = [date(2020,1,1)+timedelta(days=i) for i in range(40)]
    returns = pl.DataFrame({'trade_date':dates,'asset_id':['a']*40,'realized_return':[0.01]*40})
    universe = loaded('tradable_universe',returns.select('trade_date','asset_id'))
    request = CalculationRequest(operation_id=plugin.spec.operation_id,operation_version='1',
        as_of_date=dates[25],scope_key='research:test',parameters={'end_date':str(dates[25])},inputs=())
    def calc(frame):
        return plugin.calculate(request,(loaded('realized_returns',frame),universe)).outputs[0].tables['probe_signal_panel_v1']
    baseline = calc(returns)
    changed = returns.with_columns(pl.when(pl.col('trade_date')>dates[25]).then(3.0).otherwise(pl.col('realized_return')).alias('realized_return'))
    assert_frame_equal(baseline,calc(changed))
    assert baseline['signal_value'][13] is None
    assert baseline['signal_value'][14] == pytest.approx(1-1.01**15)
    assert baseline['signal_value'][25] == pytest.approx(1-1.01**20)


def test_calculation_publication_can_be_repeated_without_timestamp_conflict(tmp_path):
    lake,executor,pointer,universe = _build(tmp_path)
    request = _materialize_request((universe,))
    first = executor.execute(request)
    second = executor.execute(request)
    assert first == second


def test_probe_registration_is_idempotent_and_creates_no_alpha_attempt(tmp_path):
    lake = DataLake(tmp_path)
    key = register_probe(lake)
    assert register_probe(lake) == key
    with sqlite3.connect(lake.metadata/'factor_registry.sqlite') as db:
        assert db.execute('select count(*) from probe_registry').fetchone()[0] == 1
        assert db.execute('select count(*) from research_attempt').fetchone()[0] == 0
        assert db.execute('select count(*) from alpha_assertion').fetchone()[0] == 0
        assert db.execute("select assigned_role from feature_assignment where feature_id='reversal_20d_v1'").fetchone()[0] == 'probe'


def test_permutation_reproducibility_and_immutable_inputs():
    rng = np.random.default_rng(42)
    signal,labels = rng.normal(size=(40,8)),rng.normal(size=(40,8))
    labels[::3,2] = np.nan
    before = labels.copy()
    for strategy in ('within_asset_time_shuffle_raw','within_asset_time_shuffle_demeaned',
                     'within_date_cross_sectional_return_shuffle','date_axis_shuffle'):
        a = permutation_means(signal,labels,strategy=strategy,repetitions=20,seed=8)
        b = permutation_means(signal,labels,strategy=strategy,repetitions=20,seed=8)
        assert a == b and len(a)==20 and np.isfinite(a).all()
    np.testing.assert_array_equal(labels,before)


def test_alignment_uses_common_support():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(40,8))
    labels = signal.copy()
    labels[10,2] = np.nan
    result = alignment_curve(signal,labels,(-1,0,1),side='label')
    assert result['peak_offsets_trading_days'] == [0]
    assert result['observations'] == 38
    assert result['support'] == 'common_date_asset_intersection_across_offsets'
    with pytest.raises(ValueError,match='INSUFFICIENT_SUPPORT'):
        alignment_curve(signal[:2],labels[:2],(-2,0,2),side='label')


def test_materializer_reads_actual_silver_ledger_pointer(tmp_path):
    lake = DataLake(tmp_path)
    ledger = lake.metadata/'silver_versions';ledger.mkdir()
    (ledger/'_CURRENT').write_text('v.json\n')
    (ledger/'v.json').write_text('{"version_id":"v"}')
    assert MaterializeRealizedReturns(lake)._silver_version() == 'v'


def test_evaluation_source_checksum_failure_does_not_fallback(tmp_path):
    lake = DataLake(tmp_path)
    (lake.metadata/'research_pipeline_latest_attempt.json').write_text('{}')
    result = evaluation_summary(lake)
    assert result['status'] == 'blocked' and result['latest_run'] is None
    assert '校验失败' in result['excluded_run']['reason']


def test_evaluation_operation_obeys_artifact_boundaries():
    plugin = EvaluateProbe(PROJECT/'config/evaluation_framework_v1.json')
    architecture_boundary_policy().validate_spec(plugin.spec)
    assert set(plugin.spec.input_artifact_types) == {'risk_exposure_matrix','forward_labels','tradable_universe','probe_signal_panel'}


def _pipeline_fixture(tmp_path):
    from factor_matrix.calculation.services.research_pipeline import DEFAULT_CONFIG
    lake = DataLake(tmp_path/'data')
    dates = [date(2019,7,1)+timedelta(days=i) for i in range(230)]
    rng = np.random.default_rng(129)
    rows = []
    for day in dates:
        for a in range(12):
            rows.append({'trade_date':day,'asset_id':str(a),'board_id':'MAIN',
                'observation_state':'TRADED','is_suspended':False,'is_exchange_first_day':False,
                'is_tradable':day>=date(2019,10,30),'total_return':float(rng.normal(0,0.02)),'return_source':'price',
                'missing_return_policy':'observed','risk_country':1.0,'risk_size':float(a),
                'risk_industry_1':float(a<6),'risk_industry_2':float(a>=6),'is_valid':True})
    frame = pl.DataFrame(rows)
    lake.replace('returns_daily',frame.select('trade_date','asset_id','total_return','return_source','missing_return_policy','is_suspended'))
    ud = lake.root/'gold/tradable_universe/artifact_version=1/run_id=universe_fixture';ud.mkdir(parents=True)
    up = ud/'tradable_universe.parquet'
    frame.select('trade_date','asset_id','board_id','observation_state','is_suspended','is_exchange_first_day','is_tradable').write_parquet(up)
    (ud/'metadata.json').write_text(json.dumps({'run_id':'universe_fixture','silver_version_id':'silver_fixture',
        'artifact':{'sha256':file_sha256(up)}}))
    xd = lake.root/'gold/risk_exposure_matrix/run_id=l1_fixture';xd.mkdir(parents=True)
    xp = xd/'risk_exposure_matrix_v1.parquet'
    frame.select('trade_date','asset_id','is_valid','risk_country','risk_size','risk_industry_1','risk_industry_2').write_parquet(xp)
    mp = xd/'_MANIFEST.json'
    mp.write_text(json.dumps({'run_id':'l1_fixture','risk_basis_id':'basis_fixture','silver_version_id':'silver_fixture',
        'tradable_universe_run_id':'universe_fixture','start':str(dates[0]),'outputs':{'risk_exposure_matrix_v1':{
            'path':str(xp.relative_to(lake.root)),'sha256':file_sha256(xp)}}}))
    dp = xd/'diagnosis.json'
    dp.write_text(json.dumps({'status':'passed','promotion_gate':'passed','risk_basis_id':'basis_fixture',
        'history_run_id':'l1_fixture','history_manifest_sha256':file_sha256(mp)}))
    (xd.parent/'_CURRENT.json').write_text(json.dumps({'run_id':'l1_fixture','risk_basis_id':'basis_fixture',
        'manifest':str(mp.relative_to(lake.root)),'diagnostics_manifest':str(dp.relative_to(lake.root))}))
    return lake, DEFAULT_CONFIG


def test_pipeline_runs_all_stages_blocks_failed_gates_and_detects_tampering(tmp_path):
    from factor_matrix.calculation.services.research_pipeline import run_research_pipeline
    lake,config = _pipeline_fixture(tmp_path)
    manifest_path = run_research_pipeline(lake,config_path=config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['execution_status'] == 'completed'
    assert len(manifest['artifacts']) == 6
    evaluation_meta = json.loads((lake.root/manifest['artifacts'][-1]['uri']).read_text())
    table = lake.root/next(iter(evaluation_meta['tables'].values()))['uri']
    assert pl.read_parquet(table)['trade_date'].min() == date(2019,10,8)
    assert evaluation_meta['metadata']['legacy_d120_filter_reapplied'] is False
    assert manifest['status'] == 'blocked'
    assert not (lake.root/'gold/research_pipeline/_CURRENT.json').exists()
    assert evaluation_summary(lake)['latest_run'] is None
    assert run_research_pipeline(lake,config_path=config) == manifest_path
    artifact = lake.root/manifest['artifacts'][-1]['uri']
    artifact.write_text(artifact.read_text()+' ')
    with pytest.raises(ValueError,match='ARTIFACT_CHECKSUM'):
        run_research_pipeline(lake,config_path=config)


def test_pipeline_rejects_unbound_l1_diagnostics_before_registration(tmp_path):
    from factor_matrix.calculation.services.research_pipeline import run_research_pipeline
    lake,config = _pipeline_fixture(tmp_path)
    diagnosis = lake.root/'gold/risk_exposure_matrix/run_id=l1_fixture/diagnosis.json'
    payload = json.loads(diagnosis.read_text());payload['history_run_id']='different_run'
    diagnosis.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='LINEAGE_OR_GATE'):
        run_research_pipeline(lake,config_path=config)
    assert not (lake.metadata/'factor_registry.sqlite').exists()


def test_pipeline_promotes_only_when_research_acceptance_passes(tmp_path,monkeypatch):
    from factor_matrix.calculation.services.research_pipeline import run_research_pipeline
    original = EvaluateProbe.calculate
    def accepted(self,request,inputs):
        result = original(self,request,inputs)
        output = result.outputs[0]
        metadata = dict(output.metadata)
        metadata['negative_controls'] = {**metadata['negative_controls'],'passed':True,
            'variants':{k:{**v,'passed':True} if v.get('disposition')=='gate' else v
                        for k,v in metadata['negative_controls']['variants'].items()}}
        return replace(result,outputs=(replace(output,metadata=metadata),))
    # Isolate publication behavior from the stochastic outcome of fixture data.
    monkeypatch.setattr(EvaluateProbe,'calculate',accepted)
    lake,config = _pipeline_fixture(tmp_path)
    path = run_research_pipeline(lake,config_path=config)
    publication = json.loads((lake.root/'gold/research_pipeline/_CURRENT.json').read_text())
    assert publication['manifest_sha256'] == file_sha256(path)
    assert publication['status'] == 'passed'
    response = evaluation_summary(lake)
    assert response['status'] == 'ready' and response['latest_run']['scope']['deployable'] is False
    summary = lake.root/publication['summary']['path']
    summary.write_text(summary.read_text()+' ')
    assert evaluation_summary(lake)['status'] == 'blocked'
