import json
from pathlib import Path

from factor_matrix.storage import DataLake,open_duckdb
import factor_matrix.data_refresh as refresh


def test_database_timestamp_casts_use_explicit_utc():
    with open_duckdb() as c:
        assert c.execute("select current_setting('TimeZone')").fetchone()[0]=='UTC'
        assert c.execute("select cast(TIMESTAMPTZ '2026-08-06T23:00:00+00:00' as date)").fetchone()[0].isoformat()=='2026-08-06'


def test_failed_refresh_keeps_gate_closed_and_resume_reuses_successes(tmp_path,monkeypatch):
    lake=DataLake(tmp_path/'data')
    market=tmp_path/'market.json'
    market.write_text(json.dumps(dict(job='market_fact_catchup',status='passed',run_id='market',days=[],identity={'end':'2026-09-16'})))
    calls={name:0 for name in ['financial','benchmarks','industry','concepts']}
    fail={'financial':True}
    def pipeline(name):
        class Pipeline:
            def __init__(self,*args):pass
            def sync(self,*args,**kwargs):
                calls[name]+=1
                if fail.get(name):raise RuntimeError('source_failed')
                return lake.write_immutable_json(lake.root/f'{name}.json',{'status':'passed'})
        return Pipeline
    for name,attribute in [('financial','FactorDataPipeline'),('benchmarks','BenchmarkMembershipPipeline'),
                           ('industry','IndustryClassificationPipeline'),('concepts','ConceptSnapshotPipeline')]:
        monkeypatch.setattr(refresh,attribute,pipeline(name))
    for name in ['run_daily_quality','run_history_quality','run_factor_data_quality','industry_quality']:
        monkeypatch.setattr(refresh,name,lambda *a,**kw:{'status':'passed'})
    class Ledger:
        def __init__(self,lake):pass
        def current(self):return {'version_id':'test'}
    monkeypatch.setattr(refresh,'SilverVersionLedger',Ledger)
    first=refresh.finish_data_refresh(None,lake,market_manifest=market,credential_fingerprint='test',progress=lambda s:None)
    assert json.loads(first.read_text())['market_input_allowed'] is False
    fail['financial']=False
    second=refresh.finish_data_refresh(None,lake,market_manifest=market,resume_manifest=first,
                                       credential_fingerprint='test',progress=lambda s:None)
    assert json.loads(second.read_text())['market_input_allowed'] is True
    assert calls=={'financial':2,'benchmarks':1,'industry':1,'concepts':1}
