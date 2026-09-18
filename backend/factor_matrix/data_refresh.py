"""Complete an explicitly bound market catchup with slower reference datasets."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .benchmark_membership import BenchmarkMembershipPipeline
from .classification import IndustryClassificationPipeline, industry_quality
from .concepts import ConceptSnapshotPipeline
from .factor_data import FactorDataPipeline
from .factor_quality import run_factor_data_quality
from .quality import run_daily_quality, run_history_quality
from .revisioned_silver import SilverVersionLedger
from .storage import DataLake, json_hash, source_tree_hash, utc_now


def finish_data_refresh(client, lake: DataLake, *, market_manifest: Path, credential_fingerprint: str, progress=print,
                        resume_manifest: Path | None = None):
    market = json.loads(market_manifest.read_text())
    if market.get('job') != 'market_fact_catchup' or market.get('status') != 'passed':
        raise ValueError('DATA_REFRESH_REQUIRES_PASSED_MARKET_CATCHUP')
    for day in market['days']:
        for ref in day['components'].values():
            from .storage import file_sha256
            if file_sha256(lake.root / ref['path']) != ref['sha256']:
                raise ValueError('DATA_REFRESH_MARKET_COMPONENT_CHANGED')
    end = date.fromisoformat(market['identity']['end'])
    observed_day = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    started_at = utc_now()
    reusable={}
    if resume_manifest is not None:
        prior=json.loads(resume_manifest.read_text())
        if (prior.get('job')!='complete_data_refresh' or prior['definition']['market_parent']['sha256']!=lake.artifact_record(market_manifest)['sha256']
            or prior['definition']['reference_observation_date']!=str(observed_day)):
            raise ValueError('DATA_REFRESH_RESUME_PARENT_MISMATCH')
        failed={r['component'] for r in prior['errors']}
        reusable={k:v for k,v in prior['outputs'].items() if k not in failed}
    definition = {'market_parent': lake.artifact_record(market_manifest), 'end': str(end),
                  'reference_observation_date': str(observed_day), 'started_at': started_at.isoformat(),
                  'code_hash': source_tree_hash()}
    run_id = 'data_refresh_v1_' + json_hash(definition)[:16]
    directory = lake.root / 'diagnostics' / 'data_refresh_v1' / f'run_id={run_id}'
    outputs, errors = {}, []
    # Reference sources are observed NOW; never claim today's concept snapshot
    # was historically observed on each missing trading day.
    jobs = {
        'financial': lambda: FactorDataPipeline(client, lake, credential_fingerprint).sync(
            range(end.year-1,end.year+1), date(end.year,8,1), end),
        'benchmarks': lambda: BenchmarkMembershipPipeline(client, lake, credential_fingerprint).sync(date(2025,7,1),end),
        'industry': lambda: IndustryClassificationPipeline(client,lake,credential_fingerprint).sync('SW2021'),
        'concepts': lambda: ConceptSnapshotPipeline(client,lake,credential_fingerprint).sync(observed_day,progress=progress),
    }
    for name, job in jobs.items():
        progress(f'Data refresh: {name} started')
        try:
            if name in reusable:
                from .storage import file_sha256
                path=lake.root/reusable[name]['path']
                if file_sha256(path)!=reusable[name]['sha256']:
                    raise ValueError('DATA_REFRESH_RESUME_CHECKSUM_MISMATCH')
            else:
                path=job()
            outputs[name]=lake.artifact_record(path)
            payload=json.loads(path.read_text())
            if payload.get('status','passed') != 'passed' or payload.get('quality_gate',{}).get('status','passed') != 'passed':
                raise ValueError(f'{name.upper()}_SOURCE_QUALITY_NOT_PASSED')
            progress(f'Data refresh: {name} completed')
        except Exception as exc:
            errors.append({'component':name,'error_type':type(exc).__name__,'detail':str(exc)})
            progress(f'Data refresh: {name} failed: {type(exc).__name__}: {exc}')
    for name, job in {
        'daily_quality':lambda:run_daily_quality(lake,end),
        'history_quality':lambda:run_history_quality(lake),
        'financial_quality':lambda:run_factor_data_quality(lake,end),
        'industry_quality':lambda:industry_quality(lake,end,standard='SW2021',level='L2'),
    }.items():
        progress(f'Data refresh: {name} started')
        try:
            report=job()
            path=lake.write_immutable_json(directory/f'{name}.json',report)
            outputs[name]=lake.artifact_record(path)
            if report.get('status') != 'passed':
                raise ValueError(f'{name.upper()}_NOT_PASSED')
        except Exception as exc:
            errors.append({'component':name,'error_type':type(exc).__name__,'detail':str(exc)})
            progress(f'Data refresh: {name} failed: {type(exc).__name__}: {exc}')
    current=SilverVersionLedger(lake).current()
    return lake.write_immutable_json(directory/'_MANIFEST.json',{
        'schema_version':1,'run_id':run_id,'job':'complete_data_refresh','status':'passed' if not errors else 'failed',
        'definition':definition,'parent_run_ids':[market['run_id']],'outputs':outputs,'errors':errors,
        'silver_version_id':current['version_id'],'data_as_of':str(end),'observed_at':utc_now().isoformat(),
        'market_input_allowed':not errors,'real_total_cost_backtest_allowed':False,
        'research_promoted':False,'historical_concept_snapshots_synthesized':False,
        'gate_scope':'market input readiness; independent of alpha validity, fee verification and holdout access',
    })
