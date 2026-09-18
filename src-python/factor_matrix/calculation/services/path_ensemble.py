"""Versioned synthetic market ensemble; reruns the audited execution engine."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from .backtest_preview_v3 import CONTRACT as BASE, STRATEGIES, build_market, run_strategy
from .equity_execution import next_weekdays
from .strategy_fixture import plain

CONTRACT = {'id':'synthetic_path_ensemble_v1','seed':1729,'paths':1000,'sessions':252,
    'generator':'backtest_chart_preview_v3_steady','base_contract':BASE,
    'stream':'sha256(ensemble_v1:1729:index) interpreted as big-endian unsigned integer',
    'quantiles':[.05,.25,.5,.75,.95],'quantile_method':'linear',
    'percentile':'100 * (count_less + 0.5 * count_equal) / paths',
    'sampling':'independent_market_paths_common_to_all_strategies;full_execution_rerun',
    'meaning':'synthetic_model_relative_not_real_world_probability',
    'failure_policy':'abort_entire_publication_no_dropped_paths',
    'research_status':'not_eligible','real_market_data':False}
METRICS=['max_drawdown','total_return','sharpe']


def stream_seed(index):
    if index < 0: raise ValueError('NEGATIVE_STREAM_INDEX')
    return int.from_bytes(sha256(f'ensemble_v1:1729:{index}'.encode()).digest(),'big')


def run_sample(index):
    seed=stream_seed(index)
    bars=build_market('steady',seed=seed)
    days=next_weekdays('2026-01-01T00:00:00+00:00',254)
    paths=[run_strategy(*s,bars,days,include_histogram=False) for s in STRATEGIES]
    return {'index':index,'seed':str(seed),'market_sha256':json_hash(plain(bars)),
        'wealth':[[r['wealth'] for r in p['rows']] for p in paths],
        'summary':[p['summary'] for p in paths],'invariants':[p['invariants'] for p in paths]}


def percentile(values, value):
    a=np.asarray(values,dtype=float)
    return float(100*((a<value).sum()+.5*(a==value).sum())/len(a))


def histogram(values, bins=32):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():raise ValueError('NONFINITE_ENSEMBLE_METRIC')
    counts,edges=np.histogram(a,bins=bins)
    # Density as a closed step polygon, including empty outer edges.
    points=[{'x':float(edges[0]),'y':0.}]
    for i,c in enumerate(counts):
        h=float(c/len(a)/(edges[i+1]-edges[i]))
        points.extend([{'x':float(edges[i]),'y':h},{'x':float(edges[i+1]),'y':h}])
    points.append({'x':float(edges[-1]),'y':0.})
    return {'points':points,'counts':counts.tolist(),'edges':edges.tolist(),'n':len(a)}


def summarize(samples, reference):
    samples=sorted(samples,key=lambda s:s['index'])
    if [s['index'] for s in samples]!=list(range(len(samples))):raise ValueError('INCOMPLETE_STREAM_AXIS')
    series=[]
    for j,(sid,name,color) in enumerate(STRATEGIES):
        wealth=np.array([s['wealth'][j] for s in samples])
        if wealth.shape!=(len(samples),253) or not np.isfinite(wealth).all() or not (wealth[:,0]==1).all():
            raise ValueError('INVALID_WEALTH_PATH')
        bands=np.quantile(wealth,CONTRACT['quantiles'],axis=0,method='linear')
        distributions={}
        for metric in METRICS:
            values=[s['summary'][j][metric] for s in samples]
            if any(v is None for v in values):raise ValueError('UNDEFINED_ENSEMBLE_METRIC')
            h=histogram(values)
            markers=[]
            for scenario in reference['scenarios']:
                v=next(p for p in scenario['series'] if p['id']==sid)['summary'][metric]
                markers.append({'scenario':scenario['id'],'name':scenario['name'],'value':v,'percentile':percentile(values,v)})
            distributions[metric]={**h,'markers':markers}
        series.append({'id':sid,'name':name,'color':color,
            'fan':[{'day':i,**{f'q{int(q*100):02}':float(bands[k,i]) for k,q in enumerate(CONTRACT['quantiles'])}} for i in range(253)],
            'distributions':distributions})
    return {'schema_version':1,'contract':{**CONTRACT,'paths':len(samples)},'series':series,
        'invariants':{'status':'passed','market_paths':len(samples),'strategy_paths':len(samples)*4,'audited_sessions':len(samples)*4*252},
        'real_market_data':False,'research_status':'not_eligible'}


def publish_ensemble(output_dir:Path, *, reference_path:Path, workers=4):
    # No configurable seed/window/path count on the release CLI.
    if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('OUTPUT_MUST_BE_EMPTY')
    reference=json.loads(reference_path.read_text())
    if reference['schema_version']!=3 or reference['contract']!=BASE:raise ValueError('REFERENCE_CONTRACT_MISMATCH')
    lake=DataLake(output_dir)
    code_hash=source_tree_hash()
    source_files=[Path(__file__).with_name(n) for n in ['path_ensemble.py','backtest_preview_v3.py','preview_statistics.py','equity_execution.py','accounting.py','execution_invariants.py','backtest_path.py','equity_fixture.py']]
    source_hashes={str(p):file_sha256(p) for p in source_files}
    samples=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sample in pool.map(run_sample,range(CONTRACT['paths']),chunksize=1):
            if any(c['status']!='passed' for c in sample['invariants']):raise ValueError('PATH_AUDIT_FAILED')
            samples.append(sample)
            if len(samples)%25==0:print(f"ensemble {len(samples)}/{CONTRACT['paths']}",flush=True)
    if any(file_sha256(Path(p))!=h for p,h in source_hashes.items()):raise ValueError('EXECUTION_SOURCE_CHANGED_DURING_RUN')
    output=lake.write_immutable_json(output_dir/'ensemble.json',summarize(samples,reference))
    sample_file=lake.write_immutable_json(output_dir/'samples.json',{'paths':samples})
    return lake.write_immutable_json(output_dir/'_MANIFEST.json',{'schema_version':1,
        'run_id':CONTRACT['id']+'_'+json_hash(CONTRACT)[:16],'definition':CONTRACT,'code_hash_at_start':code_hash,
        'execution_source_hashes':source_hashes,'parent_run_ids':[BASE['id']],
        'reference':{'path':str(reference_path),'sha256':file_sha256(reference_path)},
        'risk_basis_id':None,'risk_set_version':None,'status':'synthetic_engineering_passed',
        'outputs':{'ensemble':lake.artifact_record(output),'samples':lake.artifact_record(sample_file)}})
