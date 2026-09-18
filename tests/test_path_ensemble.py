import numpy as np
import pytest
from factor_matrix.calculation.services.path_ensemble import run_sample, stream_seed, summarize, percentile, histogram
from factor_matrix.calculation.services.preview_statistics import describe_returns
from factor_matrix.calculation.services.backtest_preview_v3 import build_preview_data

@pytest.fixture(scope='module')
def samples():return [run_sample(i) for i in range(2)]

def test_streams_and_full_accounting(samples):
    assert stream_seed(0)==stream_seed(0)!=stream_seed(1)
    assert samples[0]==run_sample(0)
    assert samples[0]['market_sha256']!=samples[1]['market_sha256']
    for s in samples:
        assert len(s['wealth'])==4
        assert all(p[0]==1 and len(p)==253 for p in s['wealth'])
        assert all(c['status']=='passed' for c in s['invariants'])

def test_summary_order_t0_quantiles_and_complete_counts(samples):
    ref=build_preview_data()
    result=summarize(samples,ref)
    assert result==summarize(samples[::-1],ref)
    assert result['invariants']['audited_sessions']==2016
    for s in result['series']:
        assert all(v==1 for k,v in s['fan'][0].items() if k!='day')
        for r in s['fan']:
            assert r['q05']<=r['q25']<=r['q50']<=r['q75']<=r['q95']
        for d in s['distributions'].values():assert sum(d['counts'])==2 and len(d['markers'])==3
    with pytest.raises(ValueError,match='STREAM_AXIS'):summarize([samples[1]],ref)

def test_extreme_returns_not_silently_dropped():
    x=[.001,-.002]*12+[.089]
    assert describe_returns(x,include_histogram=False)['n']==25
    with pytest.raises(ValueError,match='SUPPORT'):describe_returns(x)
    assert sum(histogram([-1,0,1,100])['counts'])==4
    assert percentile([0,1,1,2],1)==50
    assert percentile([0,1],-1)==0
    assert percentile([0,1],2)==100
