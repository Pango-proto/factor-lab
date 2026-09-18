import json
import numpy as np
import pytest

from factor_matrix.calculation.services.synthetic_calibration import (
    DEFAULT_CONFIG,SyntheticDesign,reference_signal_labels,reference_statistics,
    production_signal_labels,production_statistics,generate_returns,stream,ar_path,wilson,
)


@pytest.fixture(scope='module')
def config():
    return json.loads(DEFAULT_CONFIG.read_text())


def test_reference_products_have_hand_computed_window_boundaries():
    r=np.full((1,153,128),.01)
    s,y=reference_signal_labels(r)
    assert np.isnan(s[0,13]).all()
    np.testing.assert_allclose(s[0,14],1-1.01**15,atol=1e-14)
    np.testing.assert_allclose(s[0,20],1-1.01**20,atol=1e-14)
    np.testing.assert_allclose(y[0,20],1.01**5-1,atol=1e-14)
    r[0,20,0]=.1;r[0,21,1]=.2;r[0,23,2]=np.nan
    s,y=reference_signal_labels(r)
    assert s[0,20,0]==pytest.approx(1-1.1*1.01**19)
    assert y[0,20,0]==pytest.approx(1.01**5-1)
    assert y[0,20,1]==pytest.approx(1.2*1.01**4-1)
    assert np.isnan(y[0,20,2])
    r[0,1:7,3]=np.nan
    assert np.isnan(reference_signal_labels(r)[0][0,20,3])


def test_reference_ranks_use_ties_and_have_known_ic(config):
    # Deterministic nonconstant panel; both routes must agree without new RNG.
    design=SyntheticDesign(config,0)
    i=np.arange(128)[None,:];t=np.arange(153)[:,None]
    r=.01*np.sin((i+1)*(t+1)/137)+.003*np.cos((i+5)*(t+3)/83)
    prod,counts=production_statistics(r,design,0)
    ref,refcounts=reference_statistics(r[None],design)
    assert prod==pytest.approx(ref[0],abs=1e-12)
    np.testing.assert_array_equal(counts,refcounts[0])


@pytest.mark.parametrize('scenario',[0,3])
def test_actual_production_components_match_independent_formulas_with_gaps(config,scenario):
    design=SyntheticDesign(config,scenario)
    i=np.arange(128)[None,:];t=np.arange(153)[:,None]
    returns=.01*np.sin((t+1)*(i+3)/77)
    returns[~design.finite]=np.nan
    ps,py=production_signal_labels(returns,design)
    rs,ry=reference_signal_labels(returns[None])
    np.testing.assert_allclose(ps[design.listed],rs[0][design.listed],atol=1e-12)
    # Pre-listing labels cannot enter the IC; compare actual listed rows.
    np.testing.assert_allclose(py[design.listed],ry[0][design.listed],atol=1e-12)
    a,n=production_statistics(returns,design,0)
    b,m=reference_statistics(returns[None],design)
    assert a==pytest.approx(b[0],abs=1e-12)
    np.testing.assert_array_equal(n,m[0])


def test_rng_addresses_are_batch_independent_and_references_have_no_power_injection(config):
    design=SyntheticDesign(config,4)
    batch=generate_returns(design,4,1,0,[1,2])
    single=generate_returns(design,4,1,0,[2])
    np.testing.assert_array_equal(batch[1],single[0])
    assert not np.array_equal(batch[0],batch[1],equal_nan=True)
    for p in range(3):
        seed=np.random.SeedSequence([20260818,777,4,1,0,p,4])
        np.testing.assert_array_equal(stream(4,1,0,p,4).normal(size=8),np.random.Generator(np.random.PCG64(seed)).normal(size=8))


def test_ar_recursion_and_wilson_have_known_answers():
    np.testing.assert_allclose(ar_path(np.ones((1,3)),.5,2),[[2,3,3.5]])
    low,high=wilson(200,200)
    assert low==pytest.approx(.9811546736227335)
    assert high==pytest.approx(1.)
    assert wilson(160,200)[0]<.8


def test_reference_batch_cache_matches_direct_full_lstsq_and_production(config):
    design=SyntheticDesign(config,3)
    t=np.arange(153)[:,None];i=np.arange(128)[None,:]
    returns=np.stack([.012*np.sin((t+2)*(i+1)/(63+a)) for a in range(3)])
    returns[:,~design.finite]=np.nan
    signal,_=reference_signal_labels(returns)
    for d in [20,60,96,147]:
        fit=design.fit_masks[d];x,inverse=design.reference[d]
        value=signal[:,d,fit].T
        direct=value-x@np.linalg.lstsq(x,value,rcond=1e-10)[0]
        cached=value-x@(inverse@value)
        np.testing.assert_allclose(cached,direct,atol=1e-12)
    means,counts=reference_statistics(returns,design)
    for a in range(3):
        observed,n=production_statistics(returns[a],design,0)
        assert means[a]==pytest.approx(observed,abs=1e-12)
        np.testing.assert_array_equal(counts[a],n)


def test_receipt_cannot_change_p_value_or_seed_address(tmp_path):
    import polars as pl
    from factor_matrix.storage import DataLake
    from factor_matrix.calculation.services.synthetic_calibration import verify_receipt
    lake=DataLake(tmp_path)
    path=tmp_path/'record.parquet';values=np.arange(100,dtype=float)
    frame=pl.DataFrame({'scenario_id':[0]*100,'role_id':[0]*100,'outer_id':[0]*100,
        'panel_id':range(100),'statistic':values,'seed_prefix':[[20260818,777,0,0,0,p] for p in range(100)],
        'n_valid_by_date':[[128]*128]*100})
    frame.write_parquet(path)
    receipt={'table':lake.artifact_record(path),'observed_statistic':0.,'p_value':1.,'reject':False,
        'scenario':0,'role':0,'outer':0}
    verify_receipt(lake,receipt,0,0,0,np.full(128,128))
    receipt['p_value']=.01;receipt['reject']=True
    with pytest.raises(ValueError,match='RECEIPT_MISMATCH'):
        verify_receipt(lake,receipt,0,0,0,np.full(128,128))
