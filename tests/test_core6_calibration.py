import json
import math
from pathlib import Path

import numpy as np
import pytest

from factor_matrix.calculation.services.core6_calibration import (
    deciles, portfolio_pair, standardized_summary, UnclippedSpecificState,
    run_calibration, _verified_parent,
)
from factor_matrix.storage import file_sha256


def test_portfolio_covariance_includes_common_factor_cross_terms():
    # Perfectly common factor: diversification cannot halve its variance.
    result = portfolio_pair([[1.],[1.]], [[.04]], [.01,.01], [.2,.2], [0.,0.])
    assert result['total_variance'] == pytest.approx(.045)
    assert result['specific_variance'] == pytest.approx(.005)
    assert result['total_z'] == pytest.approx(.2/math.sqrt(.045))
    hedged = portfolio_pair([[1.],[-1.]], [[.04]], [.01,.01], [.1,-.1], [0.,0.])
    assert hedged['total_variance'] == pytest.approx(.005)


def test_missing_outcome_invalidates_exante_portfolio_without_reweighting():
    result = portfolio_pair([[1.],[1.]], [[.04]], [.01,.01], [.2,np.nan], [0.,np.nan])
    assert result['status']=='missing_outcome'
    assert result['assets']==2 and result['missing_outcomes']==1
    assert result['total_variance']==pytest.approx(.045)
    assert result['total_z'] is None and result['realized_return'] is None
    with pytest.raises(ValueError,match='INVALID_FORECAST'):
        portfolio_pair([[1.]],[[.04]],[0.],[0.],[0.])


def test_bias_uses_sample_std_and_exposes_nonzero_mean():
    z=np.array([-1.,1.])/math.sqrt(2)+3
    result=standardized_summary(z,minimum=2)
    assert result['std']==pytest.approx(1.)
    assert result['mean']==pytest.approx(3.)
    assert result['rms']>3
    assert result['within_095_105'] and result['within_091_109']
    assert standardized_summary(z)['status']=='unavailable'
    assert standardized_summary(np.zeros(249),minimum=250)['std'] is None
    with pytest.raises(ValueError,match='NONFINITE'):
        standardized_summary([1.,np.nan])


def test_decile_tie_break_and_unequal_group_sizes():
    ids=[f'A{i:02d}' for i in range(23)][::-1]
    result=deciles(ids,np.ones(23))
    mapping=dict(zip(ids,result))
    assert [sum(result==g) for g in range(1,11)]==[3,3,3,2,2,2,2,2,2,2]
    assert mapping['A00']==1 and mapping['A22']==10
    assert deciles(['B','A'],[0.,0.]).tolist()==[2,1]
    with pytest.raises(ValueError,match='AXIS'):
        deciles(['A','A'],[1.,2.])


def test_unclipped_shadow_matches_independent_ragged_batch_and_is_past_only():
    half=10.393355749081836
    state=UnclippedSpecificState(half)
    history={'A':[], 'B':[]}
    for i in range(90):
        row={'A':.01*math.sin(i),'B':None if i%7==0 else .02*math.cos(i)}
        if i==80:row['A']=.9
        state.update(row)
        for a,v in row.items():
            if v is not None:history[a].append(v)
    got=state.forecast({'A':1,'B':1})
    raw={};neff={}
    for a,values in history.items():
        w=2.**(-np.arange(len(values)-1,-1,-1)/half)
        raw[a]=float(w@np.square(values)/w.sum())
        neff[a]=w.sum()**2/(w@w)
    target=np.mean(list(raw.values()))
    for a in history:
        shrink=120/(120+neff[a])
        assert got[a]==pytest.approx((1-shrink)*raw[a]+shrink*target,abs=1e-15)
    frozen=dict(got)
    state.update({'A':10.,'B':None})
    assert got==frozen
    assert state.forecast({'A':1,'B':1})['B'] is None


def test_shadow_minimum_history_and_asset_clock():
    state=UnclippedSpecificState(10.)
    for i in range(59):state.update({'A':.01})
    assert state.forecast({'A':1})['A'] is None
    state.update({'A':.01})
    assert state.forecast({'A':1})['A']==pytest.approx(.0001)
    state.update({'A':None})
    assert state.forecast({'A':1})['A'] is None


def test_changed_input_artifact_is_rejected(tmp_path):
    artifact=tmp_path/'input.txt';artifact.write_text('original')
    m=tmp_path/'_MANIFEST.json'
    m.write_text(json.dumps({'temporal_mode':'reconstructed_diagnostic_only','production_eligible':False,
        'outputs':{'input.txt':{'sha256':file_sha256(artifact)}}}))
    record={'path':'_MANIFEST.json','sha256':file_sha256(m)}
    _verified_parent(tmp_path,record)
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='PARENT_ARTIFACT'):
        _verified_parent(tmp_path,record)


def test_config_hash_required_before_any_output(tmp_path):
    p=tmp_path/'config.json';p.write_text('{}')
    with pytest.raises(ValueError,match='CONFIG_HASH'):
        run_calibration(project_root=tmp_path,config_path=p,config_sha256='wrong',output_root=tmp_path/'out')
    assert not (tmp_path/'out').exists()
