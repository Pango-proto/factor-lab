from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import json
import math

import numpy as np
import polars as pl
import pytest

from factor_matrix.calculation.l1.momentum import load_core6_design,momentum_descriptor,MomentumDefinition
from factor_matrix.calculation.services.core6_inputs import derive_board_benchmarks,industry_intervals
from factor_matrix.calculation.services.core6_model import build_core6_day,fit_core6_day,estimate_core6_risk
from factor_matrix.calculation.services.core6_reconstruction import momentum_panel,liquidity_groups
from factor_matrix.calculation.services.core6_pipeline import audit_universe
from factor_matrix.calculation.l2.config import load_return_decomposition_config
from factor_matrix.calculation.l2.label_policy import load_return_label_policy
from factor_matrix.storage import open_duckdb
from factor_matrix.research_protocol import ResearchProtocol

ROOT=Path(__file__).resolve().parents[1]
# Fixed artificial dimensions, derived from the versioned protocol; no local market artifacts.
PARAMS=ResearchProtocol.load(ROOT/'config/research_protocol_v1.json').derive(
    factor_count=38, maximum_evaluation_horizon_days=20,
    cross_section_size=5000, available_estimation_days=1400)['values']


def test_board_null_return_does_not_dilute_weight_and_prior_nonnull_cap():
    c=open_duckdb();now=datetime(2001,2,1,tzinfo=timezone.utc)
    c.register('returns_daily',pl.DataFrame({'trade_date':[date(2001,1,3)]*2,'asset_id':['A','B'],'total_return':[.1,None],'known_at':[now]*2}))
    c.register('valuation_daily',pl.DataFrame({'trade_date':[date(2001,1,1),date(2001,1,2),date(2001,1,1)],'asset_id':['A','A','B'],'float_mkt_cap':[100.,None,900.],'known_at':[now]*3}))
    c.register('security_daily_state',pl.DataFrame({'trade_date':[date(2001,1,3)]*2,'asset_id':['A','B'],'board_id':['MAIN']*2,'in_a_share_scope':[True]*2,'listed_as_of':[True]*2,'known_at':[now]*2}))
    result=derive_board_benchmarks(c,'2001-01-03','2001-01-03').row(0,named=True);c.close()
    assert result['board_return']==pytest.approx(.1)
    assert result['constituent_count']==1 and result['effective_weight_base']==100
    assert result['excluded_observations']==1 and result['source_available_at']==now


def test_industry_gap_inherits_both_adjacent_observation_times():
    early=datetime(2001,2,1,tzinfo=timezone.utc);late=early+timedelta(days=5)
    data=pl.DataFrame([{'classification_standard':'SW2021','asset_id':'A','in_date':date(2001,1,1),'out_date':date(2001,1,3),
        'l1_code':'x','l1_name':'x','l2_code':'y','l2_name':'y','known_at':early},
        {'classification_standard':'SW2021','asset_id':'A','in_date':date(2001,1,8),'out_date':None,
        'l1_code':'x','l1_name':'x','l2_code':'z','l2_name':'z','known_at':late}])
    gap=industry_intervals(data).filter(pl.col('filled_from_adjacent')).row(0,named=True)
    assert gap['source_available_at']==late and gap['l2_code'] is None
    assert gap['effective_from']==date(2001,1,4) and gap['effective_to']==date(2001,1,7)


def test_compiled_momentum_matches_reviewed_raw_function():
    days=[date(2001,1,1)+timedelta(days=i) for i in range(300)]
    now=datetime(2003,1,1,tzinfo=timezone.utc);rng=np.random.default_rng(29)
    rows=[]
    for asset in ['normal','partial','invalid','loss']:
        for i,d in enumerate(days):
            r=float(rng.normal(0,.01));source='synthetic'
            if asset=='partial' and 48<=i<70:r=None
            if asset=='invalid' and i==50:r=math.nan
            if asset=='loss' and i==55:r=-1.
            if i==90:source='resumption'
            rows.append({'trade_date':d,'asset_id':asset,'total_return':r,'return_source':source,
                'known_at':now,'available_at':now,'availability_evidence':'observed_first_seen'})
    data=pl.DataFrame(rows);definition=MomentumDefinition(21,251,200)
    fast=momentum_panel(data,days,definition,now).filter(pl.col('trade_date')==days[-1]).sort('asset_id')
    reference=momentum_descriptor(as_of=days[-1],decision_time=now,calendar=days,
        assets=data.select('asset_id').unique(),returns=data,definition=definition).sort('asset_id')
    assert fast['n_momentum_obs'].to_list()==reference['n_momentum_obs'].to_list()
    for a,b in zip(fast['momentum_raw'],reference['momentum_raw']):
        if b is None:assert a is None
        else:assert a==pytest.approx(b,abs=1e-12)


@pytest.fixture(scope='module')
def built(core6_design_project):
    spec,momentum=load_core6_design(core6_design_project/'config/risk_core6_design_v2.json',project_root=core6_design_project)
    rng=np.random.default_rng(123);n=310;days=[date(2001,1,1)+timedelta(days=i) for i in range(280)]
    ids=[f'FIXTURE{i:04d}' for i in range(n)];now=datetime(2003,1,1,tzinfo=timezone.utc)
    industries=[c.removeprefix('risk_industry_').replace('_SI','.SI') for c in spec['industry_columns']]
    assets=pl.DataFrame({'trade_date':[days[-1]]*n,'asset_id':ids,'sw_l1_code':[industries[i//10] for i in range(n)],
        'sw_l2_code':['test']*n,'board_id':['MAIN']*n,'float_mkt_cap':np.exp(rng.normal(23,1,n)),
        'exchange_list_date':[date(1999,1,1)]*n})
    returns=[];valuation=[];states=[];board=[]
    beta=rng.uniform(.4,1.7,n);sigma=rng.uniform(.002,.012,n)
    for d in days:
        b=float(rng.normal(0,.01));board.append({'trade_date':d,'board_id':'MAIN','board_return':b,'source_available_at':now})
        for i,asset in enumerate(ids):
            returns.append({'trade_date':d,'asset_id':asset,'total_return':float(beta[i]*b+rng.normal(0,sigma[i])),
                'return_source':'synthetic','known_at':now,'available_at':now,'availability_evidence':'observed_first_seen'})
            valuation.append({'trade_date':d,'asset_id':asset,'float_mkt_cap':assets['float_mkt_cap'][i],
                'turnover_rate':float(rng.lognormal(i/1000,.3)),'known_at':now})
            states.append({'trade_date':d,'asset_id':asset,'board_id':'MAIN','known_at':now})
    returns=pl.DataFrame(returns);valuation=pl.DataFrame(valuation);states=pl.DataFrame(states);board=pl.DataFrame(board)
    params=json.loads((ROOT/'config/risk_descriptor_parameters_v1.json').read_text())['definitions']
    result=build_core6_day(as_of=days[-1],knowledge_time=now,assets=assets,universe_history=states,
        returns=returns,valuation=valuation,board=board,calendar=days,spec=spec,momentum_definition=momentum,
        parameters=PARAMS,descriptor_parameters=params,metric_weights={},source_available_at=now)
    return result,spec,now


def test_full_six_style_transform_and_g3_accounting(built):
    result,spec,now=built;x=result['exposure'];n=x.height
    assert set(c for c in x.columns if c.startswith('risk_'))==set(spec['expanded_columns'])
    assert len(spec['expanded_columns'])==38
    assert result['provenance'].height==n*6
    assert result['quality']['max_abs_corr_with_controls'].max()<1e-8
    assert all(result['provenance']['metric_fallback'])
    next_day=x['trade_date'][0]+timedelta(days=1)
    y=np.asarray(x.select(spec['expanded_columns']))@np.linspace(-.005,.005,38)+np.random.default_rng(19).normal(0,.003,n)
    labels=pl.DataFrame({'asset_id':x['asset_id'],'trade_date':[next_day]*n,'total_return':y,'return_source':['synthetic']*n})
    prices=pl.DataFrame({'asset_id':x['asset_id'],'trade_date':[next_day]*n,**{c:[10.]*n for c in ['raw_open','raw_high','raw_low','raw_close']},'limit_up':[11.]*n,'limit_down':[9.]*n})
    kwargs=dict(exposure=x,labels=labels,prices=prices,spec=spec,
        config=load_return_decomposition_config(ROOT/'config/l2_return_decomposition_v1.json',derived_params=PARAMS),
        label_policy=load_return_label_policy(),next_session=next_day,outcome_start=now+timedelta(minutes=5),exposure_available_at=now)
    fit=fit_core6_day(**kwargs)
    assert fit.status=='passed' and fit.sample_count==n
    assert fit.regression_identity_error<1e-12 and fit.constraint_error<1e-8
    with pytest.raises(ValueError,match='BEFORE_OUTCOME'):
        fit_core6_day(**{**kwargs,'exposure_available_at':now+timedelta(minutes=6)})
    prices=prices.with_columns(pl.lit(next_day+timedelta(days=1)).alias('trade_date'))
    with pytest.raises(ValueError,match='PRICE_SESSION'):fit_core6_day(**{**kwargs,'prices':prices})


def test_ragged_delta_keeps_incomplete_asset_visible_and_future_rows_excluded():
    t=datetime(2001,6,1,tzinfo=timezone.utc);rng=np.random.default_rng(9)
    rows=[{'trade_date':date(2001,1,1)+timedelta(days=i),'available_at':t,'basis_id':'basis',
        'factor_returns':{'a':float(rng.normal(0,.01)),'b':float(rng.normal(0,.01))},
        'specific_returns':{'full':float(rng.normal(0,.01)),**({'short':.003} if i%2 else {})}} for i in range(80)]
    args=dict(observations=rows,asset_groups={'full':'1','short':'1'},factor_ids=['a','b'],basis_id='basis',decision_time=t,parameters=PARAMS)
    report=estimate_core6_risk(**args)
    assert report['status']=='partial_asset_coverage'
    assert report['Delta']['full']>0 and report['Delta']['short'] is None
    assert report['asset_coverage']['short']['observations']==40
    future={**rows[0],'available_at':t+timedelta(days=1),'factor_returns':{'bad':math.nan}}
    assert estimate_core6_risk(**{**args,'observations':rows+[future]})==report
    empty=estimate_core6_risk(**{**args,'observations':[]})
    assert empty['F'] is None and all(v is None for v in empty['Delta'].values())


def test_liquidity_buckets_follow_ntile_with_deterministic_ties():
    x=pl.DataFrame({'asset_id':[f'{i:02}' for i in range(23)],'risk_liquidity':[1.]*23})
    groups=liquidity_groups(x)
    assert [list(groups.values()).count(str(i)) for i in range(1,11)]==[3,3,3,2,2,2,2,2,2,2]


def test_universe_reports_overlapping_exclusions_without_changing_denominator():
    x=pl.DataFrame({'asset_id':['OK','ST_NEW','SUSP'],'is_st':[False,True,False],
        'is_suspended':[False,False,True],'board_id':['MAIN']*3,'days_since_exchange_list':[30,4,90],
        'raw_close':[10.,10.,None],'float_mkt_cap':[1e8]*3,'sw_l1_code':['x']*3})
    out=audit_universe(x,policy={'exclude_st':True,'exclude_suspended':True,'include_bse':True},d0=20)
    assert out.height==3 and out.filter('risk_eligible')['asset_id'].to_list()==['OK']
    assert out['excluded_new_listing'].sum()==1 and out['excluded_st'].sum()==1


def test_streaming_estimator_matches_registered_batch_with_gaps_and_outliers():
    from factor_matrix.calculation.services.core6_rolling import RollingRiskState
    from factor_matrix.calculation.l2.risk_modeling import ewma_factor_covariance,ewma_specific_variance
    state=RollingRiskState(['A','B','C'],['f1','f2'],PARAMS)
    rng=np.random.default_rng(66);f={'f1':[],'f2':[]};u={'A':[],'B':[],'C':[]}
    for i in range(330):
        fr={k:float(rng.normal(0,.01)) for k in f}
        sr={a:float(rng.normal(0,.01)) for a in u if a!='B' or i%3!=0}
        if i in (89,180,300):sr['A']=.9
        for k,v in fr.items():f[k].append(v)
        for k,v in sr.items():u[k].append(v)
        state.update(fr,sr)
        if i in (99,329):
            groups={a:'1' for a in sr if len(u[a])>=60}
            F,D=state.forecast(groups)
            expected_F=ewma_factor_covariance(f,PARAMS['ewma_factor_covariance_half_life_days'])
            expected_D=ewma_specific_variance({a:u[a] for a in groups},PARAMS['ewma_specific_variance_half_life_days'],group_by_asset=groups)
            for j,a in enumerate(f):
                for k,b in enumerate(f):assert F[j,k]==pytest.approx(expected_F[a,b],abs=1e-15)
            assert D==pytest.approx(expected_D,abs=1e-15)
