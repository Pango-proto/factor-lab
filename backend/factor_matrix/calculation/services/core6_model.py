"""Opt-in six-style L1, constrained G3 and sparse-asset F/Delta estimation.

Uses the existing transforms, regression and variance estimators unchanged.
Publishing/admission and actual historical availability are separate concerns.
"""
from datetime import date, datetime
import math

import numpy as np
import polars as pl

from ...canonical_definitions import exposure_orthogonalization_weight
from ..l1.builder import _impute_industry_median
from ..l1.descriptors import market_sensitivity_descriptors, liquidity_descriptor
from ..l1.momentum import momentum_descriptor
from ..l1.style_math import transform_style_column
from ..l2.design_matrix import build_daily_categorical_constrained_design
from ..l2.return_decomposition import decompose_cross_section
from ..l2.contracts import RegressionMode
from ..l2.label_policy import apply_l2b_label_policy
from ..l2.risk_modeling import ewma_factor_covariance, ewma_specific_variance


def build_core6_day(*, as_of, knowledge_time, assets, universe_history, returns,
                   valuation, board, calendar, spec, momentum_definition,
                   parameters, descriptor_parameters, metric_weights, source_available_at,
                   momentum_override=None):
    """assets is the audited frozen-d0, industry/cap complete-case domain."""
    if assets.is_empty() or assets['asset_id'].n_unique() != assets.height:
        raise ValueError('CORE6_EMPTY_OR_DUPLICATE_ASSET_AXIS')
    if source_available_at > knowledge_time:
        raise ValueError('CORE6_SOURCE_NOT_AVAILABLE')
    for frame in (returns, valuation, universe_history):
        if 'known_at' not in frame.columns or frame['known_at'].null_count() or frame['known_at'].max() > knowledge_time:
            raise ValueError('CORE6_SOURCE_NOT_AVAILABLE')
    if board['source_available_at'].null_count() or board['source_available_at'].max() > knowledge_time:
        raise ValueError('CORE6_BOARD_NOT_AVAILABLE')
    assets = assets.sort('asset_id')
    categories = ['risk_industry_' + c.replace('.', '_') for c in assets['sw_l1_code']]
    if set(categories) != set(spec['industry_columns']):
        raise ValueError('CORE6_FIXED_INDUSTRY_AXIS_UNAVAILABLE')
    if min(categories.count(c) for c in spec['industry_columns']) < parameters['minimum_category_members']:
        raise ValueError('CORE6_MINIMUM_INDUSTRY_MEMBERS')
    weights = [metric_weights.get(a, exposure_orthogonalization_weight(cap))
               for a, cap in zip(assets['asset_id'], assets['float_mkt_cap'])]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('CORE6_WEIGHT_INVALID')
    beta = descriptor_parameters['beta']
    liquidity = descriptor_parameters['liquidity']
    raw = assets.join(market_sensitivity_descriptors(
        as_of=as_of, assets=assets, universe_history=universe_history, returns=returns,
        board_benchmarks=board, lookback_days=beta['lookback_days'],
        minimum_observations=beta['minimum_observations']), on='asset_id', validate='1:1')
    raw = raw.join(liquidity_descriptor(as_of=as_of, assets=assets, valuation=valuation,
        lookback_days=liquidity['lookback_days'], minimum_observations=liquidity['minimum_observations']),
        on='asset_id', validate='1:1').with_columns(pl.col('float_mkt_cap').log().alias('size_raw'))
    momentum = momentum_override if momentum_override is not None else momentum_descriptor(
        as_of=as_of, decision_time=knowledge_time, calendar=calendar, assets=assets,
        returns=returns, definition=momentum_definition)
    if set(momentum['asset_id']) != set(assets['asset_id']) or momentum['asset_id'].n_unique() != momentum.height:
        raise ValueError('CORE6_MOMENTUM_AXIS')
    raw = raw.join(momentum.select('asset_id', 'momentum_raw', 'n_momentum_obs'), on='asset_id', validate='1:1')
    industry = [[float(category == col) for category in categories] for col in spec['industry_columns']]
    exposures = {'risk_country': [1.]*assets.height, **dict(zip(spec['industry_columns'], industry))}
    transformed, quality, provenance = {}, [], []
    size_standardized = None
    for factor in spec['style_order']:
        predecessors = (spec['momentum']['orthogonalize_after'] if factor == 'momentum' else
                        spec['legacy_style_definitions']['orthogonalize_after'][factor])
        controls = industry + [transformed[f] for f in predecessors]
        if factor == 'nonlinear_size':
            values = [v**descriptor_parameters['nonlinear_size']['power'] for v in size_standardized]
            original = values
            nvalid, nimputed, coverage, fallback = assets.height, 0, 1., False
        else:
            values, nvalid, nimputed, coverage, fallback = _impute_industry_median(raw, factor+'_raw')
            original = raw[factor+'_raw'].to_list()
        if values is None or coverage < parameters['coverage_min']:
            raise ValueError('CORE6_RAW_COVERAGE_FAILED:'+factor)
        result = transform_style_column(values, controls=list(map(list, zip(*controls))),
            control_columns=controls, weights=weights, mad_k=parameters['mad_k'],
            already_standardized=factor == 'nonlinear_size')
        if abs(result.mean_w) > 1e-8 or abs(result.sd_w-1) > 1e-8 or result.max_abs_corr_with_controls > 1e-8:
            raise ValueError('CORE6_TRANSFORM_CHECK_FAILED:'+factor)
        if factor == 'size': size_standardized = result.standardized_raw
        transformed[factor] = result.values
        exposures['risk_'+factor] = list(result.values)
        quality.append({'factor_id': factor, 'n_raw_valid': nvalid, 'n_imputed': nimputed,
                        'raw_coverage': coverage, 'market_fallback_used': fallback,
                        'mean_w': result.mean_w, 'sd_w': result.sd_w,
                        'max_abs_corr_with_controls': result.max_abs_corr_with_controls})
        valid_industries = {c for c,v in zip(assets['sw_l1_code'], original) if v is not None and math.isfinite(v)}
        obs_col = {'beta':'n_beta_obs', 'residual_volatility':'n_beta_obs',
                   'liquidity':'n_liquidity_obs', 'momentum':'n_momentum_obs'}.get(factor)
        counts = raw[obs_col].fill_null(0).to_list() if obs_col else [1]*assets.height
        for a,c,v,filled,n,weight in zip(assets['asset_id'], assets['sw_l1_code'], original, values, counts, weights):
            missing = v is None or not math.isfinite(v)
            provenance.append({'asset_id':a,'factor_id':factor,'raw_value':v,'transform_input':filled,
                'raw_missing':missing,'imputed':missing,'fallback':missing and c not in valid_industries,
                'n_obs':n,'source_available_at':source_available_at,'metric_weight':weight,
                'metric_fallback':a not in metric_weights})
    x = assets.select('trade_date','asset_id','board_id','sw_l1_code','sw_l2_code','float_mkt_cap','exchange_list_date').with_columns(
        *[pl.Series(k, v) for k,v in exposures.items()],
        pl.Series('candidate_weight', weights), pl.lit(True).alias('is_valid'),
        pl.lit(source_available_at).alias('source_available_at'))
    return {'exposure':x,'quality':pl.DataFrame(quality),'provenance':pl.DataFrame(provenance),'momentum':momentum}


def fit_core6_day(*, exposure, labels, prices, spec, config, label_policy,
                 next_session, outcome_start, exposure_available_at, categorical_blocks=None):
    """Require actual exposure publication before the next-session outcome."""
    if exposure_available_at >= outcome_start:
        raise ValueError('CORE6_EXPOSURE_NOT_AVAILABLE_BEFORE_OUTCOME')
    if next_session <= exposure['trade_date'][0]:
        raise ValueError('CORE6_LABEL_NOT_NEXT_SESSION')
    if labels['trade_date'].n_unique() != 1 or labels['trade_date'][0] != next_session:
        raise ValueError('CORE6_LABEL_SESSION_MISMATCH')
    if labels['asset_id'].n_unique() != labels.height or prices['asset_id'].n_unique() != prices.height:
        raise ValueError('CORE6_LABEL_DUPLICATE')
    joined = exposure.join(labels.select('asset_id','total_return','return_source'), on='asset_id', how='left', validate='1:1')
    joined = joined.join(prices.select('asset_id','raw_open','raw_high','raw_low','raw_close','limit_up','limit_down'),
                         on='asset_id', how='left', validate='1:1').with_columns(pl.lit(next_session).alias('return_date'))
    joined = apply_l2b_label_policy(joined,label_policy,label_date_column='return_date')
    up = pl.all_horizontal(*[pl.col(c) >= pl.col('limit_up') for c in ('raw_open','raw_high','raw_low','raw_close')])
    down = pl.all_horizontal(*[pl.col(c) <= pl.col('limit_down') for c in ('raw_open','raw_high','raw_low','raw_close')])
    joined = joined.with_columns(pl.col('label_eligible').fill_null(False).alias('model_eligible'),
        (pl.col('label_eligible') & ~(up|down).fill_null(False)).fill_null(False).alias('in_estimation_domain'),
        pl.col('total_return').alias('realized_return'))
    if prices['trade_date'].n_unique() != 1 or prices['trade_date'][0] != next_session:
        raise ValueError('CORE6_PRICE_SESSION_MISMATCH')
    for col in spec['industry_columns']:
        if joined.filter('in_estimation_domain')[col].sum() < 6:
            raise ValueError('CORE6_G3_INDUSTRY_SUPPORT_INSUFFICIENT:'+col)
    design = build_daily_categorical_constrained_design(joined, exposure_columns=spec['expanded_columns'],
        family_by_column={f:'risk' for f in spec['expanded_columns']},
        categorical_blocks=categorical_blocks or {'industry':spec['industry_columns']}, minimum_category_members=6,
        estimation_column='in_estimation_domain', base_weight_column='candidate_weight')
    if not design.constraint_identification_passed:
        raise ValueError('CORE6_G3_CONSTRAINT_IDENTIFICATION')
    result = decompose_cross_section(design.regression_input,mode=RegressionMode.RISK_ONLY,config=config)
    if result.status != 'passed':
        raise ValueError('CORE6_G3_NUMERICAL_GATE:'+result.status)
    return result


def estimate_core6_risk(*, observations, asset_groups, factor_ids, basis_id, decision_time, parameters):
    """No shrinking asset intersection: report every requested asset's history."""
    if not asset_groups or len(set(factor_ids)) != len(factor_ids):
        raise ValueError('CORE6_FORECAST_AXIS_INVALID')
    rows = sorted([r for r in observations if r['available_at'] <= decision_time],key=lambda r:r['trade_date'])
    if len({r['trade_date'] for r in rows}) != len(rows):
        raise ValueError('CORE6_DUPLICATE_G3_DATE')
    for r in rows:
        if r['basis_id'] != basis_id or set(r['factor_returns']) != set(factor_ids):
            raise ValueError('CORE6_G3_BASIS_OR_AXIS')
        if not all(math.isfinite(v) for v in r['factor_returns'].values()):
            raise ValueError('CORE6_G3_NONFINITE_FACTOR')
        if any(v is not None and not math.isfinite(v) for v in r['specific_returns'].values()):
            raise ValueError('CORE6_G3_NONFINITE_SPECIFIC')
    histories = {a:[r['specific_returns'][a] for r in rows if r['specific_returns'].get(a) is not None] for a in asset_groups}
    coverage = {a:{'observations':len(values),'last_date':next((str(r['trade_date']) for r in reversed(rows) if r['specific_returns'].get(a) is not None),None),
        'status':'eligible' if len(values)>=60 and rows[-1]['specific_returns'].get(a) is not None else 'unavailable'} for a,values in histories.items()}
    report = {'basis_id':basis_id,'factor_ids':factor_ids,'factor_observations':len(rows),
        'asset_coverage':coverage,'F':None,'Delta':{a:None for a in asset_groups},
        'status':'unavailable','reason':'insufficient_factor_history','minimum_history':60,
        'calibration_status':'not_calibrated','production_eligible':False,'horizon_sessions':1}
    if len(rows)<60: return report
    cov = ewma_factor_covariance({f:[r['factor_returns'][f] for r in rows] for f in factor_ids},parameters['ewma_factor_covariance_half_life_days'])
    F=np.array([[cov[a,b] for b in factor_ids] for a in factor_ids])
    if not np.isfinite(F).all() or np.linalg.eigvalsh(F).min() < -1e-12:
        raise ValueError('CORE6_F_NOT_PSD')
    valid={a:values for a,values in histories.items() if coverage[a]['status']=='eligible'}
    delta = ewma_specific_variance(valid,parameters['ewma_specific_variance_half_life_days'],
        group_by_asset={a:asset_groups[a] for a in valid},lookback_days=252,minimum_observations=60,
        clip_multiple=5.,prior_effective_observations=120.) if valid else {}
    for a,value in delta.items():
        if not math.isfinite(value) or value<=0: coverage[a]['status']='nonpositive_variance'
        else: report['Delta'][a]=value
    report.update(F=F.tolist(),status='computed_not_calibrated' if all(v is not None for v in report['Delta'].values()) else 'partial_asset_coverage',reason=None)
    return report
