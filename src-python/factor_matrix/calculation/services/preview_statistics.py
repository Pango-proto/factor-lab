"""Descriptive statistics for synthetic charts; no research gating or inference."""
import math
import numpy as np


def describe_returns(values, *, include_histogram=True):
    r = np.asarray(values, dtype=float)
    if len(r) < 21 or not np.isfinite(r).all():
        raise ValueError('FINITE_RETURN_SAMPLE_REQUIRED')
    n, mean, std = len(r), float(r.mean()), float(r.std(ddof=1))
    centered = r-mean
    m2 = float(np.mean(centered**2))
    q = float(np.quantile(r, .05, method='linear'))
    tail = r[r<=q]
    density, normal = [], []
    if include_histogram:
        edges = np.linspace(-.06, .06, 49)
        count, _ = np.histogram(r, edges)
        if count.sum() != n:
            raise ValueError('HISTOGRAM_SUPPORT_EXCEEDED')
        density = [{'x':float(edges[i]),'y':float(count[i]/n/(edges[i+1]-edges[i]))} for i in range(len(count))]
        density += [{'x':float(edges[-1]),'y':density[-1]['y']}]
        normal = [{'x':float(x), 'y':float(math.exp(-.5*((x-mean)/std)**2)/(std*math.sqrt(2*math.pi)))} for x in np.linspace(-.06,.06,241)] if std else []
    return {'n':n,'mean':mean,'std':std,'annualized_volatility':std*math.sqrt(252),
            'sharpe':mean/std*math.sqrt(252) if std else None,
            'skewness':float(np.mean(centered**3)/m2**1.5) if m2 else None,
            'excess_kurtosis':float(np.mean(centered**4)/m2**2-3) if m2 else None,
            'quantile_05':q,'tail_mean_05':float(tail.mean()),'var_95_loss':-q,'cvar_95_loss':-float(tail.mean()),
            'histogram':density,'normal':normal,
            'definition':'sample_std_ddof1; moment_skew_and_excess_kurtosis; linear_quantile; empirical_tail_mean; rf0'}


def price_diagnostics(prices):
    r = np.diff(prices)/np.asarray(prices[:-1])
    absolute = np.abs(r)
    demeaned = absolute-absolute.mean()
    denominator = float(demeaned@demeaned)
    acf = [{'x':lag,'y':float(demeaned[:-lag]@demeaned[lag:]/denominator) if denominator else 0.} for lag in range(1,21)]
    return {'returns':[{'x':i+1,'y':float(v)} for i,v in enumerate(r)],'acf':acf,
            'acf_reference':1.96/math.sqrt(len(r)), 'acf_reference_kind':'pointwise_white_noise_approximation_not_joint_test',
            'rolling_volatility':[{'x':i+1,'y':float(r[i-19:i+1].std(ddof=1))} for i in range(19,len(r))],
            'object':'unadjusted_synthetic_price_close_to_close_returns_no_corporate_actions',
            'rolling_window':20,'annualized':False}
