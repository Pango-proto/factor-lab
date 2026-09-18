"""Fixed reconstructed forecast/outcome diagnostics, never a release decision."""
from pathlib import Path
import json
import math

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, utc_now
from .core6_reconstruction import liquidity_groups


MODELS = ('core6', 'old44')


def deciles(ids, values):
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError('CALIBRATION_DECILE_AXIS')
    if not np.isfinite(values).all():
        raise ValueError('CALIBRATION_DECILE_NONFINITE')
    order = sorted(range(len(ids)), key=lambda i: (values[i], ids[i]))
    result = np.zeros(len(ids), dtype=int)
    q, r = divmod(len(ids), 10)
    offset = 0
    for g in range(10):
        width = q + (g < r)
        result[order[offset:offset + width]] = g + 1
        offset += width
    return result


def standardized_summary(z, minimum=60):
    """No silent omission: callers explicitly choose valid rows or whole windows."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError('CALIBRATION_NONFINITE_STANDARDIZED')
    if len(z) < minimum:
        return {'status': 'unavailable', 'observations': len(z), 'std': None,
                'mean': None, 'rms': None, 'abs_gt_3_fraction': None,
                'within_095_105': None, 'within_091_109': None}
    std = float(np.std(z, ddof=1))
    return {'status': 'diagnostic_only', 'observations': len(z), 'std': std,
            'mean': float(np.mean(z)), 'rms': float(np.sqrt(np.mean(z*z))),
            'abs_gt_3_fraction': float(np.mean(np.abs(z) > 3)),
            'within_095_105': .95 <= std <= 1.05,
            'within_091_109': .91 <= std <= 1.09}


def portfolio_pair(X, F, delta, realized, specific):
    """Ex-ante equal weights; any missing outcome invalidates the whole portfolio."""
    X, F, delta, realized, specific = [np.asarray(v, dtype=float)
                                      for v in (X, F, delta, realized, specific)]
    n = len(delta)
    if X.ndim != 2 or F.shape != (X.shape[1], X.shape[1]) or X.shape[0] != n:
        raise ValueError('CALIBRATION_PORTFOLIO_AXIS')
    if realized.shape != (n,) or specific.shape != (n,):
        raise ValueError('CALIBRATION_OUTCOME_AXIS')
    if n == 0:
        return {'status': 'empty_group', 'assets': 0, 'missing_outcomes': 0}
    if not all(np.isfinite(v).all() for v in (X, F, delta)) or np.any(delta <= 0):
        raise ValueError('CALIBRATION_INVALID_FORECAST')
    b = X.mean(axis=0)
    sv = float(delta.sum() / n**2)
    tv = float(b @ F @ b + sv)
    if not math.isfinite(tv) or tv <= 0:
        raise ValueError('CALIBRATION_INVALID_PORTFOLIO_VARIANCE')
    missing = int(np.sum(~np.isfinite(realized) | ~np.isfinite(specific)))
    result = {'status': 'missing_outcome' if missing else 'paired', 'assets': n,
              'missing_outcomes': missing, 'total_variance': tv, 'specific_variance': sv,
              'realized_return': None, 'specific_return': None, 'total_z': None, 'specific_z': None}
    if not missing:
        r, u = float(realized.mean()), float(specific.mean())
        result.update(realized_return=r, specific_return=u,
                      total_z=r/math.sqrt(tv), specific_z=u/math.sqrt(sv))
    return result


class UnclippedSpecificState:
    """Shadow estimator on the identical valid-observation clock; no outcomes ahead."""
    def __init__(self, half_life):
        self.decay = math.exp(math.log(.5)/half_life)
        self.rows = {}
        self.index = -1

    def update(self, values):
        self.index += 1
        for asset, value in values.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError('CALIBRATION_SHADOW_NONFINITE')
            count, sw, sw2, sq, _ = self.rows.get(asset, (0, 0., 0., 0., -1))
            d = self.decay
            self.rows[asset] = count+1, d*sw+1, d*d*sw2+1, d*sq+value*value, self.index

    def forecast(self, groups):
        usable = {a: self.rows[a] for a in groups if a in self.rows
                  and self.rows[a][0] >= 60 and self.rows[a][4] == self.index}
        raw = {a: row[3]/row[1] for a, row in usable.items()}
        by_group = {}
        for a, value in raw.items():
            by_group.setdefault(groups[a], []).append(value)
        means = {g: float(np.mean(v)) for g, v in by_group.items()}
        result = dict.fromkeys(groups)
        for a, row in usable.items():
            neff = row[1]**2/row[2]
            strength = 120/(120+neff)
            value = (1-strength)*raw[a]+strength*means[groups[a]]
            result[a] = value if math.isfinite(value) and value > 0 else None
        return result


def _verified_parent(root, record):
    path = (root/record['path']).resolve()
    if file_sha256(path) != record['sha256']:
        raise ValueError('CALIBRATION_PARENT_HASH')
    m = json.loads(path.read_text())
    if m.get('temporal_mode') != 'reconstructed_diagnostic_only' or m.get('production_eligible'):
        raise ValueError('CALIBRATION_PARENT_MODE')
    for relative, artifact in m['outputs'].items():
        target = (path.parent/relative).resolve()
        if not target.is_relative_to(path.parent) or file_sha256(target) != artifact['sha256']:
            raise ValueError('CALIBRATION_PARENT_ARTIFACT:'+relative)
    return path.parent, m


def _read_frame(directory, manifest, relative):
    if relative not in manifest['outputs']:
        raise ValueError('CALIBRATION_UNLISTED_INPUT:'+relative)
    frame = pl.read_parquet(directory/relative)
    if 'asset_id' in frame.columns and frame['asset_id'].n_unique() != frame.height:
        raise ValueError('CALIBRATION_DUPLICATE_ASSET')
    return frame


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run_calibration(*, project_root: Path, config_path: Path, config_sha256: str, output_root: Path):
    if file_sha256(config_path) != config_sha256:
        raise ValueError('CALIBRATION_CONFIG_HASH')
    config = json.loads(config_path.read_text())
    if config['protocol_id'] != 'core6_calibration_diagnostic_v1' or config['production_eligible']:
        raise ValueError('CALIBRATION_PROTOCOL')
    # The implementation is a fixed ruler, not a configurable threshold search.
    if config['reference_bands'] != {'l2_generic':[.95,1.05], 'g6_stratified':[.91,1.09]}:
        raise ValueError('CALIBRATION_BANDS_CHANGED')
    for relative, digest in config['bindings'].items():
        if file_sha256(project_root/relative) != digest:
            raise ValueError('CALIBRATION_BOUND_CONFIG_CHANGED:'+relative)
    rec, rm = _verified_parent(project_root, config['parents']['reconstruction'])
    roll, fm = _verified_parent(project_root, config['parents']['rolling'])
    if fm['identity']['parent_sha256'] != config['parents']['reconstruction']['sha256']:
        raise ValueError('CALIBRATION_PARENT_LINK')
    dependencies = [Path(__file__), Path(__file__).with_name('core6_reconstruction.py')]
    identity = {'config_sha256': config_sha256, 'parents': config['parents'],
                'code': {p.name:file_sha256(p) for p in dependencies},
                'numpy':np.__version__, 'polars':pl.__version__}
    run_id = 'core6_calibration_'+json_hash(identity)[:16]
    out = output_root/f'run_id={run_id}'
    manifest_path = out/'_MANIFEST.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['identity'] != identity:
            raise ValueError('CALIBRATION_IDENTITY')
        for relative, record in old['outputs'].items():
            if file_sha256(out/relative) != record['sha256']:
                raise ValueError('CALIBRATION_EXISTING_OUTPUT_CHANGED')
        return manifest_path
    if out.exists():
        raise ValueError('CALIBRATION_INCOMPLETE_RUN_RETAINED')
    out.mkdir(parents=True)
    _write_json(out/'contract.json', config)
    meta = json.loads((rec/'inputs/meta.json').read_text())
    statuses = json.loads((rec/'daily_status.json').read_text())
    calendar = meta['calendar']
    schedules = {m:json.loads((roll/(m+'_coverage.json')).read_text()) for m in MODELS}
    covariances = {m:json.loads((roll/(m+'_covariance.json')).read_text()) for m in MODELS}
    schedule = schedules['core6']
    axes = [(r['cutoff'],r['outcome_date']) for r in schedule]
    if len(set(axes)) != len(axes) or len(axes) != config['expected_forecast_dates']:
        raise ValueError('CALIBRATION_DATE_COUNT')
    if axes != sorted(axes) or axes[0][1] != config['start'] or axes[-1][1] != config['end']:
        raise ValueError('CALIBRATION_DATE_BOUNDARY')
    for cutoff, outcome in axes:
        if outcome >= config['holdout_start'] or calendar.index(outcome) != calendar.index(cutoff)+1:
            raise ValueError('CALIBRATION_TIMING')
    for m in MODELS:
        if [(r['cutoff'],r['outcome_date']) for r in schedules[m]] != axes:
            raise ValueError('CALIBRATION_MODEL_DATE_AXIS')
        if [(r['cutoff'],r['outcome_date']) for r in covariances[m]['rows']] != axes:
            raise ValueError('CALIBRATION_COVARIANCE_DATE_AXIS')
    states = {m:UnclippedSpecificState(meta['parameters']['ewma_specific_variance_half_life_days']) for m in MODELS}
    history_index = 0
    portfolios, coverage, individual, shadow_rows = [], [], [], []
    max_identity_error = 0.
    (out/'pairs').mkdir()
    for day_index, (cutoff,outcome) in enumerate(axes):
        while history_index < len(statuses) and statuses[history_index]['return_date'] <= cutoff:
            s = statuses[history_index]
            if s['status'] != 'passed':
                raise ValueError('CALIBRATION_HISTORY_STATUS')
            for m in MODELS:
                u = _read_frame(rec,rm,f"daily/{s['return_date']}/{m}_specific.parquet")
                states[m].update(dict(u.select('asset_id','specific_return').iter_rows()))
            history_index += 1
        blocks, all_ids, x_all, shadow = {}, None, {}, {}
        for m in MODELS:
            pred = _read_frame(roll,fm,f'{m}/{cutoff}.parquet').sort('asset_id')
            if pred['cutoff'].unique().to_list() != [cutoff] or pred['outcome_date'].unique().to_list() != [outcome]:
                raise ValueError('CALIBRATION_ROW_TIMING')
            x = _read_frame(rec,rm,f'daily/{outcome}/{m}_X.parquet').sort('asset_id')
            if [str(v) for v in x['trade_date'].unique()] != [cutoff]:
                raise ValueError('CALIBRATION_X_TIMING')
            if pred['asset_id'].to_list() != x['asset_id'].to_list():
                raise ValueError('CALIBRATION_X_AXIS')
            ids = pred['asset_id'].to_list()
            if all_ids is not None and ids != all_ids:
                raise ValueError('CALIBRATION_MODEL_ASSET_AXIS')
            all_ids = ids
            x_all[m] = x
            shadow[m] = states[m].forecast(liquidity_groups(x))
            columns = covariances[m]['factor_ids']
            F = np.asarray(covariances[m]['rows'][day_index]['F'])
            if F.shape != (len(columns),len(columns)) or not np.isfinite(F).all() or not np.allclose(F,F.T,atol=1e-12) or np.linalg.eigvalsh(F).min() < -1e-12:
                raise ValueError('CALIBRATION_COVARIANCE_INVALID')
            X = x.select(columns).to_numpy()
            d = pred['Delta'].to_numpy()
            total = pred['total_variance'].to_numpy()
            valid = np.isfinite(d) & (d>0) & np.isfinite(total) & (total>0)
            if pred['risk_available'].to_list() != valid.tolist():
                raise ValueError('CALIBRATION_RISK_FLAG')
            recomputed = np.einsum('ij,jk,ik->i',X,F,X)+d
            if not np.allclose(total[valid],recomputed[valid],atol=1e-12,rtol=1e-10):
                raise ValueError('CALIBRATION_VARIANCE_IDENTITY')
            u = _read_frame(rec,rm,f'daily/{outcome}/{m}_specific.parquet')
            f = _read_frame(rec,rm,f'daily/{outcome}/{m}_factor.parquet')
            if set(f['factor_id']) != set(columns) or f.height != len(columns):
                raise ValueError('CALIBRATION_FACTOR_AXIS')
            factor_map = dict(f.iter_rows())
            factor = np.array([factor_map[c] for c in columns])
            specific = pred.select('asset_id').join(u.select('asset_id','specific_return'),on='asset_id',how='left',maintain_order='left')['specific_return'].to_numpy()
            realized = X@factor+specific
            blocks[m] = dict(X=X,F=F,d=d,total=total,valid=valid,u=specific,r=realized)
        mask = blocks['core6']['valid'] & blocks['old44']['valid']
        common_ix = np.flatnonzero(mask)
        ids = [all_ids[i] for i in common_ix]
        if not np.allclose(x_all['core6']['risk_liquidity'].to_numpy(),x_all['old44']['risk_liquidity'].to_numpy(),atol=1e-12):
            raise ValueError('CALIBRATION_LIQUIDITY_AXIS')
        liq = deciles(ids,x_all['core6']['risk_liquidity'].to_numpy()[mask])
        risk = deciles(ids,blocks['core6']['d'][mask])
        valid_outcomes = mask & np.isfinite(blocks['core6']['r']) & np.isfinite(blocks['old44']['r'])
        error = float(np.max(np.abs(blocks['core6']['r'][valid_outcomes]-blocks['old44']['r'][valid_outcomes]))) if valid_outcomes.any() else 0.
        if error > 1e-10:
            raise ValueError('CALIBRATION_REALIZED_RETURN_IDENTITY')
        max_identity_error = max(max_identity_error,error)
        groups = [('overall',0,np.ones(len(ids),dtype=bool))]
        groups += [('liquidity_decile',g,liq==g) for g in range(1,11)]
        groups += [('predicted_specific_risk_decile',g,risk==g) for g in range(1,11)]
        pair = {'asset_id':all_ids,'cutoff':[cutoff]*len(all_ids),'outcome_date':[outcome]*len(all_ids),
                'common_forecast_support':mask.tolist()}
        for name, values in [('liquidity_decile',liq),('core6_risk_decile',risk)]:
            lookup = dict(zip(ids,values.tolist()))
            pair[name] = [lookup.get(a) for a in all_ids]
        for m,b in blocks.items():
            pair[m+'_Delta'] = b['d'].tolist()
            pair[m+'_total_variance'] = b['total'].tolist()
            pair[m+'_specific_return'] = b['u'].tolist()
            pair[m+'_realized_return'] = b['r'].tolist()
            pair[m+'_unclipped_Delta'] = [shadow[m][a] for a in all_ids]
            coverage.append({'model':m,'cutoff':cutoff,'outcome_date':outcome,'forecast_assets':len(all_ids),
                             'risk_unavailable':int((~b['valid']).sum()),'common_forecast_assets':len(ids),
                             'missing_common_outcomes':int((mask & ~np.isfinite(b['r'])).sum()),
                             'paired_individual_assets':int(valid_outcomes.sum())})
            for axis,g,local_mask in groups:
                ix = common_ix[local_mask]
                port = portfolio_pair(b['X'][ix],b['F'],b['d'][ix],b['r'][ix],b['u'][ix])
                portfolios.append({'model':m,'cutoff':cutoff,'outcome_date':outcome,'axis':axis,'group':g,**port})
                usable = ix[valid_outcomes[ix]]
                z = b['u'][usable]/np.sqrt(b['d'][usable])
                # Moments retained for pooled descriptive summaries, not independent inference.
                individual.append({'model':m,'outcome_date':outcome,'axis':axis,'group':g,
                    'n':len(z),'sum':float(z.sum()),'sum_sq':float(z@z),'tail_count':int((np.abs(z)>3).sum())})
                if axis == 'liquidity_decile' or axis == 'overall':
                    shadow_d = np.array([shadow[m][all_ids[i]] if shadow[m][all_ids[i]] is not None else np.nan for i in ix])
                    if not np.isfinite(shadow_d).all() or np.any(shadow_d<=0):
                        raise ValueError('CALIBRATION_SHADOW_SUPPORT')
                    ratios = np.sqrt(b['d'][ix]/shadow_d)
                    shadow_rows.append({'model':m,'cutoff':cutoff,'axis':axis,'group':g,'assets':len(ix),
                                        'median_sigma_ratio':float(np.median(ratios)) if len(ix) else None,
                                        'mean_sigma_ratio':float(np.mean(ratios)) if len(ix) else None})
        frame = pl.DataFrame(pair).with_columns(pl.selectors.float().fill_nan(None))
        frame.write_parquet(out/'pairs'/(cutoff+'.parquet'),compression='zstd')
        if (day_index+1)%100 == 0:
            print(f'core6 calibration: {day_index+1}/{len(axes)} paired dates',flush=True)
    port_frame = pl.DataFrame(portfolios)
    port_frame.write_parquet(out/'portfolio_pairs.parquet',compression='zstd')
    pl.DataFrame(coverage).write_parquet(out/'coverage.parquet',compression='zstd')
    pl.DataFrame(individual).write_parquet(out/'individual_moments.parquet',compression='zstd')
    pl.DataFrame(shadow_rows).write_parquet(out/'winsor_shadow_daily.parquet',compression='zstd')
    summary, rolling, pooled = [], [], []
    keys = [(axis,g) for axis,g,_ in groups]
    for m in MODELS:
        for axis,g in keys:
            rows = [r for r in portfolios if r['model']==m and r['axis']==axis and r['group']==g]
            for kind in ('total','specific'):
                z = [r[kind+'_z'] for r in rows if r['status']=='paired']
                summary.append({'model':m,'axis':axis,'group':g,'kind':kind,'scheduled_dates':len(rows),
                                'unavailable_dates':len(rows)-len(z),**standardized_summary(z)})
                for end in range(249,len(rows)):
                    window = rows[end-249:end+1]
                    zs = [r[kind+'_z'] for r in window if r['status']=='paired']
                    rolling.append({'model':m,'axis':axis,'group':g,'kind':kind,
                        'window_start':window[0]['outcome_date'],'window_end':window[-1]['outcome_date'],
                        **standardized_summary(zs,minimum=250)})
            moments = [r for r in individual if r['model']==m and r['axis']==axis and r['group']==g]
            n = sum(r['n'] for r in moments); a = sum(r['sum'] for r in moments); q = sum(r['sum_sq'] for r in moments)
            pooled.append({'model':m,'axis':axis,'group':g,'asset_days':n,
                'std':math.sqrt(max(0.,(q-a*a/n)/(n-1))) if n>1 else None,
                'mean':a/n if n else None,'rms':math.sqrt(q/n) if n else None,
                'abs_gt_3_fraction':sum(r['tail_count'] for r in moments)/n if n else None,
                'interpretation':'descriptive_correlated_asset_days_not_independent_samples'})
    pl.DataFrame(rolling).write_parquet(out/'rolling_250_sessions.parquet',compression='zstd')
    _write_json(out/'summary.json',{'portfolio_statistics':summary,'individual_specific_statistics':pooled,
                'outcome_identity_max_error':max_identity_error,'forecast_dates':len(axes),
                'release_gate_status':'blocked_pending_contract_and_evidence',
                'full_residual_permutation_gate':'not_run', 'historical_pit_verified':False})
    outputs = {str(p.relative_to(out)):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
               for p in out.rglob('*') if p.is_file()}
    manifest = {'schema_version':1,'run_id':run_id,'identity':identity,'created_at':utc_now().isoformat(),
        'status':'completed_reconstructed_calibration_diagnostic','parent_run_ids':[rm['run_id'],fm['run_id']],
        'temporal_mode':'reconstructed_diagnostic_only','historical_pit_verified':False,
        'production_eligible':False,'holdout_evaluated':False,'current_changed':False,
        'release_gate_status':'blocked_pending_contract_and_evidence','outputs':outputs}
    DataLake(output_root).write_immutable_json(manifest_path,manifest)
    return manifest_path
