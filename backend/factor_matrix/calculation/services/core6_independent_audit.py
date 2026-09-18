"""Read-only audit of fixed reconstructed inputs; no refit or gate promotion.

The G2 arithmetic deliberately does not call the L1 builder, projection or
quality helpers. Residual second moments are descriptive identities, not a
correlation test or a proposed covariance estimator.
"""
from datetime import date, datetime
from pathlib import Path
import json
import sqlite3

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, utc_now
from ...revisioned_silver import SilverAsOfReader
from .core6_calibration import _verified_parent, _read_frame


def metric_statistics(values, controls, weights):
    """Independent centered weighted moments, including nonconstant controls."""
    y, c, w = [np.asarray(v, dtype=float) for v in (values, controls, weights)]
    if y.ndim != 1 or c.ndim != 2 or c.shape[0] != len(y) or w.shape != y.shape:
        raise ValueError('AUDIT_METRIC_AXIS')
    if not len(y) or not all(np.isfinite(v).all() for v in (y, c, w)) or np.any(w <= 0):
        raise ValueError('AUDIT_METRIC_NONFINITE_OR_WEIGHT')
    w = w / w.sum()
    ym, cm = float(w @ y), w @ c
    yc, cc = y - ym, c - cm
    sy = float(np.sqrt(w @ (yc * yc)))
    sc = np.sqrt(w @ (cc * cc))
    if sy <= 0:
        raise ValueError('AUDIT_CONSTANT_STYLE')
    usable = sc > 1e-14
    corr = (w * yc) @ cc[:, usable] / (sy * sc[usable])
    return {'mean_w': ym, 'sd_w': sy,
            'max_abs_control_corr': float(np.max(np.abs(corr), initial=0.)),
            'constant_controls': int((~usable).sum())}


def residual_moments(specific, delta):
    """Exact z² = diagonal realized term + signed cross-product term.

Missing outcomes invalidate the whole ex-ante portfolio. Cross-products are
uncentered and must not be labeled covariance or unexplained-factor evidence.
"""
    u, d = np.asarray(specific, dtype=float), np.asarray(delta, dtype=float)
    if u.ndim != 1 or u.shape != d.shape or not len(u):
        raise ValueError('AUDIT_RESIDUAL_AXIS')
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError('AUDIT_INVALID_FORECAST')
    result = {'status': 'missing_outcome', 'assets': len(u), 'z': None,
              'z_squared': None, 'diagonal_second_moment': None,
              'cross_second_moment': None, 'identity_error': None}
    if not np.isfinite(u).all():
        return result
    # n^-2 cancels from numerator and denominator.
    diagonal = float(u @ u / d.sum())
    z = float(u.sum() / np.sqrt(d.sum()))
    cross = float((u.sum()**2 - u @ u) / d.sum())
    result.update(status='paired', z=z, z_squared=z*z,
                  diagonal_second_moment=diagonal, cross_second_moment=cross,
                  identity_error=abs(z*z-diagonal-cross))
    return result


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False)+'\n')


def _registry_snapshot(root):
    path = root/'data/metadata/factor_registry.sqlite'
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        c.execute('BEGIN')
        return {table: [dict(row) for row in c.execute(f'SELECT * FROM {table} ORDER BY rowid')]
                for table in ('family_root_declaration', 'research_attempt', 'alpha_assertion',
                              'probe_registry', 'feature_assignment', 'feature_registry')}


def _role_and_covariance_audit(root, registry):
    destination = json.loads((root/'config/g4_statistical_factor_destination_v1.json').read_text())
    source = destination['rank_decision']
    path = root/source['source_manifest']
    if file_sha256(path) != source['source_manifest_sha256']:
        raise ValueError('AUDIT_SPECTRUM_HASH')
    spectrum = json.loads(path.read_text())
    artifacts = {}
    for name, rec in spectrum['outputs'].items():
        target = root/'data'/rec['path']
        artifacts[name] = {'path': str(target.relative_to(root)),
                           'hash_matches': target.is_file() and file_sha256(target) == rec['sha256']}
    g3path = root/'data/gold/l2b_risk_only'/f"run_id={spectrum['g3_run_id']}"/'_MANIFEST.json'
    g3 = json.loads(g3path.read_text()) if g3path.exists() else {}
    return {
        'registered_momentum_declarations': [r for r in registry['family_root_declaration'] if r['family_root_id'] == 'momentum'],
        'candidate_momentum_role': 'basis', 'role_migration_status': 'unresolved_family_scope_and_versioned_storage',
        'formal_alpha_assertion_rows': len(registry['alpha_assertion']),
        'formal_research_attempt_rows': len(registry['research_attempt']),
        'registered_probes': registry['probe_registry'],
        'historical_semantic_probe_cost': {'numeric_attempt_count': None, 'status': 'incomplete_not_zero',
            'evidence': ['config/alpha_dimension_budget_v1.json',
                         'config/g4_statistical_loading_semantic_probe_v1.json',
                         'config/g4_style_loading_semantic_probe_v1.json',
                         'data/diagnostics/weight_metric_migration/legacy_cleanup_v1.json'],
            'disposition': 'preserve_superseded_and_unavailable_evidence_no_fabricated_FDR_attempts'},
        'low_rank_covariance': {
            'required_by_legacy_architecture': True, 'implemented_in_fixed_core6_or_old44_rolling_run': False,
            'legacy_selected_J': spectrum['headline']['decision_factor_count'],
            'source_manifest': str(path.relative_to(root)), 'source_sha256': file_sha256(path),
            'source_output_checks': artifacts,
            'g3_parent_hash_matches': g3path.is_file() and file_sha256(g3path) == spectrum['identity']['input_hashes']['g3_manifest'],
            'legacy_basis_id': g3.get('risk_basis_id'), 'legacy_silver_version_id': g3.get('silver_version_id'),
            'legacy_decision_sample_end': spectrum['protocol']['decision_sample_end_inclusive'],
            'applicable_to_reconstructed_basis': False,
            'reason': 'different_reestimated_residual_chain_and_global_sample_rank_not_cutoff_selected',
            'disposition': 'draft_separate_estimator_amendment_before_any_low_rank_refit_no_J_increase_to_pass_bias'}}


def _event_evidence(root, config, recon_dir):
    """Read the pinned complete Silver relation, not just extracted parent rows."""
    lake = DataLake(root/'data')
    reader = SilverAsOfReader(lake, version_id=config['silver_version_id'])
    version = reader.pinned_version
    records = {str(p.relative_to(root)): file_sha256(p) for p in (
        lake.metadata/'silver_versions'/f"{config['silver_version_id']}.json",
        lake.metadata/'base_manifests'/f"{version['base_id']}.json")}
    frames = {}
    tables = ('security_master', 'prices_daily', 'returns_daily',
              'security_daily_state', 'stock_st_daily', 'suspensions_daily')
    ids = [r['asset_id'] for r in config['missing_events']]
    knowledge = datetime.fromisoformat(config['knowledge_time'])
    conn = open_duckdb()
    try:
        for table in tables:
            base = reader._base_path(table)
            expected = reader.base['artifacts'].get(table)
            digest = file_sha256(base)
            if expected and digest != expected['sha256']:
                raise ValueError('AUDIT_BASE_HASH:'+table)
            records[str(base.relative_to(root))] = digest
            for item in version['changes']:
                if item['table'] == table:
                    path = lake.root/item['path']
                    digest = file_sha256(path)
                    if digest != item['sha256']:
                        raise ValueError('AUDIT_DELTA_HASH:'+table)
                    records[str(path.relative_to(root))] = digest
            relation = reader.relation_sql(table, knowledge)
            placeholders = ','.join('?' for _ in ids)
            # A bounded event audit entirely before the sealed Holdout.
            bounds = '' if table == 'security_master' else (
                " AND trade_date BETWEEN DATE '2024-06-20' AND DATE '2024-07-31'")
            frames[table] = conn.execute(
                f'SELECT * FROM ({relation}) WHERE asset_id IN ({placeholders}){bounds}', ids).pl()
    finally:
        conn.close()
    master = {r['asset_id']: r for r in frames['security_master'].to_dicts()}
    parent = {t: pl.scan_parquet(recon_dir/f'inputs/{t}.parquet').filter(
        pl.col('asset_id').is_in(ids)).collect()
        for t in ('prices_daily', 'returns_daily', 'security_daily_state')}
    events = []
    for item in config['missing_events']:
        a, t, cutoff = item['asset_id'], date.fromisoformat(item['outcome_date']), date.fromisoformat(item['cutoff'])
        selected = lambda f, day: f.filter((pl.col('asset_id') == a) & (pl.col('trade_date') == day))
        m = master.get(a, {})
        events.append({**item, 'master_record': m,
            'delist_date_matches_missing_outcome': m.get('delist_date') == t,
            'complete_silver_outcome_counts': {k: selected(v, t).height for k, v in frames.items() if k != 'security_master'},
            'parent_outcome_counts': {k: selected(v, t).height for k, v in parent.items()},
            'cutoff_state': selected(frames['security_daily_state'], cutoff).to_dicts(),
            'cutoff_price': selected(frames['prices_daily'], cutoff).to_dicts(),
            'outcome_st_records': selected(frames['stock_st_daily'], t).to_dicts(),
            'cutoff_st_records': selected(frames['stock_st_daily'], cutoff).to_dicts(),
            'terminal_value_status': 'unavailable_no_zero_fill_or_hindsight_deletion',
            'evidence_scope': 'local_vendor_record_observed_2026_not_historical_PIT_or_exchange_confirmation'})
    return {'source_records': records, 'events': events}, frames


def run_independent_audit(*, project_root: Path, config_path: Path, config_sha256: str, output_root: Path):
    if file_sha256(config_path) != config_sha256:
        raise ValueError('AUDIT_CONFIG_HASH')
    cfg = json.loads(config_path.read_text())
    if cfg['protocol_id'] != 'core6_independent_audit_v1' or cfg['holdout_start'] != '2025-03-01':
        raise ValueError('AUDIT_PROTOCOL')
    for rel, digest in cfg['bindings'].items():
        if file_sha256(project_root/rel) != digest:
            raise ValueError('AUDIT_BINDING:'+rel)
    rd, rm = _verified_parent(project_root, cfg['parents']['reconstruction'])
    cd, cm = _verified_parent(project_root, cfg['parents']['calibration'])
    days = json.loads((rd/'daily_status.json').read_text())
    if len(days) != 504 or days[0]['return_date'] != '2023-02-01' or days[-1]['return_date'] != '2025-02-28':
        raise ValueError('AUDIT_FIXED_WINDOW')
    if any(d['status'] != 'passed' or d['return_date'] >= cfg['holdout_start'] for d in days):
        raise ValueError('AUDIT_PARENT_DAY_STATUS')
    contract = json.loads((project_root/'config/weight_metric_contract_v1.json').read_text())['regression_base_weight']
    wp = project_root/'data'/contract['artifact_path']
    if file_sha256(wp) != contract['artifact_sha256']:
        raise ValueError('AUDIT_ORIGINAL_METRIC_HASH')
    weight_frame = pl.read_parquet(wp).select('exposure_date', 'asset_id', 'candidate_weight')
    if weight_frame.unique(['exposure_date', 'asset_id']).height != weight_frame.height:
        raise ValueError('AUDIT_DUPLICATE_METRIC_KEY')
    weights = {key[0]: dict(frame.select('asset_id', 'candidate_weight').iter_rows())
               for key, frame in weight_frame.partition_by('exposure_date', as_dict=True).items()}
    registry = _registry_snapshot(project_root)
    role_audit = _role_and_covariance_audit(project_root, registry)
    event_evidence, event_frames = _event_evidence(project_root, cfg, rd)
    identity = {'config_sha256': config_sha256, 'code_sha256': file_sha256(Path(__file__)),
                'reader_sha256': file_sha256(project_root/'backend/factor_matrix/revisioned_silver.py'),
                'parent_verifier_sha256': file_sha256(project_root/'backend/factor_matrix/calculation/services/core6_calibration.py'),
                'registry_snapshot_sha256': json_hash(registry), 'role_audit_sha256': json_hash(role_audit),
                'event_sources': event_evidence['source_records']}
    run_id = 'core6_independent_'+json_hash(identity)[:16]
    directory = output_root/f'run_id={run_id}'
    mp = directory/'_MANIFEST.json'
    if mp.exists():
        for rel, record in json.loads(mp.read_text())['outputs'].items():
            if file_sha256(directory/rel) != record['sha256']:
                raise ValueError('AUDIT_EXISTING_OUTPUT_CHANGED')
        return mp
    directory.mkdir(parents=True, exist_ok=True)
    metric_rows, fit_rows, residual_rows = [], [], []
    for day in days:
        t, e = day['return_date'], date.fromisoformat(day['exposure_date'])
        pair_name = f'pairs/{e}.parquet'
        pair = _read_frame(cd, cm, pair_name) if pair_name in cm['outputs'] else None
        for model in ('core6', 'old44'):
            x = _read_frame(rd, rm, f'daily/{t}/{model}_X.parquet').sort('asset_id')
            ids = x['asset_id'].to_list()
            w = np.array([weights.get(e, {}).get(a, np.sqrt(cap)) for a, cap in zip(ids, x['float_mkt_cap'])])
            actual = x['candidate_weight'].to_numpy()
            error = float(np.max(np.abs(actual/w-1)))
            industry = [c for c in x.columns if c.startswith('risk_industry_')]
            for style, predecessors in cfg['predecessors'][model].items():
                cols = industry + ['risk_'+c for c in predecessors]
                stats = metric_statistics(x['risk_'+style].to_numpy(), x.select(cols).to_numpy(), w)
                passed = (abs(stats['mean_w']) <= 1e-8 and abs(stats['sd_w']-1) <= 1e-8
                          and stats['max_abs_control_corr'] <= 1e-8 and error <= 1e-12)
                metric_rows.append({'model': model, 'exposure_date': e, 'return_date': t, 'style': style,
                    'assets': len(ids), 'metric_fallback_assets': sum(a not in weights.get(e, {}) for a in ids),
                    'weight_relative_error': error, **stats, 'passed': passed})
            u = _read_frame(rd, rm, f'daily/{t}/{model}_specific.parquet')
            joined = x.select('asset_id', 'candidate_weight').join(u, on='asset_id', how='left', validate='1:1')
            fit = joined.filter(pl.col('in_estimation_domain').fill_null(False))
            raw = fit['specific_return'].to_numpy()
            base = fit['candidate_weight'].to_numpy()
            eff = fit['effective_weight'].to_numpy()
            mult = fit['huber_multiplier'].to_numpy()
            if not len(raw) or not all(np.isfinite(v).all() for v in (raw, base, eff, mult)):
                raise ValueError('AUDIT_FIT_NONFINITE')
            fit_rows.append({'model': model, 'return_date': t, 'assets': len(raw),
                'ew_mean_estimation': float(raw.mean()), 'base_w_mean_estimation': float(base @ raw/base.sum()),
                'effective_w_mean_estimation': float(eff @ raw/eff.sum()),
                'effective_weight_relative_error': float(np.max(np.abs(eff/(base*mult)-1)))})
            if pair is not None:
                if pair['outcome_date'].n_unique() != 1 or str(pair['outcome_date'][0]) != t:
                    raise ValueError('AUDIT_PAIR_DATE_ALIGNMENT')
                p = pair.filter('common_forecast_support')
                moment = residual_moments(p[model+'_specific_return'].to_numpy(), p[model+'_Delta'].to_numpy())
                residual_rows.append({'model': model, 'cutoff': e, 'outcome_date': t, **moment})
    metrics, fits, residuals = map(pl.DataFrame, (metric_rows, fit_rows, residual_rows))
    summary = {'audit_scope': 'independent_G2_metric_subset_and_descriptive_residual_identities',
        'g2_full_acceptance_status': 'not_issued', 'g6a_status': 'unknown_pending_role_cost_lineage',
        'historical_pit_verified': False, 'production_eligible': False, 'holdout_evaluated': False,
        'current_changed': False, 'models': {}}
    for model in ('core6', 'old44'):
        mm, ff = metrics.filter(pl.col('model') == model), fits.filter(pl.col('model') == model)
        rr = residuals.filter((pl.col('model') == model) & (pl.col('status') == 'paired'))
        z = rr['z'].to_numpy()
        summary['models'][model] = {
            'metric_checks': mm.height, 'metric_failed': mm.filter(~pl.col('passed')).height,
            'max_abs_mean_w': float(mm['mean_w'].abs().max()), 'max_abs_sd_error': float((mm['sd_w']-1).abs().max()),
            'max_abs_control_corr': mm['max_abs_control_corr'].max(), 'max_weight_relative_error': mm['weight_relative_error'].max(),
            'fit_days': ff.height, 'mean_ew_residual_bps': float(ff['ew_mean_estimation'].mean()*1e4),
            'mean_base_w_residual_bps': float(ff['base_w_mean_estimation'].mean()*1e4),
            'max_abs_effective_w_residual': ff['effective_w_mean_estimation'].abs().max(),
            'max_effective_weight_relative_error': ff['effective_weight_relative_error'].max(),
            'paired_portfolio_days': len(z), 'missing_portfolio_days': residuals.filter(pl.col('model') == model).height-len(z),
            'specific_z_mean': float(z.mean()), 'specific_z_std_ddof1': float(z.std(ddof=1)),
            'specific_z_second_moment': float(np.mean(z*z)), 'specific_z_mean_squared': float(z.mean()**2),
            'specific_z_population_variance': float(np.var(z)),
            'mean_diagonal_second_moment': rr['diagonal_second_moment'].mean(),
            'mean_signed_cross_second_moment': rr['cross_second_moment'].mean(),
            'max_second_moment_identity_error': rr['identity_error'].max()}
    for name, frame in {'g2_metric_checks': metrics, 'fit_weight_audit': fits, 'residual_identities': residuals,
                         **{'event_'+k: v for k, v in event_frames.items()}}.items():
        frame.write_parquet(directory/(name+'.parquet'), compression='zstd')
    for name, value in {'summary': summary, 'event_evidence': event_evidence,
                         'registry_snapshot': registry, 'role_and_covariance_audit': role_audit}.items():
        _json(directory/(name+'.json'), value)
    outputs = {p.name: {'sha256': file_sha256(p)} for p in sorted(directory.iterdir()) if p.is_file()}
    manifest = {'schema_version': 1, 'run_id': run_id, 'identity': identity, 'outputs': outputs,
        'created_at': utc_now().isoformat(), 'status': 'completed_diagnostic_only',
        'parents': cfg['parents'], 'temporal_mode': 'reconstructed_diagnostic_only',
        'production_eligible': False, 'holdout_evaluated': False, 'current_changed': False}
    DataLake(project_root/'data').write_immutable_json(mp, manifest)
    return mp
