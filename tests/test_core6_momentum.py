"""Fixed artificial returns; no real market labels or production registration."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path

import polars as pl
import pytest

from factor_matrix.calculation.l1.momentum import (
    MomentumDefinition, load_core6_design, momentum_descriptor,
)
from factor_matrix.factor_engine.registry import FactorRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(core6_design_project):
    # An artificial exchange calendar with weekends omitted; not real行情.
    days = [date(2001, 1, 1) + timedelta(days=i) for i in range(500)]
    days = [d for d in days if d.weekday() < 5][:280]
    rows = [{"asset_id": "FIXTURE", "trade_date": d, "total_return": .001,
             "return_source": "synthetic_total_return",
             "available_at": datetime.combine(d, datetime.min.time(), timezone.utc) + timedelta(hours=8),
             "availability_evidence": "observed_first_seen"} for d in days]
    _, definition = load_core6_design(core6_design_project / 'config/risk_core6_design_v2.json', project_root=core6_design_project)
    return {"as_of": days[-1], "decision_time": rows[-1]['available_at'], "calendar": days,
            "assets": pl.DataFrame({"asset_id": ["FIXTURE"]}), "returns": rows, "definition": definition}


def run(sample):
    args = {**sample, "returns": pl.DataFrame(sample['returns'], schema_overrides={
        "available_at": pl.Datetime('us', 'UTC'), "trade_date": pl.Date, "total_return": pl.Float64})}
    return momentum_descriptor(**args).to_dicts()


def test_endpoints_skip_recent_and_no_older_backfill(sample):
    n = len(sample['calendar'])
    for lag in (0, 20, 252, 279):
        sample['returns'][n-1-lag]['total_return'] = 5.
    for lag, value in ((21, .1), (251, -.1)):
        sample['returns'][n-1-lag]['total_return'] = value
    row = run(sample)[0]
    assert row['momentum_raw'] == pytest.approx(1.1 * .9 * 1.001**229 - 1)
    assert row['n_momentum_obs'] == 231
    assert row['window_start'] == sample['calendar'][-252]
    assert row['window_end'] == sample['calendar'][-22]
    assert not row['raw_missing'] and not row['partial_observed_product']


@pytest.mark.parametrize('missing,valid', [(31, True), (32, False)])
def test_observation_boundary_and_partial_product(sample, missing, valid):
    start = len(sample['calendar']) - 252
    for row in sample['returns'][start:start+missing]:
        row['total_return'] = None
    result = run(sample)[0]
    assert result['n_momentum_obs'] == 231-missing
    assert result['n_null_return'] == missing
    if valid:
        assert result['momentum_raw'] == pytest.approx(1.001**200-1)
        assert result['partial_observed_product']
    else:
        assert result['momentum_raw'] is None
        assert result['unavailable_reason'] == 'insufficient_observations'


def test_future_and_unavailable_values_cannot_change_result(sample):
    baseline = run(sample)[0]
    future_revision = deepcopy(sample['returns'][50])
    future_revision['available_at'] = sample['decision_time'] + timedelta(days=1)
    future_revision['total_return'] = math.nan
    future_revision['return_source'] = None
    future_revision['availability_evidence'] = 'unverified'
    sample['returns'].append(future_revision)
    future_event = deepcopy(future_revision)
    future_event['trade_date'] = sample['as_of'] + timedelta(days=1)
    sample['returns'].append(future_event)
    result = run(sample)[0]
    assert result['momentum_raw'] == baseline['momentum_raw']
    assert result['n_momentum_obs'] == 231 and result['n_unavailable'] == 1


def test_unknown_time_and_reconstruction_are_not_historical_evidence(sample):
    sample['returns'][50]['available_at'] = None
    sample['returns'][51]['availability_evidence'] = 'assumed_18h'
    sample['returns'][52]['availability_evidence'] = 'conservative_base_first_seen'
    row = run(sample)[0]
    assert row['n_momentum_obs'] == 229
    assert row['n_unknown_availability'] == 2
    assert row['n_conservative_availability'] == 1
    assert row['source_available_at'] <= sample['decision_time']


def test_resumption_is_excluded_not_zero_filled(sample):
    sample['returns'][50]['return_source'] = 'resumption'
    sample['returns'][50]['total_return'] = 9.
    result = run(sample)[0]
    assert result['momentum_raw'] == pytest.approx(1.001**230-1)
    assert result['n_resumption_excluded'] == 1
    assert result['partial_observed_product']


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1.01])
def test_bad_visible_return_does_not_silently_drop(sample, value):
    sample['returns'][50]['total_return'] = value
    row = run(sample)[0]
    assert row['momentum_raw'] is None
    assert row['n_invalid_return'] == 1 and row['unavailable_reason'] == 'invalid_return'


def test_total_loss_and_nonfinite_compounding(sample):
    sample['returns'][50]['total_return'] = -1.
    assert run(sample)[0]['momentum_raw'] == -1.
    for row in sample['returns']:
        row['total_return'] = 1e10
    assert run(sample)[0]['unavailable_reason'] == 'nonfinite_product'


def test_common_calendar_and_absent_assets(sample):
    sample['assets'] = pl.DataFrame({'asset_id': ['MISSING', 'FIXTURE']})
    result = run(sample)
    assert [r['asset_id'] for r in result] == ['MISSING', 'FIXTURE']
    assert result[0]['momentum_raw'] is None and result[0]['n_momentum_obs'] == 0
    # Even with 209 usable returns a truncated common window is unavailable.
    sample['calendar'] = sample['calendar'][-230:]
    row = run(sample)[1]
    assert row['n_momentum_obs'] == 209
    assert row['unavailable_reason'] == 'calendar_history_insufficient'


@pytest.mark.parametrize('change,reason', [
    ('duplicate', 'REVISION_RESOLUTION'), ('calendar_duplicate', 'CALENDAR_INVALID'),
    ('missing_asof', 'ASOF_NOT_IN_CALENDAR'), ('naive', 'TIMEZONE'),
    ('early_availability', 'AVAILABILITY_PRECEDES_EVENT'), ('future_asof', 'ASOF_IN_FUTURE'),
    ('asset_duplicate', 'ASSET_AXIS'),
])
def test_ambiguous_inputs_rejected(sample, change, reason):
    if change == 'duplicate': sample['returns'].append(deepcopy(sample['returns'][50]))
    if change == 'calendar_duplicate': sample['calendar'].append(sample['calendar'][0])
    if change == 'missing_asof': sample['calendar'].pop()
    if change == 'naive': sample['decision_time'] = sample['decision_time'].replace(tzinfo=None)
    if change == 'early_availability': sample['returns'][50]['available_at'] -= timedelta(days=2)
    if change == 'future_asof': sample['decision_time'] -= timedelta(days=1)
    if change == 'asset_duplicate': sample['assets'] = pl.DataFrame({'asset_id': ['FIXTURE', 'FIXTURE']})
    with pytest.raises(ValueError, match=reason): run(sample)


def test_row_and_calendar_order_do_not_change_value(sample):
    original = run(sample)
    sample['returns'].reverse()
    sample['calendar'].reverse()
    assert run(sample) == original


def test_config_and_legacy_discovery_isolation(tmp_path, core6_design_project):
    path = core6_design_project/'config/risk_core6_design_v2.json'
    spec, definition = load_core6_design(path, project_root=core6_design_project)
    assert definition == MomentumDefinition(21, 251, 200)
    assert len(spec['expanded_columns']) == 38
    assert 'momentum' not in FactorRegistry.discover().factor_ids()
    assert 'listing_age' in FactorRegistry.discover().factor_ids()
    for mutation, reason in [('active', 'INACTIVE'), ('axis', 'AXIS'), ('policy', 'POLICY'), ('hash', 'EVIDENCE')]:
        changed = deepcopy(spec)
        if mutation == 'active': changed['active'] = True
        if mutation == 'axis': changed['expanded_columns'].pop()
        if mutation == 'policy': changed['momentum']['missing_policy'] = 'zero_fill'
        if mutation == 'hash': changed['inherited_file_hashes']['config/risk_factor_set_candidate_v1.json'] = 'wrong'
        copy = tmp_path/'candidate.json'
        copy.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=reason): load_core6_design(copy, project_root=core6_design_project)


@pytest.mark.local_data
def test_approved_core6_design_matches_local_audit_evidence():
    # The portable synthetic fixture does not attest the real approval bundle.
    spec, definition = load_core6_design(ROOT/'config/risk_core6_design_v2.json', project_root=ROOT)
    assert spec['active'] is False
    assert definition == MomentumDefinition(21, 251, 200)
