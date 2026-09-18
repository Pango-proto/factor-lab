"""Real-market input adapter with a separate, checksum-bound data gate."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from ...revisioned_silver import SilverAsOfReader, SilverVersionLedger
from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash
from ..publishing.local import _write_immutable_parquet
from .backtest_path import capacity_diagnostics
from .strategy_fixture import plain


def normalize_backtest_bars(prices: pl.DataFrame, state: pl.DataFrame, *, adv_window: int = 20) -> pl.DataFrame:
    if type(adv_window) is not int or adv_window < 1:
        raise ValueError('INVALID_ADV_WINDOW')
    keys=['trade_date','asset_id']
    for frame in (prices,state):
        if frame.height != frame.unique(keys).height:
            raise ValueError('BACKTEST_MARKET_DUPLICATE_KEYS')
    bars=state.join(prices,on=keys,how='left').sort(['asset_id','trade_date'])
    # Existing Silver volume is Tushare regular-session lots, amount is CNY
    # thousands. Preserve originals, add explicit units; do not rewrite history.
    bars=bars.with_columns(
        (pl.col('volume')*100).alias('volume_shares'),
        (pl.col('amount')*1000).alias('amount_cny'),
    ).with_columns(
        pl.when(pl.col('volume_shares').is_null() & pl.col('is_suspended') & pl.col('source_day_complete'))
        .then(pl.lit(0.0)).otherwise(pl.col('volume_shares')).alias('liquidity_volume_shares')
    )
    if bars.filter(pl.col('volume_shares').is_not_null() & (~pl.col('volume_shares').is_finite() | (pl.col('volume_shares')<0))).height:
        raise ValueError('INVALID_MARKET_VOLUME')
    bars=bars.with_columns(
        pl.col('liquidity_volume_shares').rolling_mean(window_size=adv_window,min_samples=adv_window).shift(1).over('asset_id').alias('adv_shares'),
        pl.col('liquidity_volume_shares').is_not_null().cast(pl.Int64).rolling_sum(window_size=adv_window,min_samples=1).shift(1).over('asset_id').alias('adv_observations'),
        pl.col('trade_date').shift(1).over('asset_id').alias('adv_last_observation_date'),
    )
    return bars.with_columns(pl.lit('tushare_daily_regular_session_excludes_after_hours').alias('volume_scope'))


def checked_data_gate(lake: DataLake, gate_path: Path, *, expected_sha256: str) -> dict:
    if file_sha256(gate_path)!=expected_sha256:
        raise ValueError('BACKTEST_DATA_GATE_CHECKSUM_MISMATCH')
    gate=json.loads(gate_path.read_text())
    if gate.get('job')!='complete_data_refresh' or gate.get('status')!='passed' or gate.get('market_input_allowed') is not True:
        raise ValueError('BACKTEST_REAL_DATA_GATE_CLOSED')
    if SilverVersionLedger(lake).current()['version_id']!=gate['silver_version_id']:
        raise ValueError('BACKTEST_DATA_GATE_STALE_VERSION')
    for record in gate['outputs'].values():
        if file_sha256(lake.root/record['path'])!=record['sha256']:
            raise ValueError('BACKTEST_DATA_GATE_EVIDENCE_CHANGED')
    return gate


def materialize_backtest_market(lake: DataLake, *, gate_path: Path, gate_sha256: str, start: date, end: date) -> Path:
    gate=checked_data_gate(lake,gate_path,expected_sha256=gate_sha256)
    if end<start or end>date.fromisoformat(gate['data_as_of']):
        raise ValueError('BACKTEST_MARKET_RANGE_NOT_VALIDATED')
    reader=SilverAsOfReader(lake)
    observed_at=datetime.fromisoformat(gate['observed_at'])
    prices_sql=reader.relation_sql('prices_daily',observed_at)
    state_sql=reader.relation_sql('security_daily_state',observed_at)
    warmup=start-timedelta(days=90)
    connection=open_duckdb(temp_directory=lake.root/'tmp'/'duckdb')
    try:
        # Arrow avoids Python datetime timezone adapters; original timestamps
        # remain represented by the pinned parent version/observation timestamp.
        prices=pl.from_arrow(connection.execute(f"SELECT trade_date,asset_id,raw_open,raw_high,raw_low,raw_close,adj_factor,volume,amount,limit_up,limit_down FROM ({prices_sql}) WHERE trade_date BETWEEN DATE '{warmup}' AND DATE '{end}'").fetch_arrow_table())
        state=pl.from_arrow(connection.execute(f"SELECT trade_date,asset_id,board_id,is_suspended,is_st,source_day_complete,listed_as_of FROM ({state_sql}) WHERE trade_date BETWEEN DATE '{warmup}' AND DATE '{end}' AND in_a_share_scope").fetch_arrow_table())
    finally:
        connection.close()
    bars=normalize_backtest_bars(prices,state).filter(pl.col('trade_date').is_between(start,end))
    if bars.is_empty() or bars.filter(~pl.col('source_day_complete')).height:
        raise ValueError('BACKTEST_MARKET_SOURCE_INCOMPLETE')
    definition={'gate_sha256':gate_sha256,'start':str(start),'end':str(end),'adv_window':20,'code_hash':source_tree_hash()}
    run_id='backtest_market_v1_'+json_hash(definition)[:16]
    directory=lake.root/'gold'/'backtest_market'/f'run_id={run_id}'
    path=directory/'bars.parquet'
    _write_immutable_parquet(path,bars)
    checked_data_gate(lake,gate_path,expected_sha256=gate_sha256)
    return lake.write_immutable_json(directory/'_MANIFEST.json',{
        'schema_version':1,'run_id':run_id,'status':'market_input_ready','definition':definition,
        'purpose':'real_market_input_only','parent_run_ids':[gate['run_id']], 'parent_gate':lake.artifact_record(gate_path),
        'silver_version_id':gate['silver_version_id'],'outputs':{'bars':lake.artifact_record(path)},
        'rows':bars.height,'dates':bars['trade_date'].n_unique(),'assets':bars['asset_id'].n_unique(),
        'missing_adv_rows':bars['adv_shares'].null_count(),'raw_price_unit':'CNY_per_share','volume_unit':'shares',
        'volume_source_unit':'100_share_lots','amount_source_unit':'CNY_thousands',
        'corporate_action_warning':'Raw prices are for fills; adj_factor is not an executable price or a cash dividend ledger',
        'research_status':'not_eligible','real_total_cost_backtest_allowed':False,
        'holdout_research_evaluated':False,'risk_status':'unavailable',
    })


def profile_market_capacity(lake: DataLake, *, market_manifest: Path, expected_sha256: str) -> Path:
    """Profile liquidity of actual bars; no orders, alpha or holdout performance."""
    if file_sha256(market_manifest)!=expected_sha256:
        raise ValueError('CAPACITY_MARKET_MANIFEST_CHANGED')
    source=json.loads(market_manifest.read_text())
    if source.get('status')!='market_input_ready' or source.get('purpose')!='real_market_input_only':
        raise ValueError('CAPACITY_MARKET_NOT_READY')
    record=source['outputs']['bars'];path=lake.root/record['path']
    if file_sha256(path)!=record['sha256']:
        raise ValueError('CAPACITY_MARKET_TABLE_CHANGED')
    bars=pl.read_parquet(path)
    latest=bars['trade_date'].max()
    rows=[];unpriced=[]
    for r in bars.filter(pl.col('trade_date')==latest).to_dicts():
        if r['raw_close'] is None or r['raw_close']<=0:
            unpriced.append({'asset_id':r['asset_id'],'status':'no_reference_price'})
            continue
        observed=r['adv_last_observation_date']
        rows.append({'asset_id':r['asset_id'],'decision_time':f'{latest}T16:00:00+00:00',
            'requested_quantity':0,'reference_price':str(r['raw_close']),
            'adv_shares':str(r['adv_shares']) if r['adv_shares'] is not None else None,
            'adv_observations':r['adv_observations'] or 0,'required_observations':20,
            'adv_last_observation':f'{observed}T07:00:00+00:00' if observed else None,
            'adv_available_at':f'{observed}T16:00:00+00:00' if observed else None})
    # lot_size=1 expresses a SHARE liquidity bound, never an executable order
    # lot policy. Board-specific order rules remain a separate execution gate.
    report=plain(capacity_diagnostics(rows,participation='0.01',lot_size=1))
    report.update({'unpriced_assets':unpriced,'requested_quantity_policy':'zero: liquidity profiling, no order requested',
        'historical_availability':'Declared daily-bar delay for retrospective market diagnostics; not a live order fill guarantee',
        'quantity_interpretation':'share liquidity ceiling, not exchange-compliant order sizes'})
    definition={'source_sha256':expected_sha256,'as_of_date':str(latest),'participation':'0.01','code_hash':source_tree_hash()}
    run_id='market_capacity_v1_'+json_hash(definition)[:16]
    directory=lake.root/'diagnostics'/'market_capacity_v1'/f'run_id={run_id}'
    output=lake.write_immutable_json(directory/'capacity.json',report)
    return lake.write_immutable_json(directory/'_MANIFEST.json',{'schema_version':1,'run_id':run_id,
        'status':'diagnostic_complete','purpose':'real_market_liquidity_diagnostic','definition':definition,
        'parent_run_ids':[source['run_id']],'parent':lake.artifact_record(market_manifest),
        'outputs':{'capacity':lake.artifact_record(output)},'as_of_date':str(latest),
        'available_assets':sum(r['status']=='available' for r in report['rows']),
        'unavailable_assets':len(unpriced)+sum(r['status']!='available' for r in report['rows']),
        'research_status':'not_eligible','holdout_research_evaluated':False,'economic_capacity_cny':None})
