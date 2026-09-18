"""Pinned, read-only Silver extraction and provenance for the core-six candidate."""
from datetime import datetime, timedelta
from pathlib import Path
import json

import polars as pl

from ...revisioned_silver import SilverAsOfReader, SilverVersionLedger
from ...storage import DataLake, file_sha256, open_duckdb
from ...industry_observation import _interval_rows


TABLES = ('trade_calendar', 'returns_daily', 'valuation_daily', 'security_daily_state',
          'prices_daily', 'industry_membership_history', 'benchmark_membership_history')


def industry_intervals(history):
    """Preserve the existing gap rule, including both adjacent evidence times."""
    intervals = _interval_rows(history, 'SW2021')
    source = {(r['asset_id'], r['in_date']): r for r in history.filter(
        pl.col('classification_standard') == 'SW2021').iter_rows(named=True)}
    by_end = {(r['asset_id'], r['out_date']): r for r in source.values() if r['out_date']}
    times = []
    for row in intervals.iter_rows(named=True):
        if row['industry_observation_state'] == 'OBSERVED':
            records = [source[row['asset_id'], row['effective_from']]]
        else:
            records = [by_end[row['asset_id'], row['effective_from']-timedelta(days=1)],
                       source[row['asset_id'], row['effective_to']+timedelta(days=1)]]
        times.append(max(r['known_at'] for r in records))
    return intervals.with_columns(pl.Series('source_available_at', times))


def derive_board_benchmarks(connection, start, end):
    """Prior observed cap, with the SAME usable rows in numerator/denominator."""
    return connection.execute(f"""
      WITH obs AS (
        SELECT r.trade_date,r.asset_id,r.total_return,s.board_id,
          greatest(r.known_at,s.known_at,v.known_at) source_available_at,
          v.float_mkt_cap prior_float_mkt_cap
        FROM returns_daily r JOIN security_daily_state s USING(trade_date,asset_id)
        ASOF LEFT JOIN (SELECT * FROM valuation_daily WHERE float_mkt_cap IS NOT NULL) v
          ON r.asset_id=v.asset_id AND r.trade_date>v.trade_date
        WHERE s.in_a_share_scope AND s.listed_as_of
      ), flags AS (
        SELECT *, total_return IS NOT NULL AND isfinite(total_return)
          AND prior_float_mkt_cap>0 AND isfinite(prior_float_mkt_cap) usable FROM obs
      ) SELECT trade_date,board_id,
        sum(prior_float_mkt_cap*total_return) FILTER(WHERE usable)
          /nullif(sum(prior_float_mkt_cap) FILTER(WHERE usable),0) board_return,
        count(*) FILTER(WHERE usable) constituent_count,
        count(*) FILTER(WHERE NOT coalesce(usable,false)) excluded_observations,
        sum(prior_float_mkt_cap) FILTER(WHERE usable) effective_weight_base,
        max(source_available_at) source_available_at
      FROM flags WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
      GROUP BY 1,2 ORDER BY 1,2
    """).pl()


def load_inputs(*, lake: DataLake, gate_path: Path, gate_sha256: str, spec: dict, historical_window=None):
    if file_sha256(gate_path) != gate_sha256:
        raise ValueError('CORE6_DATA_GATE_HASH')
    gate = json.loads(gate_path.read_text())
    if gate['status'] != 'passed' or not gate['market_input_allowed']:
        raise ValueError('CORE6_DATA_GATE_NOT_PASSED')
    version = SilverVersionLedger(lake).current()
    if version is None or version['version_id'] != gate['silver_version_id']:
        raise ValueError('CORE6_PINNED_SILVER_CHANGED')
    knowledge_time = datetime.fromisoformat(gate['observed_at'])
    reader = SilverAsOfReader(lake)
    source_records = {}
    for table in TABLES:
        path = reader._base_path(table)
        record = reader.base['artifacts'].get(table)
        digest = file_sha256(path)
        if record and digest != record['sha256']:
            raise ValueError('CORE6_BASE_HASH_CHANGED:' + table)
        source_records[str(path)] = digest
        for item in version['changes']:
            if item['table'] == table:
                path = lake.root / item['path']
                digest = file_sha256(path)
                if digest != item['sha256']:
                    raise ValueError('CORE6_DELTA_HASH_CHANGED:' + item['path'])
                source_records[str(path)] = digest
    connection = open_duckdb()
    try:
        for table in TABLES:
            query = reader.relation_sql(table, knowledge_time)
            columns = [r[0] for r in connection.execute(f'DESCRIBE SELECT * FROM ({query})').fetchall()]
            time_cols = [c for c in ('first_seen_at', 'revision_at', 'ingested_at', 'available_at') if c in columns]
            connection.execute(f'CREATE TEMP VIEW {table} AS SELECT *, greatest({",".join(time_cols)}) known_at FROM ({query})')
        end = gate['data_as_of']
        if historical_window is not None:
            if historical_window != (spec['paired_validation']['start'], spec['paired_validation']['end']):
                raise ValueError('CORE6_RECONSTRUCTION_WINDOW_CHANGED')
            end = historical_window[1]
        calendar = connection.execute(f"SELECT cal_date FROM trade_calendar WHERE exchange='SSE' AND is_open=1 AND cal_date<=DATE '{end}' ORDER BY 1").pl()['cal_date'].to_list()
        if not calendar or str(calendar[-1]) != end or len(calendar) < 253:
            raise ValueError('CORE6_COMMON_CALENDAR_INCOMPLETE')
        start = calendar[-253]
        if historical_window is not None:
            first = next(i for i,d in enumerate(calendar) if str(d) >= historical_window[0])
            start = calendar[first-253]
        frames = {}
        for table in ('returns_daily', 'valuation_daily', 'security_daily_state', 'prices_daily'):
            frames[table] = connection.execute(f"SELECT * FROM {table} WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}' ORDER BY trade_date,asset_id").pl()
        history = connection.execute('SELECT * FROM industry_membership_history').pl()
        membership = connection.execute('SELECT * FROM benchmark_membership_history').pl()
        intervals = industry_intervals(history)
        connection.register('core6_industry', intervals)
        latest = connection.execute(f"""
          SELECT s.*,v.float_mkt_cap,v.known_at valuation_available_at,
            p.raw_close,p.known_at price_available_at,
            i.l1_code sw_l1_code,i.l2_code sw_l2_code,i.industry_observation_state,
            i.filled_from_adjacent,i.source_available_at industry_available_at
          FROM security_daily_state s
          LEFT JOIN valuation_daily v USING(trade_date,asset_id)
          LEFT JOIN prices_daily p USING(trade_date,asset_id)
          LEFT JOIN core6_industry i ON s.asset_id=i.asset_id AND i.effective_from<=s.trade_date
            AND (i.effective_to IS NULL OR s.trade_date<=i.effective_to)
          WHERE s.trade_date=DATE '{end}' AND s.in_a_share_scope AND s.listed_as_of
          ORDER BY s.asset_id
        """).pl()
        universe_history = None
        if historical_window is not None:
            universe_history = connection.execute(f"""
              SELECT s.trade_date,s.asset_id,s.board_id,s.exchange_list_date,s.days_since_exchange_list,
                s.is_st,s.is_suspended,s.known_at,v.float_mkt_cap,p.raw_close,
                i.l1_code sw_l1_code,i.l2_code sw_l2_code,i.source_available_at industry_available_at
              FROM security_daily_state s LEFT JOIN valuation_daily v USING(trade_date,asset_id)
              LEFT JOIN prices_daily p USING(trade_date,asset_id)
              LEFT JOIN core6_industry i ON s.asset_id=i.asset_id AND i.effective_from<=s.trade_date
                AND (i.effective_to IS NULL OR s.trade_date<=i.effective_to)
              WHERE s.trade_date BETWEEN DATE '{calendar[first-1]}' AND DATE '{end}'
                AND s.in_a_share_scope AND s.listed_as_of ORDER BY s.trade_date,s.asset_id
            """).pl()
            if universe_history.unique(['trade_date','asset_id']).height != universe_history.height:
                raise ValueError('CORE6_AMBIGUOUS_HISTORY_INDUSTRY')
        if latest.height != latest['asset_id'].n_unique():
            raise ValueError('CORE6_AMBIGUOUS_INDUSTRY_MEMBERSHIP')
        board = derive_board_benchmarks(connection, start, end)
        window = spec['paired_validation']
        availability = []
        for table in ('returns_daily', 'valuation_daily', 'security_daily_state'):
            # Metadata only: no outcomes, fit or Holdout evaluation.
            audit = connection.execute(f"""SELECT trade_date,count(*) n_rows,
              min(known_at) earliest_available_at,max(known_at) latest_available_at,
              count(*) FILTER(WHERE known_at::DATE <= trade_date) known_by_event_date
              FROM {table} WHERE trade_date BETWEEN DATE '{window['start']}' AND DATE '{window['end']}'
              GROUP BY 1 ORDER BY 1""").pl().with_columns(pl.lit(table).alias('table'))
            availability.append(audit)
        calendar_available = connection.execute("SELECT max(known_at) t FROM trade_calendar WHERE exchange='SSE'").pl()['t'][0]
    finally:
        connection.close()
    if SilverVersionLedger(lake).current()['version_id'] != version['version_id']:
        raise ValueError('CORE6_SILVER_CHANGED_DURING_READ')
    frames['returns_daily'] = frames['returns_daily'].with_columns(
        pl.col('known_at').alias('available_at'),
        pl.when(pl.col('first_seen_at') == datetime.fromisoformat(reader.base_revision_at))
        .then(pl.lit('conservative_base_first_seen')).otherwise(pl.lit('observed_first_seen'))
        .alias('availability_evidence'),
    )
    return {'gate': gate, 'calendar': calendar, 'calendar_available_at': calendar_available,
            'knowledge_time': knowledge_time, 'latest': latest, 'industry': intervals,
            'board': board, 'availability': pl.concat(availability),
            'source_records': source_records, 'frames': frames,
            'universe_history':universe_history,'membership':membership}
