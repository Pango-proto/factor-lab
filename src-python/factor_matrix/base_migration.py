"""One-time, read-only validation and factual enrichments for the frozen base."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from .storage import DataLake, file_sha256, json_hash, open_duckdb, utc_now


BSE_OPEN_DATE = date(2021, 11, 15)
BSE_TRANSFER_COUNT = 71
BSE_DIRECT_IPO_COUNT = 10
BSE_OPENING_SOURCE = "https://www.gov.cn/xinwen/2021-11/12/content_5650529.htm"
BSE_CODE_SOURCE = "https://www.bse.cn/service/code_mapping.html"


def _load_snapshot(lake: DataLake, snapshot_id: str) -> dict[str, Any]:
    path = lake.metadata / "snapshots" / f"{snapshot_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_legacy_base(lake: DataLake, snapshot_id: str) -> dict[str, Any]:
    snapshot = _load_snapshot(lake, snapshot_id)
    changed: list[str] = []
    for table, artifact in snapshot["silver"].items():
        path = lake.root / artifact["path"]
        if not path.exists() or file_sha256(path) != artifact["sha256"]:
            changed.append(table)
    if changed:
        raise RuntimeError(f"LEGACY_BASE_CHANGED {changed}")

    connection = open_duckdb(lake.metadata / "catalog.duckdb", read_only=True)
    try:
        price_mismatch = connection.execute(
            """
            WITH x AS (
              SELECT asset_id, trade_date, total_return, gross_close_index,
                     lag(gross_close_index) OVER (
                       PARTITION BY asset_id ORDER BY trade_date
                     ) AS previous_gross
              FROM prices_daily
            )
            SELECT count(*) FROM x
            WHERE total_return IS NOT NULL AND previous_gross IS NOT NULL
              AND abs(total_return - (gross_close_index / previous_gross - 1)) > 1e-12
            """
        ).fetchone()[0]
        return_mismatch = connection.execute(
            """
            SELECT count(*) FROM returns_daily r
            JOIN prices_daily p USING (trade_date, asset_id)
            WHERE r.return_source='price'
              AND r.total_return IS NOT NULL AND p.total_return IS NOT NULL
              AND abs(r.total_return-p.total_return)>1e-12
            """
        ).fetchone()[0]
    finally:
        connection.close()
    if price_mismatch or return_mismatch:
        raise RuntimeError(
            f"LEGACY_BASE_RETURN_MISMATCH price={price_mismatch} returns={return_mismatch}"
        )

    path = lake.metadata / "base_manifests" / "legacy_base_v1.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "base_id": "legacy_base_v1",
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "frozen_at": utc_now().isoformat(),
        "status": "validated_frozen",
        "physical_rewrite_performed": False,
        "content_hash_policy": "sha256_file_bytes_stable_table_order",
        "return_validation": {
            "formula": "raw_close_t*adj_factor_t/(raw_close_prev*adj_factor_prev)-1",
            "price_mismatch_rows": int(price_mismatch),
            "returns_daily_price_source_mismatch_rows": int(return_mismatch),
            "returns_daily_role": "derived_view_not_immutable_input_fact",
        },
        "deprecated_columns": {
            "prices_daily": ["qfq_close", "qfq_anchor_date", "total_return"],
            "runtime_policy": "not_exposed_by_canonical_price_view",
        },
        "artifacts": {
            table: {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "role": "derived" if table == "returns_daily" else "base_fact",
            }
            for table, artifact in sorted(snapshot["silver"].items())
        },
    }
    lake.write_immutable_json(path, payload)
    return payload


def materialize_bse_transfer_batch(lake: DataLake, snapshot_id: str) -> dict[str, Any]:
    snapshot = _load_snapshot(lake, snapshot_id)
    output_dir = (
        lake.silver / "base" / f"snapshot_id={snapshot_id}" / "bse_transfer_batch"
    )
    output = output_dir / "data.parquet"
    manifest_path = lake.metadata / "base_manifests" / "bse_transfer_batch_v1.json"
    if output.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    escaped = str(temporary).replace("'", "''")
    connection = open_duckdb(
        lake.metadata / "catalog.duckdb", read_only=True,
        temp_directory=lake.root / "tmp" / "duckdb",
    )
    try:
        transfer_query = f"""
            WITH securities AS (
              SELECT asset_id, min(list_date) AS source_list_date
              FROM security_master GROUP BY asset_id
            ), observations AS (
              SELECT asset_id FROM prices_daily WHERE trade_date=DATE '{BSE_OPEN_DATE}'
              UNION SELECT asset_id FROM valuation_daily WHERE trade_date=DATE '{BSE_OPEN_DATE}'
              UNION SELECT asset_id FROM suspensions_daily WHERE trade_date=DATE '{BSE_OPEN_DATE}'
            )
            SELECT s.asset_id,
                   s.source_list_date AS neeq_list_date,
                   DATE '{BSE_OPEN_DATE}' AS exchange_list_date,
                   'SELECTED_LAYER_TRANSFER' AS transfer_type,
                   TRUE AS manually_review_required,
                   '{snapshot_id}' AS source_snapshot_id,
                   '{BSE_OPENING_SOURCE}' AS opening_source_url,
                   '{BSE_CODE_SOURCE}' AS code_source_url
            FROM securities s JOIN observations o USING(asset_id)
            WHERE right(s.asset_id,3)='.BJ'
              AND s.source_list_date < DATE '{BSE_OPEN_DATE}'
            ORDER BY s.asset_id
        """
        transfer_count = connection.execute(
            f"SELECT count(*) FROM ({transfer_query})"
        ).fetchone()[0]
        direct_count = connection.execute(
            f"""
            WITH securities AS (
              SELECT asset_id,min(list_date) AS source_list_date
              FROM security_master GROUP BY asset_id
            )
            SELECT count(*) FROM securities s
            JOIN prices_daily p USING(asset_id)
            WHERE p.trade_date=DATE '{BSE_OPEN_DATE}'
              AND right(s.asset_id,3)='.BJ'
              AND s.source_list_date=DATE '{BSE_OPEN_DATE}'
            """
        ).fetchone()[0]
        if transfer_count != BSE_TRANSFER_COUNT or direct_count != BSE_DIRECT_IPO_COUNT:
            raise RuntimeError(
                "BSE_OPENING_BATCH_COUNT_MISMATCH "
                f"transfer={transfer_count} direct={direct_count}"
            )
        connection.execute(f"COPY ({transfer_query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()
    os.replace(temporary, output)
    payload = {
        "schema_version": 1,
        "artifact_id": "bse_transfer_batch_v1",
        "status": "validated_frozen",
        "source_snapshot_id": snapshot_id,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "exchange_open_date": BSE_OPEN_DATE.isoformat(),
        "transfer_count": BSE_TRANSFER_COUNT,
        "direct_ipo_count_validation": BSE_DIRECT_IPO_COUNT,
        "selection_rule": "BSE asset with price, valuation or suspension observation on opening date and source_list_date before opening date",
        "official_count_source": BSE_OPENING_SOURCE,
        "code_mapping_source": BSE_CODE_SOURCE,
        "output": lake.artifact_record(output),
    }
    payload["content_hash"] = json_hash(payload["output"])
    lake.write_immutable_json(manifest_path, payload)
    return payload


def materialize_security_daily_state(lake: DataLake, snapshot_id: str) -> dict[str, Any]:
    """Scan the frozen base once and persist policy-free daily security facts."""
    snapshot = _load_snapshot(lake, snapshot_id)
    transfer = (
        lake.silver / "base" / f"snapshot_id={snapshot_id}"
        / "bse_transfer_batch" / "data.parquet"
    )
    if not transfer.exists():
        raise RuntimeError("BSE_TRANSFER_BATCH_REQUIRED")
    output_dir = (
        lake.silver / "base" / f"snapshot_id={snapshot_id}"
        / "security_daily_state"
    )
    output = output_dir / "data.parquet"
    manifest_path = lake.metadata / "base_manifests" / "security_daily_state_v1.json"
    if output.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    escaped_output = str(temporary).replace("'", "''")
    escaped_transfer = str(transfer).replace("'", "''")
    state_query = f"""
        WITH bounds AS (
          SELECT min(trade_date) AS lo, max(trade_date) AS hi FROM prices_daily
        ), open_dates AS (
          SELECT DISTINCT cal_date AS trade_date FROM trade_calendar,bounds
          WHERE is_open=1 AND cal_date BETWEEN lo AND hi
        ), securities AS (
          SELECT asset_id, symbol, market, exchange, list_status,
                 list_date AS source_list_date, delist_date,
                 CASE WHEN t.asset_id IS NOT NULL THEN t.exchange_list_date
                      ELSE list_date END AS exchange_list_date
          FROM security_master s
          LEFT JOIN read_parquet('{escaped_transfer}') t USING(asset_id)
        ), listed_grid AS (
          SELECT d.trade_date,s.*
          FROM securities s JOIN open_dates d
            ON d.trade_date>=s.exchange_list_date
           AND (s.delist_date IS NULL OR d.trade_date<s.delist_date)
        ), pre_exchange_observations AS (
          SELECT DISTINCT x.trade_date,s.*
          FROM securities s
          JOIN (
            SELECT trade_date,asset_id FROM prices_daily
            UNION SELECT trade_date,asset_id FROM valuation_daily
            UNION SELECT trade_date,asset_id FROM suspensions_daily
          ) x USING(asset_id)
          WHERE s.exchange='BSE' AND x.trade_date<s.exchange_list_date
        ), grid AS (
          SELECT * FROM listed_grid
          UNION ALL SELECT * FROM pre_exchange_observations
        ), joined AS (
          SELECT g.*,
                 p.asset_id IS NOT NULL AS has_price,
                 v.asset_id IS NOT NULL AS has_valuation,
                 l.asset_id IS NOT NULL AS has_limit,
                 st.asset_id IS NOT NULL AS is_st,
                 coalesce(su.is_suspended,FALSE) AS is_suspended,
                 p.raw_close,p.raw_open,p.raw_high,p.raw_low,l.limit_up,l.limit_down
          FROM grid g
          LEFT JOIN prices_daily p USING(trade_date,asset_id)
          LEFT JOIN valuation_daily v USING(trade_date,asset_id)
          LEFT JOIN price_limits_daily l USING(trade_date,asset_id)
          LEFT JOIN (SELECT DISTINCT trade_date,asset_id FROM stock_st_daily) st
            USING(trade_date,asset_id)
          LEFT JOIN (
            SELECT trade_date,asset_id,
                   max(CASE WHEN suspend_type='S' THEN 1 ELSE 0 END)::BOOLEAN AS is_suspended
            FROM suspensions_daily GROUP BY trade_date,asset_id
          ) su USING(trade_date,asset_id)
        ), classified AS (
          SELECT *,
                 trade_date>=exchange_list_date
                   AND (delist_date IS NULL OR trade_date<delist_date)
                   AS in_a_share_scope,
                 CASE
                   WHEN exchange='BSE' AND trade_date<exchange_list_date THEN 'NEEQ'
                   ELSE exchange
                 END AS venue,
                 CASE
                   WHEN market='北交所' OR exchange='BSE' OR asset_id LIKE '%.BJ' THEN 'BSE'
                   WHEN market='科创板' OR symbol LIKE '688%' OR symbol LIKE '689%' THEN 'STAR'
                   WHEN market='创业板' OR symbol LIKE '300%' OR symbol LIKE '301%' THEN 'CHINEXT'
                   WHEN market='主板' OR exchange IN ('SSE','SZSE','SH','SZ') THEN 'MAIN'
                   ELSE 'UNKNOWN'
                 END AS board_id
          FROM joined
        )
        SELECT trade_date,asset_id,
               asset_id AS source_asset_id,
               'CANONICALIZED_LEGACY_BASE' AS source_identity_state,
               source_list_date,
               CASE WHEN exchange='BSE' THEN source_list_date END AS neeq_list_date,
               exchange_list_date,delist_date,list_status,
               venue,board_id,in_a_share_scope,
               in_a_share_scope AS listed_as_of,
               CASE WHEN in_a_share_scope THEN
                 row_number() OVER (
                   PARTITION BY asset_id,in_a_share_scope ORDER BY trade_date
                 )-1
               END::INTEGER AS days_since_exchange_list,
               has_price,has_valuation,has_limit,is_st,is_suspended,
               coalesce(has_limit AND raw_close>=limit_up, FALSE) AS is_limit_up,
               coalesce(has_limit AND raw_close<=limit_down, FALSE) AS is_limit_down,
               CASE
                 WHEN NOT in_a_share_scope THEN 'PRE_EXCHANGE_NEEQ'
                 WHEN trade_date=exchange_list_date THEN 'LISTING_DAY'
                 WHEN has_price AND has_valuation THEN 'TRADED'
                 WHEN is_suspended THEN 'SUSPENDED_CONFIRMED'
                 WHEN NOT has_price AND NOT has_valuation THEN 'NONTRADING_INFERRED'
                 ELSE 'SOURCE_INCOMPLETE'
               END AS observation_state,
               TRUE AS source_day_complete,
               '{snapshot_id}' AS source_snapshot_id,
               TIMESTAMP '{utc_now().replace(tzinfo=None).isoformat()}' AS revision_at
        FROM classified
        ORDER BY trade_date,asset_id
    """
    connection = open_duckdb(
        lake.metadata / "catalog.duckdb", read_only=True,
        temp_directory=lake.root / "tmp" / "duckdb",
    )
    try:
        connection.execute(
            f"COPY ({state_query}) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()
    os.replace(temporary, output)

    check = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        path = str(output).replace("'", "''")
        stats = check.execute(
            f"""
            SELECT count(*) AS rows,count(DISTINCT trade_date) AS dates,
                   count(DISTINCT asset_id) AS assets,
                   count(*)-count(DISTINCT (trade_date,asset_id)) AS duplicates,
                   count(*) FILTER (
                     WHERE observation_state='PRE_EXCHANGE_NEEQ' AND in_a_share_scope
                   ) AS scope_violations,
                   count(*) FILTER (
                     WHERE trade_date=DATE '{BSE_OPEN_DATE}'
                       AND board_id='BSE' AND in_a_share_scope
                   ) AS bse_opening_assets
            FROM read_parquet('{path}')
            """
        ).fetchone()
    finally:
        check.close()
    if stats[3] or stats[4] or stats[5] != BSE_TRANSFER_COUNT+BSE_DIRECT_IPO_COUNT:
        raise RuntimeError(f"SECURITY_DAILY_STATE_QUALITY_FAILED {stats}")
    payload = {
        "schema_version": 1,
        "artifact_id": "security_daily_state_v1",
        "status": "validated_frozen",
        "source_snapshot_id": snapshot_id,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "policy_free": True,
        "primary_key": ["trade_date", "asset_id"],
        "observation_states": [
            "TRADED", "LISTING_DAY", "SUSPENDED_CONFIRMED",
            "NONTRADING_INFERRED", "PRE_EXCHANGE_NEEQ", "SOURCE_INCOMPLETE",
        ],
        "source_complete_policy": "validated base market days are complete; per-asset partial observations are classified explicitly",
        "quality_gate": {
            "status": "passed",
            "rows": stats[0], "dates": stats[1], "assets": stats[2],
            "duplicate_rows": stats[3], "scope_violations": stats[4],
            "bse_opening_assets": stats[5],
        },
        "output": lake.artifact_record(output),
    }
    lake.write_immutable_json(manifest_path, payload)
    return payload
