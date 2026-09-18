"""Chronological market-fact backfill; never regenerate historical concept snapshots.

Component quality and full L0 readiness are deliberately separate. This service
does not activate research artifacts, change frozen samples or refresh risk bases.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .board_benchmarks import IndexDailyPipeline
from .boards import sync_board_membership
from .factor_data import FactorDataPipeline
from .incremental_market import IncrementalMarketPipeline
from .normalize import response_frame
from .revisioned_silver import SilverAsOfReader
from .storage import DataLake, json_hash, open_duckdb, source_tree_hash, utc_now


def open_dates(response, start: date, end: date) -> list[date]:
    frame = response_frame(response)
    if not {"cal_date", "is_open"} <= set(frame.columns):
        raise ValueError("CATCHUP_CALENDAR_SCHEMA")
    rows = frame.select("cal_date", "is_open").to_dicts()
    parsed = [(datetime.strptime(r["cal_date"], "%Y%m%d").date(), r["is_open"]) for r in rows]
    expected = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    if len(parsed) != len(expected) or {r[0] for r in parsed} != expected or any(r[1] not in (0, 1) for r in parsed):
        raise ValueError("CATCHUP_CALENDAR_INCOMPLETE")
    return sorted(d for d, is_open in parsed if is_open)


def coverage(lake: DataLake, dates: list[date]) -> dict:
    reader = SilverAsOfReader(lake)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    output = {}
    try:
        for table in ("prices_daily", "valuation_daily", "price_limits_daily", "security_daily_state", "returns_daily", "index_prices_daily", "risk_free_daily"):
            sql = reader.relation_sql(table, utc_now())
            rows = connection.execute(f'SELECT trade_date,count(*) FROM ({sql}) GROUP BY trade_date').fetchall()
            counts = dict(rows)
            output[table] = {"latest_date": str(max(counts)) if counts else None,
                             "requested_date_counts": {str(d): counts.get(d, 0) for d in dates},
                             "missing_dates": [str(d) for d in dates if not counts.get(d)]}
    finally:
        connection.close()
    return output


def backfill_market(client, lake: DataLake, *, start: date, end: date, credential_fingerprint: str,
                    progress=print, today: date | None = None) -> Path:
    today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if end < start or end >= today:
        raise ValueError("CATCHUP_REQUIRES_COMPLETED_PRIOR_DAYS")
    response = client.query("trade_cal", {"exchange": "SSE", "start_date": start.strftime("%Y%m%d"),
                                         "end_date": end.strftime("%Y%m%d")}, ["cal_date", "is_open"])
    bronze = lake.save_bronze(response, utc_now())
    dates = open_dates(response, start, end)
    identity = {"start": str(start), "end": str(end), "code_hash": source_tree_hash(), "calendar": bronze}
    run_id = "market_catchup_" + json_hash(identity)[:16]
    records, errors = [], []
    for index, day in enumerate(dates, 1):
        progress(f"Market catchup {index}/{len(dates)}: {day}")
        try:
            market = IncrementalMarketPipeline(client, lake, credential_fingerprint).sync(day)
            payload = json.loads(market.read_text())
            if payload["status"] != "passed":
                raise RuntimeError(f"MARKET_QUALITY_FAILED {payload.get('quality_gate')}")
            board = sync_board_membership(lake, day)
            indexes = IndexDailyPipeline(client, lake, credential_fingerprint).sync(day, day)
            risk = FactorDataPipeline(client, lake, credential_fingerprint).sync_risk_free(day, day)
            records.append({"trade_date": str(day), "status": "passed", "components": {
                "market": lake.artifact_record(market), "boards": lake.artifact_record(lake.manifests / f"{board['run_id']}.json"),
                "indexes": lake.artifact_record(indexes), "risk_free": lake.artifact_record(risk)}})
            progress(f"Market facts complete: {day}")
        except Exception as exc:
            errors.append({"trade_date": str(day), "error_type": type(exc).__name__, "detail": str(exc)})
            progress(f"Market catchup stopped: {day}: {type(exc).__name__}: {exc}")
            break  # Do not bridge an unverified missing day when constructing returns.
    observed = coverage(lake, dates)
    passed = not errors and all(not r["missing_dates"] for r in observed.values())
    if passed:
        lake.refresh_catalog()
    return lake.write_immutable_json(lake.manifests / f"{run_id}.json", {
        "schema_version": 1, "run_id": run_id, "job": "market_fact_catchup", "status": "passed" if passed else "failed",
        "identity": identity, "observed_at": utc_now().isoformat(), "days": records, "errors": errors,
        "coverage": observed, "full_l0_readiness": "not_evaluated", "research_promoted": False,
        "historical_concept_snapshots_synthesized": False,
    })
