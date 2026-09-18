"""One authoritative L0 readiness gate over revisioned Silver facts."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .board_benchmarks import INDEX_NAMES
from .concepts import concept_scan_passed, concept_snapshot_passed
from .revisioned_silver import SilverAsOfReader, SilverVersionLedger
from .storage import DataLake, json_hash, open_duckdb, source_tree_hash


DEFAULT_POLICY = Path("config/l0_production_v1.json")


def load_l0_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("L0_POLICY_SCHEMA_UNSUPPORTED")
    return payload


def _passed_manifest(lake: DataLake, job: str, target: date) -> Path | None:
    matches: list[tuple[str, Path]] = []
    for path in lake.manifests.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("job") != job or payload.get("quality_gate", {}).get("status") != "passed":
            continue
        configured_end = payload.get("config", {}).get("end")
        configured_risk_end = payload.get("config", {}).get("risk_end")
        observed_date = payload.get("trade_date") or configured_end or configured_risk_end
        if observed_date == target.isoformat() and payload.get("silver_version_id"):
            matches.append((payload.get("created_at") or payload.get("observed_at") or "", path))
    return max(matches, default=("", None), key=lambda item: item[0])[1]


def index_daily_passed(lake: DataLake, target: date) -> bool:
    return _passed_manifest(lake, "index_daily_sync", target) is not None


def risk_free_daily_passed(lake: DataLake, target: date) -> bool:
    return _passed_manifest(lake, "risk_free_daily_sync", target) is not None


def l0_ready_passed(lake: DataLake, target: date) -> bool:
    pattern = f"l0_ready_{target:%Y%m%d}_*.json"
    current = SilverVersionLedger(lake).current()
    if current is None:
        return False
    for path in lake.manifests.glob(pattern):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            payload.get("status") == "passed"
            and payload.get("silver_version_id") == current["version_id"]
            and payload.get("code_hash") == source_tree_hash()
        ):
            return True
    return False


def _latest_observation_date(lake: DataLake, job: str, fallback: date) -> date:
    dates: list[date] = []
    for path in lake.manifests.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("job") != job:
                continue
            value = payload.get("snapshot_date") or payload.get("created_at")
            if value:
                dates.append(date.fromisoformat(str(value)[:10]))
        except (OSError, ValueError):
            continue
    return max(dates, default=fallback)


def publish_l0_readiness(
    lake: DataLake,
    target: date,
    *,
    component_manifests: list[Path],
    policy_path: Path = DEFAULT_POLICY,
) -> Path:
    policy = load_l0_policy(policy_path)
    current = SilverVersionLedger(lake).current()
    if current is None:
        raise RuntimeError("L0_READY_REQUIRES_SILVER_VERSION")
    observed_at = datetime.now(UTC)
    reader = SilverAsOfReader(lake)
    state = reader.relation_sql("security_daily_state", observed_at)
    indexes = reader.relation_sql("index_prices_daily", observed_at)
    risk_free = reader.relation_sql("risk_free_daily", observed_at)
    industries = reader.relation_sql("industry_membership_history", observed_at)
    financial = reader.relation_sql("financial_pit", observed_at)
    income = reader.relation_sql("income_pit", observed_at)
    cashflow = reader.relation_sql("cashflow_pit", observed_at)
    benchmarks = reader.relation_sql("benchmark_membership_history", observed_at)
    standard = policy["industry"]["standard"]
    level = policy["industry"]["level"].lower()
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        state_stats = connection.execute(f"""
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE in_a_share_scope) AS in_scope,
                   count(*) FILTER (
                     WHERE in_a_share_scope AND observation_state='SOURCE_INCOMPLETE'
                   ) AS incomplete,
                   count(*) FILTER (
                     WHERE in_a_share_scope AND board_id NOT IN ('MAIN','CHINEXT','STAR','BSE')
                   ) AS unknown_board,
                   count(DISTINCT board_id) FILTER (WHERE in_a_share_scope) AS boards
            FROM ({state}) WHERE trade_date=DATE '{target.isoformat()}'
        """).fetchone()
        index_rows = connection.execute(f"""
            SELECT count(DISTINCT index_code) FROM ({indexes})
            WHERE trade_date=DATE '{target.isoformat()}'
        """).fetchone()[0]
        risk_stats = connection.execute(f"""
            SELECT count(*),count(*) FILTER (WHERE rate_date>=trade_date)
            FROM ({risk_free}) WHERE trade_date=DATE '{target.isoformat()}'
        """).fetchone()
        industry_stats = connection.execute(f"""
            WITH eligible AS (
              SELECT asset_id FROM ({state})
              WHERE trade_date=DATE '{target.isoformat()}' AND in_a_share_scope AND board_id!='BSE'
            ), active AS (
              SELECT asset_id,count(*) AS n FROM ({industries})
              WHERE classification_standard='{standard}'
                AND in_date<=DATE '{target.isoformat()}'
                AND (out_date IS NULL OR out_date>DATE '{target.isoformat()}')
                AND {level}_code IS NOT NULL
              GROUP BY asset_id
            )
            SELECT count(*) AS eligible,
                   count(*) FILTER (WHERE active.asset_id IS NOT NULL) AS covered,
                   count(*) FILTER (WHERE active.n>1) AS ambiguous
            FROM eligible LEFT JOIN active USING(asset_id)
        """).fetchone()
        static_stats = connection.execute(f"""
            SELECT
              (SELECT count(*) FROM ({financial})) AS financial_rows,
              (SELECT count(*) FROM ({income})) AS income_rows,
              (SELECT count(*) FROM ({cashflow})) AS cashflow_rows,
              (SELECT count(*) FROM ({benchmarks})) AS benchmark_rows,
              (SELECT count(DISTINCT benchmark_id) FROM ({benchmarks})) AS benchmarks
        """).fetchone()
    finally:
        connection.close()

    base_frozen = date.fromisoformat(
        json.loads(
            (lake.metadata / "base_manifests" / f"{current['base_id']}.json").read_text(
                encoding="utf-8"
            )
        )["frozen_at"][:10]
    )
    industry_observed = _latest_observation_date(
        lake, "industry_classification_sync", base_frozen
    )
    concept_observed = _latest_observation_date(
        lake, "concept_membership_snapshot", base_frozen
    )
    coverage = industry_stats[1] / industry_stats[0] if industry_stats[0] else 0.0
    checks = {
        "security_state_present": bool(state_stats[0]),
        "security_state_source_complete": state_stats[2] == 0,
        "board_assignment_complete": state_stats[3] == 0 and state_stats[4] == 4,
        "official_indexes_complete": index_rows == len(INDEX_NAMES),
        "risk_free_present_and_lagged": risk_stats[0] == 1 and risk_stats[1] == 0,
        "industry_coverage": coverage >= float(policy["industry"]["minimum_coverage"]),
        "industry_unambiguous": industry_stats[2] == 0,
        "industry_refresh_current": (
            target - industry_observed
        ).days <= int(policy["industry"]["full_refresh_max_calendar_days"]),
        "concept_daily_observed": concept_scan_passed(lake, target)
        or concept_snapshot_passed(lake, target),
        "concept_reconciliation_current": (
            target - concept_observed
        ).days <= int(policy["concepts"]["full_reconciliation_max_calendar_days"]),
        "financial_pit_present": static_stats[0] > 0,
        "income_pit_present": static_stats[1] > 0,
        "cashflow_pit_present": static_stats[2] > 0,
        "benchmark_membership_present": static_stats[3] > 0 and static_stats[4] >= 3,
        "external_facts_configured": (Path("config/external_facts_v1.json")).exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    inputs = [lake.artifact_record(path) for path in component_manifests]
    identity = {
        "target": target.isoformat(), "silver_version_id": current["version_id"],
        "policy_hash": json_hash(policy), "code_hash": source_tree_hash(),
        "inputs": inputs, "checks": checks,
    }
    run_id = f"l0_ready_{target:%Y%m%d}_{json_hash(identity)[:12]}"
    return lake.write_manifest(run_id, {
        "schema_version": 1, "run_id": run_id, "job": "l0_readiness",
        "status": "passed" if not failed else "failed", "trade_date": target.isoformat(),
        "observed_at": observed_at.isoformat(), "silver_version_id": current["version_id"],
        "code_hash": source_tree_hash(),
        "policy": policy, "policy_hash": json_hash(policy), "inputs": inputs,
        "quality_gate": {
            "status": "passed" if not failed else "failed", "checks": checks,
            "failed_checks": failed,
            "counts": {
                "security_state_rows": state_stats[0], "in_scope_assets": state_stats[1],
                "official_indexes": index_rows, "risk_free_rows": risk_stats[0],
                "industry_eligible": industry_stats[0], "industry_covered": industry_stats[1],
                "industry_coverage": coverage,
                "financial_rows": static_stats[0], "income_rows": static_stats[1],
                "cashflow_rows": static_stats[2], "benchmark_membership_rows": static_stats[3],
                "benchmarks": static_stats[4],
            },
        },
    })
