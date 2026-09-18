"""G0 full-history diagnostics before any L1/L2 artifact is allowed to run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..revisioned_silver import SilverAsOfReader, SilverVersionLedger
from ..storage import DataLake, json_hash, open_duckdb, source_tree_hash
from ..calculation.l2.label_policy import load_return_label_policy
from .diff import compare_reports
from .history_queries import industry_summary, one, return_distribution, rows


DEFAULT_CONFIG = Path("config/g0_history_diagnostics_v1.json")


def _latest_previous_report(lake: DataLake, current_version_id: str) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in lake.manifests.glob("g0_history_*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("job") != "g0_history_diagnostics"
                or manifest.get("silver_version_id") == current_version_id
                or manifest.get("execution_status") != "passed"
            ):
                continue
            report_path = lake.root / manifest["outputs"]["report"]["path"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            candidates.append((manifest["as_of_timestamp"], report))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return max(candidates, default=("", None), key=lambda item: item[0])[1]


def _build_report(
    lake: DataLake, config: dict[str, Any], version: dict[str, Any], run_id: str
) -> dict[str, Any]:
    as_of = datetime.fromisoformat(version["observed_at"])
    reader = SilverAsOfReader(lake, version["base_id"])
    state = reader.relation_sql("security_daily_state", as_of)
    returns = reader.relation_sql("returns_daily", as_of)
    financial = reader.relation_sql("financial_pit", as_of)
    industry = reader.relation_sql("industry_membership_history", as_of)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        history = one(connection, f"""
            SELECT min(trade_date) AS minimum_trade_date,
                   max(trade_date) AS maximum_trade_date,
                   count(DISTINCT trade_date) AS trading_dates
            FROM ({state})
        """)
        universe = one(connection, f"""
            WITH daily AS (
              SELECT trade_date, count(*) FILTER (WHERE in_a_share_scope) AS n
              FROM ({state}) GROUP BY 1
            )
            SELECT min(n) AS minimum_count, max(n) AS maximum_count,
                   arg_max(n, trade_date) AS latest_count
            FROM daily
        """)
        move_limit = int(config["diff"]["largest_universe_moves"])
        universe["largest_daily_moves"] = rows(connection, f"""
            WITH daily AS (
              SELECT trade_date, count(*) FILTER (WHERE in_a_share_scope) AS n
              FROM ({state}) GROUP BY 1
            ), moved AS (
              SELECT trade_date,n,n-lag(n) OVER (ORDER BY trade_date) AS change
              FROM daily
            )
            SELECT trade_date,n,change FROM moved WHERE change IS NOT NULL
            ORDER BY abs(change) DESC,trade_date LIMIT {move_limit}
        """)
        board_transitions = rows(connection, f"""
            WITH daily AS (
              SELECT trade_date,count(DISTINCT board_id) FILTER (WHERE in_a_share_scope) AS n,
                     string_agg(DISTINCT board_id, ',' ORDER BY board_id)
                       FILTER (WHERE in_a_share_scope) AS board_ids
              FROM ({state}) GROUP BY 1
            ), marked AS (
              SELECT *,lag(n) OVER (ORDER BY trade_date) AS previous_n FROM daily
            )
            SELECT trade_date,n AS board_count,board_ids FROM marked
            WHERE previous_n IS NULL OR n!=previous_n ORDER BY trade_date
        """)
        state_totals = rows(connection, f"""
            SELECT observation_state,count(*) AS n FROM ({state})
            GROUP BY 1 ORDER BY 1
        """)
        sample_limit = int(config["diff"]["sample_rows"])
        incomplete_samples = rows(connection, f"""
            SELECT trade_date,asset_id,board_id,has_price,has_valuation,has_limit,
                   is_suspended,source_day_complete
            FROM ({state}) WHERE in_a_share_scope
              AND observation_state='SOURCE_INCOMPLETE'
            ORDER BY trade_date,asset_id LIMIT {sample_limit}
        """)
        neeq_by_year = rows(connection, f"""
            SELECT year(trade_date) AS year,count(*) AS n FROM ({state})
            WHERE observation_state='PRE_EXCHANGE_NEEQ'
            GROUP BY 1 ORDER BY 1
        """)
        policy_rows = rows(connection, f"""
            SELECT revision_policy,count(*) AS n FROM ({financial})
            GROUP BY 1 ORDER BY 1
        """)
        financial_summary = one(connection, f"""
            SELECT count(*) AS revision_count,
                   count(*) FILTER (
                     WHERE revision_policy='conservative_revision_first_seen'
                   ) AS conservative_first_seen_count
            FROM ({financial})
        """)
        total = financial_summary["revision_count"] or 0
        financial_summary["conservative_first_seen_ratio"] = (
            financial_summary["conservative_first_seen_count"] / total if total else None
        )
        financial_summary["revision_policy_counts"] = {
            row["revision_policy"]: row["n"] for row in policy_rows
        }
        levels = config["industry"]["metadata_levels"]
        industry_summaries = {
            level: industry_summary(
                connection, state, industry,
                config["industry"]["classification_standard"], level,
            )
            for level in levels
        }
        warning = float(config["return_distribution"]["absolute_warning_threshold"])
        extreme = float(config["return_distribution"]["absolute_extreme_threshold"])
        transition = return_distribution(
            connection,
            f"""SELECT state.trade_date,state.asset_id,state.exchange_list_date,
                         state.board_id,returns.total_return
                  FROM ({state}) state LEFT JOIN ({returns}) returns
                    USING(trade_date,asset_id)""",
            "trade_date=exchange_list_date AND board_id='BSE'",
            [], warning, extreme,
        )
        resumption = return_distribution(
            connection,
            f"""SELECT returns.*,state.exchange_list_date
                  FROM ({returns}) returns JOIN ({state}) state
                    USING(trade_date,asset_id)""",
            "return_source='resumption' AND trade_date<>exchange_list_date",
            [], warning, extreme,
        )
    finally:
        connection.close()

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "diagnostic_id": config["diagnostic_id"],
        "silver_version_id": version["version_id"],
        "as_of_timestamp": version["observed_at"],
        "history": history,
        "universe": universe,
        "board_cardinality_transitions": board_transitions,
        "observation_states": {
            "totals": {row["observation_state"]: row["n"] for row in state_totals},
            "source_incomplete_samples": incomplete_samples,
            "pre_exchange_neeq_by_year": neeq_by_year,
        },
        "industry": {
            "classification_standard": config["industry"]["classification_standard"],
            "regression_level": config["industry"]["regression_level"],
            "levels": industry_summaries,
        },
        "financial_pit": financial_summary,
        "transition_first_day_returns": transition,
        "resumption_returns_excluding_exchange_first_day": resumption,
    }
    label_policy_path = Path(config["l2_label_policy_path"])
    label_policy = load_return_label_policy(label_policy_path)
    report["l2_readiness"] = {
        "status": "ready_for_l1_l2_build",
        "blocking_findings": [],
        "resolved_findings": [
            "BSE_EXCHANGE_FIRST_DAY_EXCLUDED_BY_FROZEN_LABEL_POLICY",
            "RESUMPTION_DAY_EXCLUDED_FROM_ONE_DAY_REGRESSION_AND_RETAINED_FOR_PNL",
        ],
        "label_policy_id": label_policy.policy_id,
        "label_policy_sha": json_hash(
            json.loads(label_policy_path.read_text(encoding="utf-8"))
        ),
    }
    return report


def run_g0_history_diagnostics(
    lake: DataLake, config_path: Path = DEFAULT_CONFIG
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("G0_DIAGNOSTIC_CONFIG_SCHEMA_UNSUPPORTED")
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("G0_DIAGNOSTICS_REQUIRES_SILVER_VERSION")
    identity = {
        "diagnostic_id": config["diagnostic_id"],
        "silver_version_id": version["version_id"],
        "config_sha": json_hash(config),
        "code_hash": source_tree_hash(),
    }
    run_id = f"g0_history_{json_hash(identity)[:16]}"
    manifest_path = lake.manifests / f"{run_id}.json"
    if manifest_path.exists():
        return manifest_path

    report = _build_report(lake, config, version, run_id)
    previous = _latest_previous_report(lake, version["version_id"])
    report["comparison"] = compare_reports(report, previous)
    output = lake.root / "diagnostics" / "g0_history" / f"run_id={run_id}" / "report.json"
    lake.write_immutable_json(output, report)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "g0_history_diagnostics",
        "execution_status": "passed",
        "l2_readiness": report["l2_readiness"],
        "as_of_timestamp": version["observed_at"],
        "silver_version_id": version["version_id"],
        "input_version_manifest": lake.artifact_record(
            lake.metadata / "silver_versions" / f"{version['version_id']}.json"
        ),
        "config": {"path": str(config_path), "sha256": json_hash(config)},
        "code_hash": identity["code_hash"],
        "outputs": {"report": lake.artifact_record(output)},
        "comparison": report["comparison"],
    }
    return lake.write_immutable_json(manifest_path, manifest)
