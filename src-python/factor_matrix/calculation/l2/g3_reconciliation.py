"""Auditable G3 row-waterfall reconciliation before G4 residual diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, utc_now
from .g3_gate import G3_GATE_VERSION
from .g3_runner import load_current_g3_manifest


def _quoted(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def run_g3_reconciliation(lake: DataLake) -> Path:
    l1_current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    l1_current = json.loads(l1_current_path.read_text(encoding="utf-8"))
    l1_manifest_path = lake.root / l1_current["manifest"]
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    g3_current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    g3_current = json.loads(g3_current_path.read_text(encoding="utf-8"))
    l1 = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    if (
        g3.get("status") != "passed"
        or g3.get("l1_run_id") != l1.get("run_id")
        or g3_current.get("gate_version") != G3_GATE_VERSION
        or not g3_current.get("gate_attestation_manifest")
    ):
        raise RuntimeError("G3_RECONCILIATION_REQUIRES_MATCHING_PASSED_CURRENT")
    exposure_path = lake.root / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    universe_matches = list((lake.root / "gold" / "tradable_universe").glob(
        f"artifact_version=1/run_id={l1['tradable_universe_run_id']}/tradable_universe.parquet"
    ))
    if len(universe_matches) != 1:
        raise RuntimeError("G3_RECONCILIATION_UNIVERSE_NOT_UNIQUE")
    universe_path = universe_matches[0]
    quality_path = lake.root / g3["outputs"]["factor_regression_quality_v1"]["path"]
    specific_path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    returns_path = lake.silver / "returns_daily" / "data.parquet"

    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        common = f"""
        WITH date_map AS (
          SELECT exposure_date, trade_date AS return_date
          FROM read_parquet('{_quoted(quality_path)}')
        ), universe_signal AS (
          SELECT u.*, m.return_date
          FROM read_parquet('{_quoted(universe_path)}') u
          JOIN date_map m ON u.trade_date=m.exposure_date
        ), exposure AS (
          SELECT e.*, m.return_date
          FROM read_parquet('{_quoted(exposure_path)}') e
          JOIN date_map m ON e.trade_date=m.exposure_date
          WHERE e.is_valid
        ), labelled AS (
          SELECT e.trade_date, e.return_date, e.asset_id, u.exchange_list_date,
                 r.total_return, r.return_source,
                 CASE
                   WHEN e.return_date=u.exchange_list_date THEN 'exchange_first_day'
                   WHEN r.return_source='resumption' THEN 'resumption_day'
                   WHEN r.total_return IS NULL OR NOT isfinite(r.total_return) THEN 'invalid_return'
                   ELSE 'eligible'
                 END AS label_state
          FROM exposure e
          JOIN universe_signal u USING(trade_date,return_date,asset_id)
          LEFT JOIN read_parquet('{_quoted(returns_path)}') r
            ON r.trade_date=e.return_date AND r.asset_id=e.asset_id
        )
        """
        counts = connection.execute(common + f"""
        SELECT
          (SELECT count(*) FROM universe_signal) AS universe_signal_rows,
          (SELECT count(*) FROM universe_signal WHERE NOT base_tradable_without_listing_age)
            AS base_policy_excluded,
          (SELECT count(*) FROM universe_signal
            WHERE base_tradable_without_listing_age AND days_since_exchange_list<20)
            AS d0_excluded,
          (SELECT count(*) FROM universe_signal
            WHERE base_tradable_without_listing_age AND days_since_exchange_list>=20)
            AS frozen_d0_rows,
          (SELECT count(*) FROM universe_signal
            WHERE base_tradable_without_listing_age AND days_since_exchange_list>=20
              AND (sw_l1_code IS NULL OR board_id IS NULL)) AS structure_missing,
          (SELECT count(*) FROM exposure) AS l1_valid_exposure_rows,
          (SELECT count(*) FROM labelled WHERE label_state='exchange_first_day')
            AS exchange_first_day,
          (SELECT count(*) FROM labelled WHERE label_state='resumption_day') AS resumption_day,
          (SELECT count(*) FROM labelled WHERE label_state='invalid_return') AS invalid_return,
          (SELECT count(*) FROM labelled WHERE label_state='eligible') AS label_eligible,
          (SELECT sum(huber_downweight_count) FROM read_parquet('{_quoted(quality_path)}'))
            AS huber_downweighted,
          (SELECT sum(sample_count) FROM read_parquet('{_quoted(quality_path)}'))
            AS quality_estimation_rows,
          (SELECT count(*) FROM read_parquet('{_quoted(specific_path)}')) AS specific_rows,
          (SELECT count(*) FROM read_parquet('{_quoted(specific_path)}')
            WHERE in_estimation_domain) AS specific_estimation_rows,
          (SELECT count(*) FROM read_parquet('{_quoted(specific_path)}')
            WHERE exclusion_reason='limit_locked') AS limit_locked_rows
        """).pl()
        exact_reasons = connection.execute(common + """
        SELECT exclusion_reasons, count(*)::BIGINT AS rows
        FROM universe_signal
        WHERE NOT base_tradable_without_listing_age
        GROUP BY exclusion_reasons ORDER BY rows DESC, exclusion_reasons
        """).pl()
    finally:
        connection.close()

    row = counts.row(0, named=True)
    reconciled = (
        row["universe_signal_rows"] - row["base_policy_excluded"]
        - row["d0_excluded"] - row["structure_missing"]
        - row["exchange_first_day"] - row["resumption_day"]
        - row["invalid_return"]
    )
    if reconciled != row["specific_rows"]:
        raise RuntimeError("G3_ROW_WATERFALL_IDENTITY_FAILED")
    estimation_reconciled = row["specific_rows"] - row["limit_locked_rows"]
    if not (
        estimation_reconciled == row["specific_estimation_rows"]
        == row["quality_estimation_rows"]
    ):
        raise RuntimeError("G3_ESTIMATION_DOMAIN_WATERFALL_IDENTITY_FAILED")
    l1_dates = int(l1["counts"]["dates"])
    warmup_dates = int(l1["counts"]["invalid_dates"])
    g3_dates = int(g3["counts"]["dates"])
    no_next_label_dates = l1_dates - warmup_dates - g3_dates
    if warmup_dates != 60 or no_next_label_dates != 1:
        raise RuntimeError("G3_UNIFIED_WARMUP_RECONCILIATION_FAILED")

    identity: dict[str, Any] = {
        "g3_run_id": g3["run_id"], "l1_run_id": l1["run_id"],
        "gate_version": g3_current["gate_version"],
        "gate_config_sha": g3_current["gate_config_sha"],
        "gate_attestation_manifest": g3_current["gate_attestation_manifest"],
        "input_hashes": {
            "g3_manifest": file_sha256(g3_manifest_path),
            "l1_manifest": file_sha256(l1_manifest_path),
            "g3_attestation": file_sha256(
                lake.root / g3_current["gate_attestation_manifest"]
            ),
            "universe": file_sha256(universe_path),
            "returns": file_sha256(returns_path),
        },
    }
    run_id = f"g3_reconciliation_{g3['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "l2b_reconciliation" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    counts_path = run_dir / "row_waterfall_v1.parquet"
    reasons_path = run_dir / "base_exclusion_reasons_v1.parquet"
    counts.write_parquet(counts_path)
    exact_reasons.write_parquet(reasons_path)
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g3_row_reconciliation",
        "status": "passed", "created_at": utc_now().isoformat(),
        "g3_run_id": g3["run_id"], "l1_run_id": l1["run_id"],
        "gate_version": g3_current["gate_version"],
        "gate_config_sha": g3_current["gate_config_sha"],
        "gate_attestation_manifest": g3_current["gate_attestation_manifest"],
        "identity": identity,
        "input_hashes": identity["input_hashes"],
        "warmup_reconciliation": {
            "l1_dates": l1_dates, "unified_warmup_dates": warmup_dates,
            "g3_return_dates": g3_dates, "latest_without_next_label": no_next_label_dates,
            "equation": "1669 + 60 + 1 = 1730",
        },
        "row_waterfall": row,
        "waterfall_identity_passed": True,
        "outputs": {
            "row_waterfall_v1": lake.artifact_record(counts_path),
            "base_exclusion_reasons_v1": lake.artifact_record(reasons_path),
        },
    }
    return lake.write_immutable_json(manifest_path, manifest)


def bind_g3_reconciliation_to_current(lake: DataLake, reconciliation_path: Path) -> Path:
    """Make the passed immutable reconciliation discoverable from G3 _CURRENT."""
    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if reconciliation.get("status") != "passed" or reconciliation.get("g3_run_id") != current.get("run_id"):
        raise RuntimeError("G3_RECONCILIATION_BINDING_INVALID")
    updated = {
        **current,
        "reconciliation_manifest": str(reconciliation_path.relative_to(lake.root)),
        "reconciliation_sha256": file_sha256(reconciliation_path),
        "updated_at": utc_now().isoformat(),
    }
    temporary = current_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, current_path)
    return current_path
