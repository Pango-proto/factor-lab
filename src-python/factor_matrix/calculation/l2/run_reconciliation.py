"""Reusable immutable numeric reconciliation between two G3 history runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now


DEFAULT_CONFIG = Path("config/g3_run_reconciliation_v1.json")


def _quoted(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _load_g3_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("job") != "g3_l2b_risk_only_history":
        raise ValueError(f"RECONCILIATION_G3_HISTORY_MANIFEST_INVALID:{path}")
    return payload


def run_g3_run_reconciliation(
    lake: DataLake, *, old_manifest_path: Path, new_manifest_path: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> Path:
    old = _load_g3_manifest(old_manifest_path)
    new = _load_g3_manifest(new_manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("RECONCILIATION_CONFIG_SCHEMA_UNSUPPORTED")
    if old["l1_run_id"] != new["l1_run_id"]:
        raise ValueError("RECONCILIATION_L1_LINEAGE_MISMATCH")

    def output(manifest: dict[str, Any], name: str) -> Path:
        return lake.root / manifest["outputs"][name]["path"]

    old_factor, new_factor = output(old, "factor_returns_v1"), output(new, "factor_returns_v1")
    old_specific, new_specific = output(old, "specific_returns_v1"), output(new, "specific_returns_v1")
    old_quality, new_quality = output(old, "factor_regression_quality_v1"), output(new, "factor_regression_quality_v1")
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        factor_join = f"""
          SELECT n.trade_date,n.factor_id,n.factor_return new_value,o.factor_return old_value
          FROM read_parquet('{_quoted(new_factor)}') n
          JOIN read_parquet('{_quoted(old_factor)}') o
            USING(trade_date,factor_id,regression_mode)
          WHERE n.factor_return IS NOT NULL AND o.factor_return IS NOT NULL
        """
        factor_detail = connection.execute(f"""
          WITH joined AS ({factor_join})
          SELECT factor_id,count(*) n_observations,corr(new_value,old_value) correlation,
                 sqrt(avg(pow(new_value-old_value,2))) rmse,
                 max(abs(new_value-old_value)) maximum_absolute_change
          FROM joined GROUP BY factor_id ORDER BY correlation NULLS FIRST,factor_id
        """).pl()
        overall_factor = connection.execute(f"""
          WITH joined AS ({factor_join})
          SELECT count(*) n_observations,corr(new_value,old_value) correlation,
                 sqrt(avg(pow(new_value-old_value,2))) rmse,
                 max(abs(new_value-old_value)) maximum_absolute_change
          FROM joined
        """).pl().row(0, named=True)
        specific = connection.execute(f"""
          SELECT count(*) n_observations,
                 corr(n.specific_return,o.specific_return) correlation,
                 sqrt(avg(pow(n.specific_return-o.specific_return,2))) rmse,
                 max(abs(n.specific_return-o.specific_return)) maximum_absolute_change,
                 sum((n.is_outlier_flagged!=o.is_outlier_flagged)::INTEGER)
                   downweight_flag_change_count
          FROM read_parquet('{_quoted(new_specific)}') n
          JOIN read_parquet('{_quoted(old_specific)}') o
            USING(trade_date,asset_id,regression_mode)
          WHERE n.specific_return IS NOT NULL AND o.specific_return IS NOT NULL
        """).pl().row(0, named=True)
        quality = connection.execute(f"""
          SELECT count(*) n_dates,
                 avg(n.full_label_r_squared-o.full_label_r_squared) mean_r_squared_change,
                 max(abs(n.full_label_r_squared-o.full_label_r_squared)) maximum_absolute_r_squared_change,
                 avg(n.huber_downweight_rate-o.huber_downweight_rate) mean_downweight_rate_change,
                 max(n.condition_number) new_maximum_condition_number,
                 max(o.condition_number) old_maximum_condition_number
          FROM read_parquet('{_quoted(new_quality)}') n
          JOIN read_parquet('{_quoted(old_quality)}') o USING(trade_date,regression_mode)
        """).pl().row(0, named=True)
        axes = connection.execute(f"""
          SELECT
            (SELECT count(*) FROM read_parquet('{_quoted(old_factor)}')) old_factor_rows,
            (SELECT count(*) FROM read_parquet('{_quoted(new_factor)}')) new_factor_rows,
            (SELECT count(*) FROM read_parquet('{_quoted(old_specific)}')) old_specific_rows,
            (SELECT count(*) FROM read_parquet('{_quoted(new_specific)}')) new_specific_rows
        """).pl().row(0, named=True)
    finally:
        connection.close()

    factor_axis_equal = axes["old_factor_rows"] == axes["new_factor_rows"] == overall_factor["n_observations"]
    specific_axis_equal = axes["old_specific_rows"] == axes["new_specific_rows"] == specific["n_observations"]
    failing = factor_detail.filter(
        factor_detail["correlation"].is_null()
        | (factor_detail["correlation"] < float(config["minimum_factor_return_correlation"]))
    )
    reasons: list[str] = []
    if config["require_identical_factor_axis"] and not factor_axis_equal:
        reasons.append("factor_axis_changed")
    if config["require_identical_specific_return_axis"] and not specific_axis_equal:
        reasons.append("specific_return_axis_changed")
    if failing.height:
        reasons.append("factor_return_correlation_below_threshold")
    if float(specific["correlation"]) < float(config["minimum_specific_return_correlation"]):
        reasons.append("specific_return_correlation_below_threshold")
    invalidation = bool(reasons)

    identity = {
        "old_run_id": old["run_id"], "new_run_id": new["run_id"],
        "input_hashes": {
            "old_manifest": file_sha256(old_manifest_path),
            "new_manifest": file_sha256(new_manifest_path),
            "config": file_sha256(config_path),
        },
        "code_sha": source_tree_hash(),
    }
    run_id = f"g3_run_reconciliation_{new['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g3_run_reconciliation" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    factor_path = run_dir / "factor_return_reconciliation_v1.parquet"
    failing_path = run_dir / "factors_below_correlation_threshold_v1.parquet"
    factor_detail.write_parquet(factor_path)
    failing.write_parquet(failing_path)
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g3_run_reconciliation_v1",
        "status": "passed", "created_at": utc_now().isoformat(), "identity": identity,
        "old_run_id": old["run_id"], "new_run_id": new["run_id"],
        "factor_axis_equal": factor_axis_equal, "specific_return_axis_equal": specific_axis_equal,
        "overall_factor_return": overall_factor, "specific_return": specific,
        "quality_change": quality,
        "downstream_invalidation": invalidation,
        "downstream_invalidation_reasons": reasons or ["none_thresholds_satisfied"],
        "policy": config,
        "outputs": {
            "factor_return_reconciliation_v1": lake.artifact_record(factor_path),
            "factors_below_correlation_threshold_v1": lake.artifact_record(failing_path),
        },
    }
    return lake.write_immutable_json(manifest_path, manifest)
