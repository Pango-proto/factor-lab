"""Historical one-shot probe of risk loadings versus prospective economic dimensions.

Despite the legacy module name, this does not define or test an alpha factor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .research_log import SemanticProbeLedger


IMPLEMENTATION_ID = "g4_statistical_alpha_overlap_v1"


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _classification(value: float) -> str:
    if value < 0.1:
        return "alpha_dimension_largely_unabsorbed"
    if value <= 0.3:
        return "partial_absorption_record_expected_decay"
    return "material_absorption_requires_g6_neutralization_decision"


def run_g4_statistical_alpha_overlap(
    lake: DataLake, *, spectrum_manifest_path: Path,
    config_path: Path = Path("config/g4_statistical_alpha_overlap_v1.json"),
) -> Path:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("run_policy") != "exactly_once_per_risk_basis":
        raise ValueError("G4_ALPHA_OVERLAP_RUN_POLICY_INVALID")
    spectrum = json.loads(spectrum_manifest_path.read_text(encoding="utf-8"))
    if spectrum.get("g3_run_id") != protocol["g3_run_id"]:
        raise ValueError("G4_ALPHA_OVERLAP_G3_LINEAGE_MISMATCH")
    risk_basis_id = spectrum.get("risk_basis_id")
    if not risk_basis_id or not spectrum.get("risk_metric_id"):
        raise ValueError("G4_ALPHA_OVERLAP_REQUIRES_EXPLICIT_RISK_BASIS_AND_METRIC")
    if protocol.get("risk_basis_id") != risk_basis_id:
        raise ValueError("G4_ALPHA_OVERLAP_RISK_BASIS_MISMATCH")
    j_stable = int(spectrum["headline"]["J_stable"])
    loadings_path = lake.root / spectrum["outputs"][
        "statistical_factor_stable_loadings_v1"
    ]["path"]
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    financial_path = lake.silver / "financial_pit" / "data.parquet"
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "spectrum_run_id": spectrum["run_id"], "g3_run_id": spectrum["g3_run_id"],
        "input_hashes": {
            "protocol": file_sha256(config_path),
            "spectrum_manifest": file_sha256(spectrum_manifest_path),
            "stable_loadings": file_sha256(loadings_path),
            "valuation": file_sha256(valuation_path),
            "returns": file_sha256(returns_path),
            "financial_pit": file_sha256(financial_path),
        },
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_alpha_overlap_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_statistical_alpha_overlap" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    existing = list((lake.root / "diagnostics" / "g4_statistical_alpha_overlap").glob(
        "run_id=*/_MANIFEST.json"
    ))
    if manifest_path.exists():
        return manifest_path
    for path in existing:
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_basis = previous.get("risk_basis_id")
        if not previous_basis or not previous.get("risk_metric_id"):
            raise ValueError("G4_ALPHA_OVERLAP_LEGACY_MANIFEST_FORBIDDEN")
        if previous_basis == risk_basis_id:
            raise RuntimeError("G4_ALPHA_OVERLAP_BASIS_ALREADY_CONSUMED")
    run_dir.mkdir(parents=True, exist_ok=False)

    decision_end = protocol["decision_sample_end_inclusive"]
    loading_columns = [f"statistical_loading_{index}" for index in range(1, j_stable + 1)]
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute(f"""
          CREATE TEMP TABLE return_features AS
          WITH windows AS (
            SELECT trade_date,asset_id,
              count(total_return) FILTER(WHERE total_return>-1) OVER mom momentum_n,
              sum(CASE WHEN total_return>-1 THEN ln(1+total_return) END) OVER mom momentum_log,
              count(total_return) FILTER(WHERE total_return>-1) OVER rev reversal_n,
              sum(CASE WHEN total_return>-1 THEN ln(1+total_return) END) OVER rev reversal_log
            FROM read_parquet('{_q(returns_path)}')
            WINDOW mom AS (PARTITION BY asset_id ORDER BY trade_date ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING),
                   rev AS (PARTITION BY asset_id ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
          )
          SELECT trade_date,asset_id,
            CASE WHEN momentum_n>=200 THEN exp(momentum_log)-1 END momentum_12_1,
            CASE WHEN reversal_n>=15 THEN -(exp(reversal_log)-1) END reversal_1m
          FROM windows
        """)
        connection.execute(f"""
          CREATE TEMP TABLE asset_growth AS
          WITH endpoints AS (
            SELECT DISTINCT window_end,asset_id
            FROM read_parquet('{_q(loadings_path)}')
            WHERE within_decision_sample AND window_end<=DATE '{decision_end}'
          ), current_candidates AS (
            SELECT e.window_end,e.asset_id,f.report_period,f.total_assets,
              row_number() OVER(
                PARTITION BY e.window_end,e.asset_id
                ORDER BY f.report_period DESC,f.available_at DESC,f.revision_id DESC
              ) rn
            FROM endpoints e JOIN read_parquet('{_q(financial_path)}') f
              ON f.asset_id=e.asset_id AND CAST(f.available_at AS DATE)<=e.window_end
            WHERE f.total_assets>0
          ), current_report AS (
            SELECT * EXCLUDE(rn) FROM current_candidates WHERE rn=1
          ), prior_candidates AS (
            SELECT c.window_end,c.asset_id,c.total_assets current_assets,
                   p.total_assets prior_assets,
              row_number() OVER(
                PARTITION BY c.window_end,c.asset_id
                ORDER BY p.available_at DESC,p.revision_id DESC
              ) rn
            FROM current_report c LEFT JOIN read_parquet('{_q(financial_path)}') p
              ON p.asset_id=c.asset_id
             AND p.report_period=CAST(c.report_period-INTERVAL 1 YEAR AS DATE)
             AND CAST(p.available_at AS DATE)<=c.window_end
             AND p.total_assets>0
          )
          SELECT window_end,asset_id,
            CASE WHEN prior_assets>0 THEN current_assets/prior_assets-1 END asset_growth_yoy
          FROM prior_candidates WHERE rn=1
        """)
        loading_select = ",".join(f'l."{column}"' for column in loading_columns)
        feature_frame = connection.execute(f"""
          SELECT l.window_end,l.asset_id,{loading_select},
            CASE WHEN v.pb IS NOT NULL AND v.pb!=0 THEN 1/v.pb END book_to_price,
            CASE WHEN v.pe_ttm IS NOT NULL AND v.pe_ttm!=0 THEN 1/v.pe_ttm END earnings_to_price,
            r.momentum_12_1,r.reversal_1m,
            CASE WHEN v.pe_ttm IS NOT NULL AND v.pe_ttm!=0 THEN v.pb/v.pe_ttm END roe_ttm_implied,
            g.asset_growth_yoy
          FROM read_parquet('{_q(loadings_path)}') l
          LEFT JOIN read_parquet('{_q(valuation_path)}') v
            ON v.trade_date=l.window_end AND v.asset_id=l.asset_id
          LEFT JOIN return_features r
            ON r.trade_date=l.window_end AND r.asset_id=l.asset_id
          LEFT JOIN asset_growth g
            ON g.window_end=l.window_end AND g.asset_id=l.asset_id
          WHERE l.within_decision_sample AND l.window_end<=DATE '{decision_end}'
          ORDER BY l.window_end,l.asset_id
        """).pl()
    finally:
        connection.close()

    available_features = [
        item["id"] for item in protocol["features"] if item["availability"] == "available"
    ]
    window_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for key, daily in feature_frame.group_by("window_end", maintain_order=True):
        window_end = key[0] if isinstance(key, tuple) else key
        total = daily.height
        for feature in available_features:
            selected_columns = [*loading_columns, feature]
            valid = daily.select(selected_columns).drop_nulls().filter(
                pl.all_horizontal(
                    [pl.col(column).is_finite() for column in selected_columns]
                )
            )
            if valid.height <= j_stable + 2:
                continue
            x = valid.select(loading_columns).to_numpy()
            y = valid[feature].to_numpy()
            design = np.column_stack([np.ones(valid.height), x])
            fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            total_sum = float(np.sum(np.square(y - y.mean())))
            r_squared = 1.0 - float(np.sum(np.square(y - fitted))) / total_sum if total_sum > 0 else 0.0
            window_rows.append({
                "window_end": window_end, "feature_id": feature,
                "n_observations": valid.height, "loading_universe_count": total,
                "coverage": valid.height / total, "subspace_r_squared": r_squared,
            })
            for column_index, column in enumerate(loading_columns):
                correlation = float(np.corrcoef(x[:, column_index], y)[0, 1])
                correlation_rows.append({
                    "window_end": window_end, "feature_id": feature,
                    "statistical_factor_id": column, "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                })

    windows = pl.DataFrame(window_rows)
    correlations = pl.DataFrame(correlation_rows)
    summary = windows.group_by("feature_id").agg(
        pl.len().alias("evaluation_windows"),
        pl.col("coverage").median().alias("median_coverage"),
        pl.col("subspace_r_squared").median().alias("median_subspace_r_squared"),
        pl.col("subspace_r_squared").max().alias("maximum_subspace_r_squared"),
    ).with_columns(
        pl.col("median_subspace_r_squared").map_elements(
            _classification, return_dtype=pl.String
        ).alias("classification")
    ).sort("median_subspace_r_squared", descending=True)
    correlation_summary = correlations.group_by(
        "feature_id", "statistical_factor_id"
    ).agg(
        pl.col("absolute_correlation").median().alias("median_absolute_correlation"),
        pl.col("absolute_correlation").max().alias("maximum_absolute_correlation"),
    ).sort("feature_id", "statistical_factor_id")
    availability = pl.DataFrame(protocol["features"]).select(
        "id", "source", "availability", "formula"
    ).rename({"id": "feature_id"})
    outputs: dict[str, Any] = {}
    for name, frame in {
        "alpha_overlap_window_v1": windows,
        "alpha_overlap_summary_v1": summary,
        "alpha_overlap_factor_correlation_v1": correlation_summary,
        "alpha_feature_availability_v1": availability,
    }.items():
        path = run_dir / f"{name}.parquet"
        _write(path, frame)
        outputs[name] = lake.artifact_record(path)
    ledger = SemanticProbeLedger(lake.metadata / "research_protocol.sqlite")
    cumulative_count_before = ledger.count()
    basis_count_before = ledger.count_for_basis(risk_basis_id)
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "g4_statistical_alpha_overlap_once", "status": "passed",
        "created_at": utc_now().isoformat(), "identity": identity,
        "decision_sample_end_inclusive": decision_end,
        "maximum_evaluated_window_end": str(feature_frame["window_end"].max()),
        "statistical_factor_count": j_stable, "protocol": protocol,
        "risk_basis_id": risk_basis_id,
        "risk_metric_id": spectrum["risk_metric_id"],
        "run_count_authorized": 1, "return_label_metrics_produced": False,
        "semantic_probe_accounting": {
            "cumulative_cross_basis_count_before_run": cumulative_count_before,
            "cumulative_cross_basis_count_after_run": cumulative_count_before + 1,
            "current_basis_count_before_run": basis_count_before,
            "current_basis_count_after_run": basis_count_before + 1,
            "included_in_l2a_bh_denominator": False,
            "exclusion_rationale": "The probe tests whether the frozen risk representation leaves semantic structure in exposures; it makes no assertion that a factor predicts returns.",
        },
        "headline": {
            "materially_absorbed_features": summary.filter(
                pl.col("median_subspace_r_squared") > 0.3
            )["feature_id"].to_list(),
            "partially_absorbed_features": summary.filter(
                pl.col("median_subspace_r_squared").is_between(0.1, 0.3, closed="both")
            )["feature_id"].to_list(),
            "largely_unabsorbed_features": summary.filter(
                pl.col("median_subspace_r_squared") < 0.1
            )["feature_id"].to_list(),
            "unavailable_features": availability.filter(
                pl.col("availability") != "available"
            )["feature_id"].to_list(),
        },
        "outputs": outputs,
    }
    written = lake.write_immutable_json(manifest_path, manifest)
    ledger.record(
        event_id=run_id, risk_basis_id=risk_basis_id,
        descriptor_manifest_sha=json_hash(protocol["features"]),
        protocol_sha=file_sha256(config_path), created_at=manifest["created_at"],
    )
    return written
