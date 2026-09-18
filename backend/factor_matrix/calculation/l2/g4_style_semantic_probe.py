"""Exactly-once exposure-only probe of economic descriptors versus style risk columns."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .research_log import SemanticProbeLedger


IMPLEMENTATION_ID = "g4_style_loading_semantic_probe_v1"


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def run_g4_style_loading_semantic_probe(
    lake: DataLake,
    *,
    config_path: Path = Path("config/g4_style_loading_semantic_probe_v1.json"),
) -> Path:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("run_policy") != "exactly_once_per_risk_basis":
        raise ValueError("G4_STYLE_SEMANTIC_PROBE_POLICY_INVALID")
    base = lake.root / "diagnostics" / "g4_style_loading_semantic_probe"

    spectrum_manifest = (
        lake.root / "diagnostics" / "g4_statistical_spectrum"
        / f"run_id={protocol['spectrum_run_id']}" / "_MANIFEST.json"
    )
    spectrum = json.loads(spectrum_manifest.read_text(encoding="utf-8"))
    loadings_path = lake.root / spectrum["outputs"][
        "statistical_factor_stable_loadings_v1"
    ]["path"]
    l1_dir = lake.root / "gold" / "risk_exposure_matrix" / f"run_id={protocol['l1_run_id']}"
    l1_path = l1_dir / "risk_exposure_matrix_v1.parquet"
    l1_manifest = json.loads((l1_dir / "_MANIFEST.json").read_text(encoding="utf-8"))
    risk_basis_id = l1_manifest.get("risk_basis_id")
    if not risk_basis_id or not l1_manifest.get("risk_metric_id"):
        raise ValueError("G4_STYLE_SEMANTIC_PROBE_REQUIRES_EXPLICIT_RISK_BASIS_AND_METRIC")
    if protocol.get("risk_basis_id") != risk_basis_id:
        raise ValueError("G4_STYLE_SEMANTIC_PROBE_RISK_BASIS_MISMATCH")
    for path in base.glob("run_id=*/_MANIFEST.json"):
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_basis = previous.get("risk_basis_id")
        if not previous_basis or not previous.get("risk_metric_id"):
            raise ValueError("G4_STYLE_SEMANTIC_PROBE_LEGACY_MANIFEST_FORBIDDEN")
        if previous_basis == risk_basis_id:
            raise RuntimeError("G4_STYLE_SEMANTIC_PROBE_BASIS_ALREADY_CONSUMED")
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    financial_path = lake.silver / "financial_pit" / "data.parquet"
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "input_hashes": {
            "protocol": file_sha256(config_path),
            "spectrum_manifest": file_sha256(spectrum_manifest),
            "stable_loading_universe": file_sha256(loadings_path),
            "l1_exposures": file_sha256(l1_path),
            "returns": file_sha256(returns_path),
            "valuation": file_sha256(valuation_path),
            "financial_pit": file_sha256(financial_path),
        },
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_style_semantic_probe_{json_hash(identity)[:16]}"
    run_dir = base / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    decision_end = protocol["decision_sample_end_inclusive"]
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute(f"""
          CREATE TEMP TABLE endpoints AS
          SELECT DISTINCT window_end,asset_id
          FROM read_parquet('{_q(loadings_path)}')
          WHERE within_decision_sample AND window_end<=DATE '{decision_end}'
        """)
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
          WITH current_candidates AS (
            SELECT e.window_end,e.asset_id,f.report_period,f.total_assets,
              row_number() OVER(PARTITION BY e.window_end,e.asset_id
                ORDER BY f.report_period DESC,f.available_at DESC,f.revision_id DESC) rn
            FROM endpoints e JOIN read_parquet('{_q(financial_path)}') f
              ON f.asset_id=e.asset_id AND CAST(f.available_at AS DATE)<=e.window_end
            WHERE f.total_assets>0
          ), current_report AS (
            SELECT * EXCLUDE(rn) FROM current_candidates WHERE rn=1
          ), prior_candidates AS (
            SELECT c.window_end,c.asset_id,c.total_assets current_assets,p.total_assets prior_assets,
              row_number() OVER(PARTITION BY c.window_end,c.asset_id
                ORDER BY p.available_at DESC,p.revision_id DESC) rn
            FROM current_report c LEFT JOIN read_parquet('{_q(financial_path)}') p
              ON p.asset_id=c.asset_id
             AND p.report_period=CAST(c.report_period-INTERVAL 1 YEAR AS DATE)
             AND CAST(p.available_at AS DATE)<=c.window_end AND p.total_assets>0
          )
          SELECT window_end,asset_id,
            CASE WHEN prior_assets>0 THEN current_assets/prior_assets-1 END asset_growth_yoy
          FROM prior_candidates WHERE rn=1
        """)
        frame = connection.execute(f"""
          SELECT e.window_end,e.asset_id,v.float_mkt_cap,
            x.risk_beta,x.risk_residual_volatility,x.risk_liquidity,x.risk_listing_age,
            CASE WHEN v.pb IS NOT NULL AND v.pb!=0 THEN 1/v.pb END book_to_price,
            CASE WHEN v.pe_ttm IS NOT NULL AND v.pe_ttm!=0 THEN 1/v.pe_ttm END earnings_to_price,
            r.momentum_12_1,r.reversal_1m,
            CASE WHEN v.pe_ttm IS NOT NULL AND v.pe_ttm!=0 THEN v.pb/v.pe_ttm END roe_ttm_implied,
            g.asset_growth_yoy
          FROM endpoints e
          JOIN read_parquet('{_q(l1_path)}') x
            ON x.trade_date=e.window_end AND x.asset_id=e.asset_id AND x.is_valid
          LEFT JOIN read_parquet('{_q(valuation_path)}') v
            ON v.trade_date=e.window_end AND v.asset_id=e.asset_id
          LEFT JOIN return_features r ON r.trade_date=e.window_end AND r.asset_id=e.asset_id
          LEFT JOIN asset_growth g ON g.window_end=e.window_end AND g.asset_id=e.asset_id
          ORDER BY e.window_end,e.asset_id
        """).pl()
    finally:
        connection.close()

    rows: list[dict[str, Any]] = []
    descriptors = list(protocol["descriptors"])
    for key, daily in frame.group_by("window_end", maintain_order=True):
        window_end = key[0] if isinstance(key, tuple) else key
        for style_id, columns in protocol["style_bases"].items():
            style_column = columns[0]
            for descriptor in descriptors:
                selected = ["asset_id", "float_mkt_cap", style_column, descriptor]
                valid = daily.select(selected).drop_nulls().filter(
                    pl.all_horizontal([
                        pl.col(column).is_finite()
                        for column in ("float_mkt_cap", style_column, descriptor)
                    ])
                )
                if valid.height < 3:
                    continue
                x = valid[style_column].to_numpy()
                y = valid[descriptor].to_numpy()
                market_cap = valid["float_mkt_cap"].to_numpy()
                weights = np.sqrt(market_cap)
                contract = l1_manifest.get("weight_metric_contract")
                if contract:
                    weight_path = lake.root / contract["artifact_path"]
                    metric = pl.scan_parquet(weight_path).filter(
                        pl.col("exposure_date") == window_end
                    ).select("asset_id", "candidate_weight").collect()
                    if not metric.is_empty():
                        valid = valid.with_columns(pl.Series("_row", range(valid.height))).join(
                            metric, on="asset_id", how="left", validate="1:1"
                        ).sort("_row")
                        weights = valid.select(
                            pl.coalesce(
                                "candidate_weight", pl.col("float_mkt_cap").sqrt()
                            ).alias("resolved_weight")
                        )["resolved_weight"].to_numpy()
                weight_sum = float(weights.sum())
                x_mean = float(np.sum(weights * x) / weight_sum)
                y_mean = float(np.sum(weights * y) / weight_sum)
                numerator = float(np.sum(weights * (x - x_mean) * (y - y_mean)))
                denominator = float(np.sqrt(
                    np.sum(weights * np.square(x - x_mean))
                    * np.sum(weights * np.square(y - y_mean))
                ))
                correlation = numerator / denominator if denominator > 0 else float("nan")
                if not math.isfinite(correlation):
                    continue
                rows.append({
                    "window_end": window_end, "style_id": style_id,
                    "descriptor": descriptor, "n_observations": valid.height,
                    "coverage": valid.height / daily.height,
                    "correlation": correlation, "r_squared": correlation * correlation,
                })
    daily = pl.DataFrame(rows)
    summary = daily.group_by("style_id", "descriptor").agg(
        pl.len().alias("evaluation_windows"),
        pl.col("coverage").median().alias("median_coverage"),
        pl.col("r_squared").quantile(0.25).alias("r_squared_p25"),
        pl.col("r_squared").median().alias("r_squared_median"),
        pl.col("r_squared").quantile(0.75).alias("r_squared_p75"),
        pl.col("correlation").median().alias("median_signed_correlation"),
        pl.col("correlation").abs().median().alias("median_absolute_correlation"),
    ).sort("descriptor", "r_squared_median", descending=[False, True])
    outputs: dict[str, Any] = {}
    for name, output in {
        "style_semantic_probe_daily_v1": daily,
        "style_semantic_probe_summary_v1": summary,
    }.items():
        path = run_dir / f"{name}.parquet"
        _write(path, output)
        outputs[name] = lake.artifact_record(path)
    ledger = SemanticProbeLedger(lake.metadata / "research_protocol.sqlite")
    cumulative_count_before = ledger.count()
    basis_count_before = ledger.count_for_basis(risk_basis_id)
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "g4_style_loading_semantic_probe", "status": "passed",
        "created_at": utc_now().isoformat(), "identity": identity,
        "derived_from_run_id": protocol["descriptor_source_run_id"],
        "derived_from_spectrum_run_id": protocol["spectrum_run_id"],
        "l1_run_id": protocol["l1_run_id"], "protocol": protocol,
        "risk_basis_id": risk_basis_id,
        "risk_metric_id": l1_manifest["risk_metric_id"],
        "semantic_probe_accounting": {
            "cumulative_cross_basis_count_before_run": cumulative_count_before,
            "cumulative_cross_basis_count_after_run": cumulative_count_before + 1,
            "current_basis_count_before_run": basis_count_before,
            "current_basis_count_after_run": basis_count_before + 1,
            "included_in_l2a_bh_denominator": False,
            "exclusion_rationale": "The probe tests whether the frozen risk representation leaves semantic structure in exposures; it makes no assertion that a factor predicts returns.",
        },
        "maximum_evaluated_window_end": str(daily["window_end"].max()),
        "return_label_metrics_produced": False,
        "automatic_neutralization_decision": False,
        "outputs": outputs,
    }
    manifest_path = lake.write_immutable_json(run_dir / "_MANIFEST.json", manifest)
    ledger.record(
        event_id=run_id, risk_basis_id=risk_basis_id,
        descriptor_manifest_sha=json_hash(protocol["descriptors"]),
        protocol_sha=file_sha256(config_path), created_at=manifest["created_at"],
    )
    return manifest_path
