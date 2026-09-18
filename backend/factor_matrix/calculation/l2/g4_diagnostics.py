"""Immutable G4.0 diagnostics over the corrected full-label G3 residual product."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from ...research_protocol import ResearchProtocol
from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .config import load_return_decomposition_config
from .contracts import RegressionMode
from .design_matrix import build_daily_categorical_constrained_design
from .g3_runner import _active_factor_columns, load_current_g3_manifest
from .return_decomposition import decompose_cross_section


DEFAULT_CONFIG = Path("config/g4_validation_protocol_v1.json")
DEFAULT_EXTERNAL_FACTS = Path("config/external_facts_v1.json")
INCREMENTAL_R2_IMPLEMENTATION_ID = "full_label_canonical_wls_incremental_r2_v1"


def _current_base_weight_path(lake: DataLake, g3: dict[str, Any]) -> Path | None:
    if g3.get("base_weight_scheme_id", "sqrt_cap") == "sqrt_cap":
        return None

    # The selected-weight freeze manifest is intentionally an input-only
    # artifact and may not have a published G3 pointer.  Resolve the current
    # weight through the immutable contract and the G3 input hash first.
    contract_path = lake.root.parent / "config" / "weight_metric_contract_v1.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        base = contract.get("regression_base_weight", {})
        artifact_path = lake.root / str(base.get("artifact_path", ""))
        expected_sha = base.get("artifact_sha256")
        observed_sha = g3.get("identity", {}).get("input_hashes", {}).get(
            "base_weight_artifact"
        )
        if (
            base.get("scheme_id") == g3.get("base_weight_scheme_id")
            and artifact_path.exists()
            and expected_sha
            and expected_sha == observed_sha
            and file_sha256(artifact_path) == expected_sha
        ):
            return artifact_path

    # Backward-compatible fallback for older freeze manifests that explicitly
    # recorded the resulting G3 run.
    for path in (lake.root / "diagnostics" / "g4_wls_freeze").glob(
        "run_id=*/_MANIFEST.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("new_g3_run_id") == g3["run_id"]:
            return lake.root / payload["outputs"]["selected_wls_weights_v1"]["path"]
    raise RuntimeError("G4_CURRENT_BASE_WEIGHT_ARTIFACT_NOT_FOUND")


def _incremental_r2_frame(
    connection: Any, *, risk_columns: list[str], base_weight_path: Path | None,
    decomposition_config: Any, minimum_category_members: int,
) -> pl.DataFrame:
    baseline_columns = [
        column for column in risk_columns
        if column == "risk_country"
        or column.startswith(("risk_industry_", "risk_board_"))
    ]
    exposure_select = ",".join(f'e."{column}"' for column in baseline_columns)
    weight_join = ""
    weight_select = ""
    if base_weight_path is not None:
        escaped = _sql_path(base_weight_path)
        weight_join = (
            f" JOIN read_parquet('{escaped}') w ON w.exposure_date=s.exposure_date"
            " AND w.asset_id=s.asset_id"
        )
        weight_select = ",w.candidate_weight"
    reader = connection.execute(f"""
        SELECT s.trade_date,s.exposure_date,s.asset_id,r.total_return realized_return,
               s.in_estimation_domain,v.float_mkt_cap{weight_select},{exposure_select}
        FROM specific s
        JOIN exposure e ON e.trade_date=s.exposure_date AND e.asset_id=s.asset_id
        JOIN valuation v ON v.trade_date=s.exposure_date AND v.asset_id=s.asset_id
        JOIN returns r ON r.trade_date=s.trade_date AND r.asset_id=s.asset_id
        {weight_join}
        WHERE s.specific_return IS NOT NULL
        ORDER BY s.exposure_date,s.asset_id
    """).fetch_record_batch(200_000)
    carry: pl.DataFrame | None = None
    rows: list[dict[str, Any]] = []

    def process(frame: pl.DataFrame) -> None:
        for key, daily in frame.group_by("exposure_date", maintain_order=True):
            exposure_date = key[0] if isinstance(key, tuple) else key
            daily = daily.with_columns(pl.lit(True).alias("model_eligible"))
            active, blocks = _active_factor_columns(daily, baseline_columns)
            design = build_daily_categorical_constrained_design(
                daily, exposure_columns=active,
                family_by_column={column: "risk" for column in active},
                categorical_blocks=blocks,
                minimum_category_members=minimum_category_members,
                estimation_column="in_estimation_domain",
                base_weight_column=(
                    "candidate_weight" if base_weight_path is not None else None
                ),
            )
            result = decompose_cross_section(
                design.regression_input, mode=RegressionMode.RISK_ONLY,
                config=decomposition_config,
            )
            rows.append({
                "trade_date": daily["trade_date"][0], "exposure_date": exposure_date,
                "baseline_weighted_r_squared": result.r_squared,
                "baseline_unweighted_r_squared": result.unweighted_r_squared,
            })

    for batch in reader:
        frame = pl.from_arrow(batch)
        if carry is not None:
            frame = pl.concat([carry, frame])
        last_date = frame["exposure_date"][-1]
        complete = frame.filter(pl.col("exposure_date") != last_date)
        carry = frame.filter(pl.col("exposure_date") == last_date)
        if not complete.is_empty():
            process(complete)
    if carry is not None and not carry.is_empty():
        process(carry)
    return pl.DataFrame(rows)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _event_annotations(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for fact in payload.get("facts", []):
        value = fact.get("value") or {}
        event = value.get("event")
        if event:
            result.setdefault(fact["effective_from"], []).append(str(event))
    return result


def run_g4_attribution_diagnostics(
    lake: DataLake, *, config_path: Path = DEFAULT_CONFIG,
    external_facts_path: Path = DEFAULT_EXTERNAL_FACTS,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("G4_PROTOCOL_SCHEMA_UNSUPPORTED")
    diagnostic = config["g4_0_attribution_diagnostics"]
    if (
        diagnostic["incremental_r2"].get("implementation_id")
        != INCREMENTAL_R2_IMPLEMENTATION_ID
    ):
        raise ValueError("G4_INCREMENTAL_R2_IMPLEMENTATION_CONFIG_MISMATCH")
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    l1_current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    l1_current = json.loads(l1_current_path.read_text(encoding="utf-8"))
    l1_manifest_path = lake.root / l1_current["manifest"]
    l1 = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    if g3["l1_run_id"] != l1["run_id"]:
        raise RuntimeError("G4_REQUIRES_MATCHING_G3_L1_CURRENT")
    quality_path = lake.root / g3["outputs"]["factor_regression_quality_v1"]["path"]
    specific_path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    exposure_path = lake.root / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    base_weight_path = _current_base_weight_path(lake, g3)
    exposure_schema = pl.read_parquet_schema(exposure_path)
    risk_columns = [
        name for name, dtype in exposure_schema.items()
        if name.startswith("risk_") and dtype.is_float()
    ]
    derived = dict(g3["identity"]["derived_params"])
    decomposition_config = load_return_decomposition_config(
        Path("config/l2_return_decomposition_v1.json"), derived_params=derived,
    )
    inputs = {
        "g3_manifest": g3_manifest_path,
        "l1_manifest": l1_manifest_path,
        "quality": quality_path,
        "specific_returns": specific_path,
        "risk_exposure": exposure_path,
        "valuation": valuation_path,
        "returns": returns_path,
        "protocol": config_path,
        "external_facts": external_facts_path,
    }
    if base_weight_path is not None:
        inputs["base_weight_artifact"] = base_weight_path
    identity = {
        "g3_run_id": g3["run_id"], "l1_run_id": l1["run_id"],
        "input_hashes": {name: file_sha256(path) for name, path in inputs.items()},
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_attribution_{g3['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_attribution" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise RuntimeError(f"G4_PARTIAL_RUN_CONFLICT:{run_dir}")

    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute(f"CREATE TEMP VIEW quality AS SELECT * FROM read_parquet('{_sql_path(quality_path)}')")
        connection.execute(f"CREATE TEMP VIEW specific AS SELECT * FROM read_parquet('{_sql_path(specific_path)}')")
        connection.execute(f"CREATE TEMP VIEW exposure AS SELECT * FROM read_parquet('{_sql_path(exposure_path)}')")
        connection.execute(f"CREATE TEMP VIEW valuation AS SELECT * FROM read_parquet('{_sql_path(valuation_path)}')")
        connection.execute(f"CREATE TEMP VIEW returns AS SELECT * FROM read_parquet('{_sql_path(returns_path)}')")
        daily = connection.execute("""
            SELECT trade_date, exposure_date, label_sample_count, sample_count,
                   huber_downweight_count, huber_downweight_rate,
                   full_label_r_squared, estimation_domain_r_squared,
                   full_label_unweighted_r_squared,
                   estimation_domain_unweighted_r_squared,
                   condition_number
            FROM quality ORDER BY trade_date
        """).pl()
        enriched_sql = """
            WITH joined AS (
              SELECT s.trade_date,s.exposure_date,s.asset_id,s.specific_return,
                     s.is_outlier_flagged,s.huber_weight_multiplier,
                     e.board_id,e.sw_l1_code,e.risk_liquidity,
                     e.risk_residual_volatility,v.float_mkt_cap
              FROM specific s
              JOIN exposure e ON e.trade_date=s.exposure_date AND e.asset_id=s.asset_id
              JOIN valuation v ON v.trade_date=s.exposure_date AND v.asset_id=s.asset_id
              WHERE s.in_estimation_domain AND s.specific_return IS NOT NULL
                AND v.float_mkt_cap>0
            ), ranked AS (
              SELECT *,
                ntile(10) OVER(PARTITION BY exposure_date ORDER BY float_mkt_cap) AS size_decile,
                ntile(10) OVER(PARTITION BY exposure_date ORDER BY risk_liquidity) AS liquidity_decile
              FROM joined
            )
        """
        groups = connection.execute(enriched_sql + """
            , stacked AS (
              SELECT 'board' AS dimension, CAST(board_id AS VARCHAR) AS group_id,
                     specific_return, is_outlier_flagged
              FROM ranked
              UNION ALL
              SELECT 'market_cap_decile', CAST(size_decile AS VARCHAR),
                     specific_return, is_outlier_flagged
              FROM ranked
              UNION ALL
              SELECT 'industry_l1', CAST(sw_l1_code AS VARCHAR),
                     specific_return, is_outlier_flagged
              FROM ranked
              UNION ALL
              SELECT 'liquidity_decile', CAST(liquidity_decile AS VARCHAR),
                     specific_return, is_outlier_flagged
              FROM ranked
            )
            SELECT dimension,group_id,count(*) n_observations,
                   count(*) FILTER(WHERE is_outlier_flagged) n_downweighted,
                   avg(is_outlier_flagged::INTEGER) downweight_rate,
                   avg(abs(specific_return)) FILTER(WHERE is_outlier_flagged) mean_abs_u_downweighted,
                   avg(abs(specific_return)) FILTER(WHERE NOT is_outlier_flagged) mean_abs_u_not_downweighted
            FROM stacked WHERE group_id IS NOT NULL
            GROUP BY dimension,group_id ORDER BY dimension,group_id
        """).pl()
        residual_tail = connection.execute(enriched_sql + """
            SELECT is_outlier_flagged,count(*) n_observations,
                   quantile_cont(abs(specific_return),0.5) p50_abs_u,
                   quantile_cont(abs(specific_return),0.9) p90_abs_u,
                   quantile_cont(abs(specific_return),0.99) p99_abs_u,
                   max(abs(specific_return)) max_abs_u
            FROM ranked GROUP BY is_outlier_flagged ORDER BY is_outlier_flagged
        """).pl()
        pit_asset_standardized_tail = connection.execute(enriched_sql + """
            , trailing_rows AS (
              SELECT *,
                count(specific_return) OVER asset_history AS n_prior,
                quantile_cont(abs(specific_return),0.5) OVER asset_history
                  / 0.6744897501960817 AS trailing_sigma
              FROM ranked
              WINDOW asset_history AS (
                PARTITION BY asset_id ORDER BY exposure_date
                ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING
              )
            ), valid AS (
              SELECT *,specific_return/trailing_sigma AS standardized_u
              FROM trailing_rows
              WHERE n_prior>=60 AND trailing_sigma>0
            ), stacked AS (
              SELECT 'overall' AS dimension,'all' AS group_id,standardized_u FROM valid
              UNION ALL
              SELECT 'liquidity_decile',CAST(liquidity_decile AS VARCHAR),standardized_u
              FROM valid
            ), moments AS (
              SELECT dimension,group_id,count(*) n_observations,
                     avg(standardized_u) mean_standardized_u,
                     var_pop(standardized_u) variance_standardized_u
              FROM stacked GROUP BY dimension,group_id
            )
            SELECT s.dimension,s.group_id,m.n_observations,
                   m.mean_standardized_u,m.variance_standardized_u,
                   avg(pow(s.standardized_u-m.mean_standardized_u,4))
                     /pow(m.variance_standardized_u,2)-3 AS excess_kurtosis_standardized_u
            FROM stacked s JOIN moments m USING(dimension,group_id)
            GROUP BY s.dimension,s.group_id,m.n_observations,
                     m.mean_standardized_u,m.variance_standardized_u
            ORDER BY s.dimension,s.group_id
        """).pl()
        liquidity_shape_daily = connection.execute(enriched_sql + """
            , moments AS (
              SELECT exposure_date,liquidity_decile,avg(specific_return) mean_u
              FROM ranked GROUP BY exposure_date,liquidity_decile
            )
            SELECT r.exposure_date,r.liquidity_decile,count(*) n_observations,
                   m.mean_u,
                   var_pop(r.specific_return) variance_u,
                   avg(pow(r.specific_return-m.mean_u,4))/pow(var_pop(r.specific_return),2)-3
                     excess_kurtosis_u,
                   avg(r.is_outlier_flagged::INTEGER) downweight_rate
            FROM ranked r JOIN moments m USING(exposure_date,liquidity_decile)
            GROUP BY r.exposure_date,r.liquidity_decile,m.mean_u
            ORDER BY r.exposure_date,r.liquidity_decile
        """).pl()
        liquidity_shape = liquidity_shape_daily.group_by("liquidity_decile").agg(
            pl.col("n_observations").sum(),
            pl.col("variance_u").median().alias("median_daily_variance_u"),
            pl.col("excess_kurtosis_u").median().alias("median_daily_excess_kurtosis_u"),
            pl.col("downweight_rate").median().alias("median_daily_downweight_rate"),
        ).sort("liquidity_decile")
        liquidity_resid_vol_corr = connection.execute(enriched_sql + """
            SELECT exposure_date,
                   corr(risk_liquidity,risk_residual_volatility) liquidity_resid_vol_corr
            FROM ranked GROUP BY exposure_date ORDER BY exposure_date
        """).pl()
        wls_deciles = connection.execute(enriched_sql + """
            SELECT exposure_date,size_decile,count(*) n_observations,
                   avg(ln(float_mkt_cap)) mean_log_float_mkt_cap,
                   sqrt(avg(specific_return*specific_return)) residual_rms,
                   ln(sqrt(avg(specific_return*specific_return))) log_residual_rms
            FROM ranked GROUP BY exposure_date,size_decile
            ORDER BY exposure_date,size_decile
        """).pl()
        wls_slopes = connection.execute(enriched_sql + """
            , deciles AS (
              SELECT exposure_date,size_decile,avg(ln(float_mkt_cap)) x,
                     ln(sqrt(avg(specific_return*specific_return))) y
              FROM ranked GROUP BY exposure_date,size_decile
            )
            SELECT exposure_date,regr_slope(y,x) residual_volatility_cap_slope,
                   regr_r2(y,x) regression_r_squared
            FROM deciles GROUP BY exposure_date ORDER BY exposure_date
        """).pl()
        incremental_r2 = _incremental_r2_frame(
            connection, risk_columns=risk_columns, base_weight_path=base_weight_path,
            decomposition_config=decomposition_config,
            minimum_category_members=int(derived["minimum_category_members"]),
        )
    finally:
        connection.close()

    incremental_r2 = incremental_r2.join(
        daily.select(
            "trade_date", pl.col("full_label_r_squared").alias("full_weighted_r_squared"),
            pl.col("full_label_unweighted_r_squared").alias("full_unweighted_r_squared"),
        ), on="trade_date", how="inner", validate="1:1",
    ).with_columns(
        (pl.col("full_weighted_r_squared") - pl.col("baseline_weighted_r_squared"))
        .alias("incremental_weighted_r_squared"),
        (pl.col("full_unweighted_r_squared") - pl.col("baseline_unweighted_r_squared"))
        .alias("incremental_unweighted_r_squared"),
        pl.lit(True).alias("non_decisional"),
    )

    corr = float(daily.select(pl.corr("huber_downweight_rate", "full_label_r_squared")).item())
    quantile_values = diagnostic["downweight_quantiles"]
    summary_rows = [{
        "metric": "downweight_rate", "statistic": f"q{value:g}",
        "value": float(daily["huber_downweight_rate"].quantile(float(value))),
        "non_decisional": True,
    } for value in quantile_values]
    summary_rows.extend([
        {"metric": "downweight_vs_full_r_squared", "statistic": "pearson_corr", "value": corr, "non_decisional": True},
        {"metric": "full_label_r_squared", "statistic": "mean", "value": float(daily["full_label_r_squared"].mean()), "non_decisional": True},
        {"metric": "full_label_unweighted_r_squared", "statistic": "mean", "value": float(daily["full_label_unweighted_r_squared"].mean()), "non_decisional": True},
        {"metric": "weighted_minus_unweighted_r_squared", "statistic": "mean", "value": float((daily["full_label_r_squared"] - daily["full_label_unweighted_r_squared"]).mean()), "non_decisional": True},
        {"metric": "wls_cap_exponent", "statistic": "median_daily_slope", "value": float(wls_slopes["residual_volatility_cap_slope"].median()), "non_decisional": True},
        {"metric": "wls_cap_exponent", "statistic": "mean_daily_slope", "value": float(wls_slopes["residual_volatility_cap_slope"].mean()), "non_decisional": True},
    ])
    summary = pl.DataFrame(summary_rows)
    annotations = _event_annotations(external_facts_path)
    top_days = daily.sort("huber_downweight_rate", descending=True).head(
        int(diagnostic["top_day_count"])
    ).with_columns(
        pl.col("trade_date").map_elements(
            lambda value: "|".join(annotations.get(value.isoformat(), [])) or None,
            return_dtype=pl.String,
        ).alias("registered_event_annotation")
    )
    outputs = {
        "daily_diagnostics_v1": daily,
        "summary_v1": summary,
        "top_downweight_days_v1": top_days,
        "downweight_group_distribution_v1": groups,
        "residual_tail_comparison_v1": residual_tail,
        "pit_asset_standardized_tail_v1": pit_asset_standardized_tail,
        "liquidity_residual_shape_v1": liquidity_shape,
        "liquidity_residual_shape_daily_v1": liquidity_shape_daily,
        "liquidity_resid_vol_correlation_daily_v1": liquidity_resid_vol_corr,
        "wls_cap_deciles_v1": wls_deciles,
        "wls_cap_slope_daily_v1": wls_slopes,
        "incremental_r_squared_daily_v1": incremental_r2,
    }
    output_records: dict[str, Any] = {}
    for name, frame in outputs.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        output_records[name] = lake.artifact_record(path)
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g4_0_attribution_diagnostics",
        "status": "passed", "created_at": utc_now().isoformat(),
        "g3_run_id": g3["run_id"], "l1_run_id": l1["run_id"],
        "identity": identity, "protocol": config,
        "headline": {
            "downweight_r_squared_correlation": corr,
            "mean_full_label_r_squared": float(daily["full_label_r_squared"].mean()),
            "mean_full_label_unweighted_r_squared": float(
                daily["full_label_unweighted_r_squared"].mean()
            ),
            "mean_weighted_minus_unweighted_r_squared": float(
                (daily["full_label_r_squared"] - daily["full_label_unweighted_r_squared"]).mean()
            ),
            "median_downweight_rate": float(daily["huber_downweight_rate"].median()),
            "p99_downweight_rate": float(daily["huber_downweight_rate"].quantile(0.99)),
            "maximum_downweight_rate": float(daily["huber_downweight_rate"].max()),
            "median_daily_wls_cap_slope": float(wls_slopes["residual_volatility_cap_slope"].median()),
            "median_daily_liquidity_resid_vol_correlation": float(
                liquidity_resid_vol_corr["liquidity_resid_vol_corr"].median()
            ),
            "mean_incremental_weighted_r_squared": float(
                incremental_r2["incremental_weighted_r_squared"].mean()
            ),
            "mean_incremental_unweighted_r_squared": float(
                incremental_r2["incremental_unweighted_r_squared"].mean()
            ),
        },
        "outputs": output_records,
    }
    return lake.write_immutable_json(manifest_path, manifest)
