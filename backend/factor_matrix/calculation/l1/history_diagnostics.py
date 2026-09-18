"""Read-only G2 history diagnostics and the explicit L1 promotion gate."""

from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now


STYLE_COLUMNS = (
    "risk_size", "risk_beta", "risk_residual_volatility", "risk_liquidity",
    "risk_nonlinear_size", "risk_listing_age",
)
AUTOCORRELATION_LAGS = (1, 5, 21)
ORTHOGONALITY_TOLERANCE = 1e-8
STAR_START = date(2019, 7, 22)
BSE_START = date(2021, 11, 15)

STYLE_PREDECESSORS = {
    "risk_size": (),
    "risk_beta": ("risk_size",),
    "risk_residual_volatility": ("risk_size", "risk_beta"),
    "risk_liquidity": ("risk_size",),
    "risk_nonlinear_size": ("risk_size",),
    "risk_listing_age": ("risk_size",),
}


def _weighted_corr(left: list[float], right: list[float], weights: list[float]) -> float:
    total = sum(weights)
    left_mean = sum(w * x for w, x in zip(weights, left)) / total
    right_mean = sum(w * x for w, x in zip(weights, right)) / total
    covariance = sum(
        w * (x - left_mean) * (y - right_mean)
        for w, x, y in zip(weights, left, right)
    )
    left_ss = sum(w * (x - left_mean) ** 2 for w, x in zip(weights, left))
    right_ss = sum(w * (y - right_mean) ** 2 for w, y in zip(weights, right))
    denominator = math.sqrt(left_ss * right_ss)
    return covariance / denominator if denominator > 0 else 0.0


def _independent_metric_orthogonality(
    lake: DataLake, history: dict[str, Any], exposure_path: Path, *, start: date,
) -> pl.DataFrame:
    contract = history.get("weight_metric_contract")
    if not contract:
        raise RuntimeError("G2_WEIGHT_METRIC_CONTRACT_MISSING")
    weight_path = lake.root / contract["artifact_path"]
    if file_sha256(weight_path) != contract["artifact_sha256"]:
        raise RuntimeError("G2_WEIGHT_METRIC_ARTIFACT_HASH_MISMATCH")
    exposure = pl.scan_parquet(exposure_path).filter(
        pl.col("is_valid") & (pl.col("trade_date") >= start)
    ).collect()
    weights = pl.scan_parquet(weight_path).select(
        pl.col("exposure_date").alias("trade_date"), "asset_id", "candidate_weight"
    ).collect()
    valuation = pl.scan_parquet(lake.silver / "valuation_daily" / "data.parquet").filter(
        pl.col("trade_date") >= start
    ).select("trade_date", "asset_id", "float_mkt_cap").collect()
    joined = exposure.join(weights, on=("trade_date", "asset_id"), how="left").join(
        valuation, on=("trade_date", "asset_id"), how="left", validate="m:1"
    )
    rows: list[dict[str, Any]] = []
    for key, daily in joined.partition_by("trade_date", as_dict=True).items():
        trade_date = key[0] if isinstance(key, tuple) else key
        custom_count = daily["candidate_weight"].is_not_null().sum()
        if custom_count:
            if daily["float_mkt_cap"].null_count():
                raise RuntimeError("G2_FALLBACK_MARKET_CAP_MISSING")
            metric = [
                float(weight) if weight is not None else math.sqrt(cap)
                for weight, cap in zip(
                    daily["candidate_weight"].to_list(),
                    daily["float_mkt_cap"].to_list(),
                )
            ]
            scheme_id = contract["orthogonalization_scheme_id"]
        else:
            if daily["float_mkt_cap"].null_count():
                raise RuntimeError("G2_FALLBACK_MARKET_CAP_MISSING")
            metric = [math.sqrt(value) for value in daily["float_mkt_cap"].to_list()]
            scheme_id = contract["missing_date_fallback_scheme_id"]
        industry_columns = [
            column for column in daily.columns if column.startswith("risk_industry_")
        ]
        for factor, predecessors in STYLE_PREDECESSORS.items():
            values = daily[factor].to_list()
            for control in (*industry_columns, *predecessors):
                correlation = abs(_weighted_corr(values, daily[control].to_list(), metric))
                rows.append({
                    "trade_date": trade_date, "factor_id": factor.removeprefix("risk_"),
                    "control_id": control.removeprefix("risk_"),
                    "weight_scheme_id": scheme_id, "absolute_weighted_correlation": correlation,
                })
    return pl.DataFrame(rows)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _artifact_path(lake: DataLake, manifest: dict[str, Any], key: str) -> Path:
    return lake.root / manifest["outputs"][key]["path"]


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _autocorrelation(connection) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for lag in AUTOCORRELATION_LAGS:
        aggregates = ",\n".join(
            f"corr(a.{column}, b.{column})::DOUBLE AS {column}"
            for column in STYLE_COLUMNS
        )
        wide = connection.execute(f"""
            SELECT b.trade_date, count(*)::BIGINT AS n_common, {aggregates}
            FROM exposure a
            JOIN date_index da ON da.trade_date=a.trade_date
            JOIN date_index db ON db.rn=da.rn+{lag}
            JOIN exposure b ON b.trade_date=db.trade_date AND b.asset_id=a.asset_id
            WHERE a.is_valid AND b.is_valid
            GROUP BY b.trade_date
        """).pl()
        frames.append(wide.unpivot(
            index=["trade_date", "n_common"],
            on=list(STYLE_COLUMNS), variable_name="factor_id", value_name="rho",
        ).with_columns(
            pl.col("factor_id").str.strip_prefix("risk_"),
            pl.lit(lag).cast(pl.Int32).alias("lag_days"),
        ))
    return pl.concat(frames).sort("factor_id", "lag_days", "trade_date")


def _board_industry_cross(connection) -> pl.DataFrame:
    return connection.execute("""
        WITH cells AS (
          SELECT trade_date, board_id, sw_l1_code, count(*)::BIGINT AS n_members
          FROM exposure
          GROUP BY trade_date, board_id, sw_l1_code
        ), totals AS (
          SELECT *,
                 sum(n_members) OVER (PARTITION BY trade_date, board_id) AS board_total,
                 sum(n_members) OVER (PARTITION BY trade_date, sw_l1_code) AS industry_total
          FROM cells
        )
        SELECT *, n_members::DOUBLE/board_total AS share_of_board,
               n_members::DOUBLE/industry_total AS share_of_industry,
               n_members=board_total AS fills_entire_board,
               n_members=industry_total AS fills_entire_industry,
               n_members=board_total AND n_members=industry_total AS exact_duplicate_block
        FROM totals ORDER BY trade_date, board_id, sw_l1_code
    """).pl()


def _adjacent_diff(connection) -> pl.DataFrame:
    aggregates = ",\n".join(
        [f"avg(abs(b.{column}-a.{column}))::DOUBLE AS mae_{column}" for column in STYLE_COLUMNS]
        + [f"corr(a.{column}, b.{column})::DOUBLE AS rho_{column}" for column in STYLE_COLUMNS]
    )
    wide = connection.execute(f"""
        SELECT b.trade_date, count(*)::BIGINT AS n_common, {aggregates}
        FROM exposure a
        JOIN date_index da ON da.trade_date=a.trade_date
        JOIN date_index db ON db.rn=da.rn+1
        JOIN exposure b ON b.trade_date=db.trade_date AND b.asset_id=a.asset_id
        WHERE a.is_valid AND b.is_valid
        GROUP BY b.trade_date
    """).pl()
    mae = wide.unpivot(
        index=["trade_date", "n_common"],
        on=[f"mae_{column}" for column in STYLE_COLUMNS],
        variable_name="factor_id", value_name="mean_abs_change",
    ).with_columns(pl.col("factor_id").str.strip_prefix("mae_risk_"))
    rho = wide.unpivot(
        index=["trade_date", "n_common"],
        on=[f"rho_{column}" for column in STYLE_COLUMNS],
        variable_name="factor_id", value_name="rho_1d",
    ).with_columns(pl.col("factor_id").str.strip_prefix("rho_risk_"))
    frame = mae.join(rho, on=["trade_date", "n_common", "factor_id"], how="inner")
    dates = frame.select("trade_date").unique().sort("trade_date").with_columns(
        (pl.col("trade_date").dt.month() != pl.col("trade_date").shift(1).dt.month())
        .fill_null(False).alias("is_month_boundary")
    )
    return frame.join(dates, on="trade_date", how="left").sort("trade_date", "factor_id")


def _board_counts(
    connection, *, star_model_start: date, bse_model_start: date,
) -> pl.DataFrame:
    frame = connection.execute("""
        SELECT trade_date, count(DISTINCT board_id)::INTEGER AS n_boards,
               count(*)::BIGINT AS n_rows
        FROM exposure GROUP BY trade_date ORDER BY trade_date
    """).pl()
    return _apply_expected_board_counts(
        frame, star_model_start=star_model_start, bse_model_start=bse_model_start,
    )


def _apply_expected_board_counts(
    frame: pl.DataFrame, *, star_model_start: date, bse_model_start: date,
) -> pl.DataFrame:
    return frame.select("trade_date", "n_boards", "n_rows").with_columns(
        pl.when(pl.col("trade_date") < star_model_start).then(pl.lit(2))
        .when(pl.col("trade_date") < bse_model_start).then(pl.lit(3))
        .otherwise(pl.lit(4)).alias("expected_n_boards")
    ).with_columns(
        (pl.col("n_boards") == pl.col("expected_n_boards")).alias("passed")
    )


def _model_eligible_board_start(lake: DataLake, legal_start: date, d0: int) -> date:
    calendar = pl.read_parquet(lake.silver / "trade_calendar" / "data.parquet").filter(
        (pl.col("exchange") == "SSE")
        & (pl.col("is_open") == 1)
        & (pl.col("cal_date") >= legal_start)
    ).select("cal_date").unique().sort("cal_date")
    dates = calendar["cal_date"].to_list()
    if len(dates) <= d0:
        raise RuntimeError("L1_BOARD_BOUNDARY_CALENDAR_INCOMPLETE")
    return dates[d0]


def run_l1_history_diagnostics(
    lake: DataLake, history_manifest_path: Path,
    listing_policy_path: Path = Path("config/new_listing_policy_v1.json"),
) -> Path:
    history = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    if history.get("job") != "l1_risk_exposure_history":
        raise ValueError("L1_DIAGNOSTICS_REQUIRES_HISTORY_MANIFEST")
    if history.get("universe_variant") != "frozen_d0":
        raise ValueError("L1_DIAGNOSTICS_REQUIRES_FROZEN_D0")
    exposure_path = _artifact_path(lake, history, "risk_exposure_matrix_v1")
    quality_path = _artifact_path(lake, history, "exposure_quality_v1")
    listing_policy = json.loads(listing_policy_path.read_text(encoding="utf-8"))
    d0 = int(listing_policy["d0_model_exclusion"]["value"])
    star_model_start = _model_eligible_board_start(lake, STAR_START, d0)
    bse_model_start = _model_eligible_board_start(lake, BSE_START, d0)
    reused_from: str | None = None
    reusable: dict[str, Any] | None = None
    for candidate in sorted(
        (lake.root / "diagnostics" / "l1_history").glob("run_id=*/_MANIFEST.json"),
        reverse=True,
    ):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("history_run_id") == history["run_id"]:
            reusable = payload
            break
    if reusable is not None:
        reused_from = reusable["run_id"]
        autocorrelation = pl.read_parquet(
            lake.root / reusable["outputs"]["exposure_autocorrelation_v1"]["path"]
        )
        cross = pl.read_parquet(
            lake.root / reusable["outputs"]["board_industry_cross_v1"]["path"]
        )
        adjacent = pl.read_parquet(
            lake.root / reusable["outputs"]["boundary_diff_v1"]["path"]
        )
        old_counts = pl.read_parquet(
            lake.root / reusable["outputs"]["board_count_boundaries_v1"]["path"]
        )
        board_counts = _apply_expected_board_counts(
            old_counts, star_model_start=star_model_start,
            bse_model_start=bse_model_start,
        )
    else:
        connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
        try:
            connection.execute(
                f"CREATE TEMP VIEW exposure AS SELECT * FROM read_parquet('{_sql_path(exposure_path)}')"
            )
            connection.execute("""
                CREATE TEMP TABLE date_index AS
                SELECT trade_date, row_number() OVER (ORDER BY trade_date) AS rn
                FROM (SELECT DISTINCT trade_date FROM exposure)
            """)
            autocorrelation = _autocorrelation(connection)
            cross = _board_industry_cross(connection)
            adjacent = _adjacent_diff(connection)
            board_counts = _board_counts(
                connection, star_model_start=star_model_start,
                bse_model_start=bse_model_start,
            )
        finally:
            connection.close()

    quality = pl.read_parquet(quality_path)
    dates = quality.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    burn_in = int(history["derived_params"]["burn_in_days"])
    post_burn_start = dates[burn_in] if len(dates) > burn_in else None
    post_burn = quality.filter(pl.col("trade_date") >= post_burn_start) if post_burn_start else quality.head(0)
    style_quality = post_burn.filter(pl.col("role") == "style_risk")
    if post_burn_start is None:
        raise RuntimeError("G2_POST_BURN_DOMAIN_EMPTY")
    independent_orthogonality = _independent_metric_orthogonality(
        lake, history, exposure_path, start=post_burn_start,
    )
    max_orthogonality = independent_orthogonality[
        "absolute_weighted_correlation"
    ].max()
    failed_post_burn_quality = style_quality.filter(pl.col("status") != "passed").height

    daily_change = adjacent.group_by("trade_date", "is_month_boundary").agg(
        pl.col("mean_abs_change").mean().alias("mean_abs_change")
    )
    non_boundary = daily_change.filter(~pl.col("is_month_boundary"))["mean_abs_change"]
    boundary = daily_change.filter(pl.col("is_month_boundary"))["mean_abs_change"]
    non_boundary_q99 = float(non_boundary.quantile(0.99, interpolation="nearest"))
    boundary_median = float(boundary.median())
    month_boundary_passed = boundary_median <= non_boundary_q99
    board_boundary_passed = bool(board_counts["passed"].all())
    post_burn_invalid = int(history["counts"]["post_burn_invalid_dates"])
    orthogonality_passed = (
        max_orthogonality is not None
        and float(max_orthogonality) <= ORTHOGONALITY_TOLERANCE
        and failed_post_burn_quality == 0
    )
    quality_summary = post_burn.group_by("factor_id", "role").agg(
        pl.col("coverage_ratio").min().alias("minimum_coverage_ratio"),
        pl.col("coverage_ratio").median().alias("median_coverage_ratio"),
        pl.col("n_excluded").max().alias("maximum_excluded"),
        (
            (pl.col("n_winsorized_low") + pl.col("n_winsorized_high"))
            / pl.when(pl.col("n_valid") > 0).then(pl.col("n_valid")).otherwise(None)
        ).max().alias("maximum_winsorized_ratio"),
        pl.col("min_category_members").min().alias("minimum_category_members"),
    ).sort("factor_id").to_dicts()
    status = "passed" if (
        board_boundary_passed and month_boundary_passed
        and post_burn_invalid == 0 and orthogonality_passed
    ) else "failed"

    identity = {
        "history_run_id": history["run_id"],
        "history_sha256": file_sha256(history_manifest_path),
        "code_sha": source_tree_hash(),
        "listing_policy_sha256": file_sha256(listing_policy_path),
    }
    run_id = f"l1_history_diagnostics_{history['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "l1_history" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    autocorrelation_path = run_dir / "exposure_autocorrelation_v1.parquet"
    cross_path = run_dir / "board_industry_cross_v1.parquet"
    boundary_path = run_dir / "boundary_diff_v1.parquet"
    board_counts_path = run_dir / "board_count_boundaries_v1.parquet"
    orthogonality_path = run_dir / "independent_metric_orthogonality_v1.parquet"
    _write_parquet(autocorrelation_path, autocorrelation)
    _write_parquet(cross_path, cross)
    _write_parquet(boundary_path, adjacent)
    _write_parquet(board_counts_path, board_counts)
    _write_parquet(orthogonality_path, independent_orthogonality)

    rho_summary = autocorrelation.group_by("factor_id", "lag_days").agg(
        pl.col("rho").mean().alias("mean_rho"),
        pl.col("rho").median().alias("median_rho"),
        pl.len().alias("n_pairs"),
    ).sort("factor_id", "lag_days").to_dicts()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "l1_history_diagnostics",
        "status": status,
        "created_at": utc_now().isoformat(),
        "history_run_id": history["run_id"],
        "history_manifest": str(history_manifest_path.resolve()),
        "history_manifest_sha256": identity["history_sha256"],
        "silver_version_id": history["silver_version_id"],
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "universe_variant": history["universe_variant"],
        "code_sha": identity["code_sha"],
        "reused_computation_from": reused_from,
        "checks": {
            "post_burn_invalid_dates": {"observed": post_burn_invalid, "passed": post_burn_invalid == 0},
            "post_burn_style_quality": {"failed_rows": failed_post_burn_quality, "passed": failed_post_burn_quality == 0},
            "weighted_orthogonality": {
                "maximum": max_orthogonality,
                "tolerance": ORTHOGONALITY_TOLERANCE,
                "passed": orthogonality_passed,
                "metric_source": "history.weight_metric_contract.regression_base_weight",
                "independently_recomputed": True,
            },
            "board_regime_boundaries": {
                "d0": d0,
                "star_legal_start": STAR_START.isoformat(),
                "star_model_start": star_model_start.isoformat(),
                "bse_legal_start": BSE_START.isoformat(),
                "bse_model_start": bse_model_start.isoformat(),
                "rule": "legal_start shifted by d0 open trading days; empty categories removed",
                "failed_dates": board_counts.filter(~pl.col("passed")).height,
                "passed": board_boundary_passed,
            },
            "month_partition_boundary": {"boundary_median": boundary_median, "non_boundary_q99": non_boundary_q99, "rule": "boundary_median<=non_boundary_q99", "passed": month_boundary_passed},
            "exact_board_industry_duplicate_blocks": {"observed": cross.filter(pl.col("exact_duplicate_block")).height, "gate": "reported_for_G3_not_a_G2_failure"},
        },
        "autocorrelation_summary": rho_summary,
        "quality_time_series_source": history["outputs"]["exposure_quality_v1"],
        "post_burn_quality_summary": quality_summary,
        "outputs": {
            "exposure_autocorrelation_v1": lake.artifact_record(autocorrelation_path),
            "board_industry_cross_v1": lake.artifact_record(cross_path),
            "boundary_diff_v1": lake.artifact_record(boundary_path),
            "board_count_boundaries_v1": lake.artifact_record(board_counts_path),
            "independent_metric_orthogonality_v1": lake.artifact_record(orthogonality_path),
        },
        "promotion_gate": "passed" if status == "passed" else "blocked",
    }
    return lake.write_immutable_json(manifest_path, manifest)


def promote_l1_history(
    lake: DataLake, history_manifest_path: Path, diagnostics_manifest_path: Path,
) -> Path:
    history = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_manifest_path.read_text(encoding="utf-8"))
    if history.get("universe_variant") != "frozen_d0":
        raise RuntimeError("L1_PROMOTION_REQUIRES_FROZEN_D0")
    if diagnostics.get("status") != "passed" or diagnostics.get("promotion_gate") != "passed":
        raise RuntimeError("L1_HISTORY_DIAGNOSTICS_NOT_PASSED")
    if diagnostics.get("history_run_id") != history.get("run_id"):
        raise RuntimeError("L1_HISTORY_DIAGNOSTICS_LINEAGE_MISMATCH")
    if diagnostics.get("history_manifest_sha256") != file_sha256(history_manifest_path):
        raise RuntimeError("L1_HISTORY_MANIFEST_CHANGED_AFTER_DIAGNOSTICS")
    current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    temporary = current_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "run_id": history["run_id"],
        "manifest": str(history_manifest_path.resolve().relative_to(lake.root)),
        "diagnostics_run_id": diagnostics["run_id"],
        "diagnostics_manifest": str(diagnostics_manifest_path.resolve().relative_to(lake.root)),
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "universe_variant": "frozen_d0",
        "updated_at": utc_now().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, current_path)
    return current_path
