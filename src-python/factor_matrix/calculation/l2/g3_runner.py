"""Formal G3 risk-only regression consuming the committed L1 artifact."""

from __future__ import annotations

import json
import hashlib
import os
from bisect import bisect_left
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from ...research_protocol import ResearchProtocol
from ...revisioned_silver import SilverVersionLedger
from ...storage import DataLake, file_sha256, json_hash, utc_now
from .config import load_return_decomposition_config
from .contracts import RegressionMode
from .design_matrix import build_daily_categorical_constrained_design
from .label_policy import apply_l2b_label_policy, load_return_label_policy
from .return_decomposition import cross_section_statistics, decompose_cross_section
from .g3_gate import G3_GATE_CONFIG_PATH, G3_GATE_VERSION


def g3_calculation_code_hash() -> str:
    """Hash code that changes G3 numeric outputs, excluding orchestration/UI."""
    package_root = Path(__file__).resolve().parents[2]
    l2_root = Path(__file__).resolve().parent
    files = [
        l2_root / name for name in (
            "config.py", "contracts.py", "design_matrix.py", "g3_runner.py",
            "label_policy.py", "linear_algebra.py", "return_decomposition.py",
        )
    ]
    files.extend([
        package_root / "canonical_definitions.py",
        package_root / "research_protocol.py",
    ])
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def load_current_g3_manifest(lake: DataLake) -> tuple[dict[str, Any], Path]:
    """Load the consumable G3 current; superseded runs are audit-only."""
    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("status") != "active":
        raise RuntimeError(
            f"G3_CURRENT_NOT_CONSUMABLE status={current.get('status', 'missing')}"
        )
    manifest_path = lake.root / current["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed" or manifest.get("run_id") != current.get("run_id"):
        raise RuntimeError("G3_CURRENT_MANIFEST_NOT_PASSED_OR_MISMATCHED")
    return manifest, manifest_path


def _load_l1_lineage(lake: DataLake) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("universe_variant") != "frozen_d0":
        raise RuntimeError("G3_REQUIRES_FROZEN_D0_L1_CURRENT")
    history_path = lake.root / current["manifest"]
    diagnostics_path = lake.root / current["diagnostics_manifest"]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("status") != "passed" or diagnostics.get("history_run_id") != history.get("run_id"):
        raise RuntimeError("G3_REQUIRES_PASSED_MATCHING_L1_DIAGNOSTICS")
    if history.get("risk_factor_set_status") != "candidate":
        raise RuntimeError("G3_RISK_ONLY_EXPECTS_CANDIDATE_RISK_SET")
    version = SilverVersionLedger(lake).current()
    if version is None or version["version_id"] != history["silver_version_id"]:
        raise RuntimeError("G3_L1_SILVER_VERSION_MISMATCH")
    return history, diagnostics, history_path, diagnostics_path


def _next_return_map(
    exposure_dates: list[date], return_dates: list[date], *, start: date, end: date,
) -> pl.DataFrame:
    ordered_returns = sorted(set(return_dates))
    rows: list[dict[str, date]] = []
    for exposure_date in sorted(set(exposure_dates)):
        position = bisect_left(ordered_returns, exposure_date)
        while position < len(ordered_returns) and ordered_returns[position] <= exposure_date:
            position += 1
        if position < len(ordered_returns) and start <= ordered_returns[position] <= end:
            rows.append({"exposure_date": exposure_date, "trade_date": ordered_returns[position]})
    if not rows:
        raise ValueError("G3_NO_NEXT_DAY_LABELS_IN_RANGE")
    return pl.DataFrame(rows)


def _active_factor_columns(daily: pl.DataFrame, risk_columns: list[str]) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    eligible = daily.filter(pl.col("in_estimation_domain"))
    blocks: dict[str, tuple[str, ...]] = {}
    active_categorical: set[str] = set()
    for block_id, prefix in (("industry", "risk_industry_"), ("board", "risk_board_")):
        columns = tuple(
            column for column in risk_columns
            if column.startswith(prefix)
            and eligible.get_column(column).fill_null(0.0).abs().max() > 0
        )
        if columns:
            blocks[block_id] = columns
            active_categorical.update(columns)
    factor_columns = [
        column for column in risk_columns
        if column in active_categorical
        or (
            not column.startswith(("risk_industry_", "risk_board_", "risk_index_"))
            or eligible.get_column(column).fill_null(0.0).abs().max() > 0
        )
    ]
    return factor_columns, blocks


def run_g3_risk_only(
    lake: DataLake, *, start: date, end: date, publish_current: bool = False,
    l2_config_path: Path = Path("config/l2_return_decomposition_v1.json"),
    l1_config_path: Path = Path("config/l1_risk_exposure_v1.json"),
    protocol_path: Path = Path("config/research_protocol_v1.json"),
    _derived_params_override: dict[str, Any] | None = None,
    _input_hashes_override: dict[str, str] | None = None,
    _base_weight_artifact_path: Path | None = None,
    _base_weight_scheme_id: str = "sqrt_cap",
) -> Path:
    wall_start = perf_counter()
    if end < start:
        raise ValueError("G3_DATE_RANGE_INVALID")
    history, diagnostics, history_path, diagnostics_path = _load_l1_lineage(lake)
    metric_contract = history.get("weight_metric_contract")
    if not metric_contract or not history.get("risk_metric_id"):
        raise RuntimeError("G3_L1_WEIGHT_METRIC_CONTRACT_MISSING")
    expected_weight_path = lake.root / metric_contract["artifact_path"]
    if file_sha256(expected_weight_path) != metric_contract["artifact_sha256"]:
        raise RuntimeError("G3_WEIGHT_METRIC_ARTIFACT_HASH_MISMATCH")
    if _base_weight_artifact_path is None:
        _base_weight_artifact_path = expected_weight_path
    elif file_sha256(_base_weight_artifact_path) != metric_contract["artifact_sha256"]:
        raise RuntimeError("G3_WEIGHT_METRIC_ARTIFACT_LINEAGE_MISMATCH")
    if _base_weight_scheme_id == "sqrt_cap":
        _base_weight_scheme_id = metric_contract["regression_base_scheme_id"]
    if _base_weight_scheme_id != metric_contract["regression_base_scheme_id"]:
        raise RuntimeError("G3_WEIGHT_METRIC_SCHEME_MISMATCH")
    exposure_path = lake.root / history["outputs"]["risk_exposure_matrix_v1"]["path"]
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    prices_path = lake.silver / "prices_daily" / "data.parquet"
    universe_candidates = list((lake.root / "gold" / "tradable_universe").glob(
        f"artifact_version=1/run_id={history['tradable_universe_run_id']}/tradable_universe.parquet"
    ))
    if len(universe_candidates) != 1:
        raise RuntimeError("G3_TRADABLE_UNIVERSE_ARTIFACT_NOT_UNIQUE")
    universe_path = universe_candidates[0]

    exposure_dates = pl.scan_parquet(exposure_path).select("trade_date").unique().collect()["trade_date"].to_list()
    return_dates = pl.scan_parquet(returns_path).select("trade_date").unique().collect()["trade_date"].to_list()
    date_map = _next_return_map(exposure_dates, return_dates, start=start, end=end)
    selected_exposure_dates = date_map["exposure_date"].to_list()
    selected_return_dates = date_map["trade_date"].to_list()
    exposure_schema = pl.read_parquet_schema(exposure_path)
    exposure = pl.scan_parquet(exposure_path).filter(
        pl.col("trade_date").is_in(selected_exposure_dates) & pl.col("is_valid")
    ).collect()
    risk_columns = [
        column for column, dtype in exposure_schema.items()
        if column.startswith("risk_") and dtype.is_float()
    ]
    if not risk_columns or "risk_country" not in risk_columns:
        raise RuntimeError("G3_L1_RISK_COLUMNS_MISSING")
    returns = pl.scan_parquet(returns_path).filter(
        pl.col("trade_date").is_in(selected_return_dates)
    ).select("trade_date", "asset_id", "total_return", "return_source").collect()
    valuation = pl.scan_parquet(valuation_path).filter(
        pl.col("trade_date").is_in(selected_exposure_dates)
    ).select("trade_date", "asset_id", "float_mkt_cap").collect()
    return_prices = pl.scan_parquet(prices_path).filter(
        pl.col("trade_date").is_in(selected_return_dates)
    ).select(
        "trade_date", "asset_id", "raw_open", "raw_high", "raw_low", "raw_close",
        "limit_up", "limit_down",
    ).collect()
    universe = pl.scan_parquet(universe_path).filter(
        pl.col("trade_date").is_in(selected_exposure_dates)
    ).select("trade_date", "asset_id", "exchange_list_date").collect()
    joined = (
        exposure.join(
            date_map.rename({"exposure_date": "trade_date", "trade_date": "return_date"}),
            on="trade_date", how="inner", validate="m:1",
        )
        .join(returns.rename({"trade_date": "return_date"}), on=("return_date", "asset_id"), how="left", validate="m:1")
        .join(
            return_prices.rename({"trade_date": "return_date"}),
            on=("return_date", "asset_id"), how="left", validate="m:1",
        )
        .join(universe, on=("trade_date", "asset_id"), how="left", validate="m:1")
        .join(valuation, on=("trade_date", "asset_id"), how="left", validate="m:1")
    )
    if _base_weight_artifact_path is not None:
        custom_weights = pl.scan_parquet(_base_weight_artifact_path).filter(
            pl.col("exposure_date").is_in(selected_exposure_dates)
        ).select("exposure_date", "asset_id", "candidate_weight").collect().rename(
            {"exposure_date": "trade_date"}
        )
        joined = joined.join(
            custom_weights, on=("trade_date", "asset_id"), how="left", validate="m:1"
        )
    joined = apply_l2b_label_policy(
        joined, load_return_label_policy(), label_date_column="return_date"
    ).rename({"total_return": "realized_return"}).with_columns(
        (pl.col("is_valid") & pl.col("label_eligible")).alias("model_eligible"),
        (
            (
                pl.col("limit_up").is_not_null()
                & (pl.col("raw_open") >= pl.col("limit_up"))
                & (pl.col("raw_high") >= pl.col("limit_up"))
                & (pl.col("raw_low") >= pl.col("limit_up"))
                & (pl.col("raw_close") >= pl.col("limit_up"))
            )
            | (
                pl.col("limit_down").is_not_null()
                & (pl.col("raw_open") <= pl.col("limit_down"))
                & (pl.col("raw_high") <= pl.col("limit_down"))
                & (pl.col("raw_low") <= pl.col("limit_down"))
                & (pl.col("raw_close") <= pl.col("limit_down"))
            )
        ).fill_null(False).alias("is_limit_locked"),
    ).with_columns(
        (pl.col("model_eligible") & ~pl.col("is_limit_locked")).alias(
            "in_estimation_domain"
        )
    )
    if (
        _base_weight_artifact_path is not None
        and joined.filter(pl.col("model_eligible") & pl.col("candidate_weight").is_null()).height
    ):
        raise RuntimeError("G3_CUSTOM_WEIGHT_COVERAGE_INCOMPLETE")

    if _derived_params_override is None:
        protocol = ResearchProtocol.load(protocol_path)
        derived = protocol.derive(
            factor_count=len(risk_columns), maximum_evaluation_horizon_days=1,
            cross_section_size=int(joined.group_by("trade_date").len()["len"].max()),
            available_estimation_days=len(exposure_dates),
        )["values"]
        derived["new_listing_days_by_regime"] = history["derived_params"]["new_listing_days_by_regime"]
        derived["unresolved_required"] = []
    else:
        derived = dict(_derived_params_override)
    config = load_return_decomposition_config(l2_config_path, derived_params=derived)
    minimum_members = int(derived["minimum_category_members"])
    query_seconds = perf_counter() - wall_start

    factor_rows: list[dict[str, Any]] = []
    specific_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    pivot_start = perf_counter()
    daily_partitions = joined.partition_by("trade_date", as_dict=True)
    pivot_seconds = perf_counter() - pivot_start
    solve_start = perf_counter()
    for key, daily in daily_partitions.items():
        exposure_date = key[0] if isinstance(key, tuple) else key
        return_date = daily["return_date"][0]
        factor_columns, blocks = _active_factor_columns(daily, risk_columns)
        design = build_daily_categorical_constrained_design(
            daily,
            exposure_columns=factor_columns,
            family_by_column={column: "risk" for column in factor_columns},
            categorical_blocks=blocks,
            minimum_category_members=minimum_members,
            estimation_column="in_estimation_domain",
            base_weight_column=(
                "candidate_weight" if _base_weight_artifact_path is not None else None
            ),
        )
        result = None
        failure_reason = ""
        if design.constraint_identification_passed:
            try:
                result = decompose_cross_section(
                    design.regression_input, mode=RegressionMode.RISK_ONLY, config=config
                )
            except ValueError as exc:
                if not str(exc).startswith(("WLS_", "LINEAR_SYSTEM_")):
                    raise
                failure_reason = str(exc)
        else:
            failure_reason = "G3_CONSTRAINT_IDENTIFICATION_FAILED"
        values = result.factor_returns if result else (None,) * len(factor_columns)
        factor_rows.extend({
            "trade_date": return_date, "exposure_date": exposure_date,
            "factor_id": factor_id, "regression_mode": "risk_only",
            "factor_family": "risk", "factor_return": value,
        } for factor_id, value in zip(factor_columns, values))
        assets = result.included_assets if result else design.regression_input.asset_ids
        residuals = result.specific_returns if result else (None,) * len(assets)
        estimation_flags = result.in_estimation_domain if result else (False,) * len(assets)
        estimation_weights = result.estimation_weights if result else (None,) * len(assets)
        huber_multipliers = result.huber_weight_multipliers if result else (None,) * len(assets)
        specific_rows.extend({
            "trade_date": return_date, "exposure_date": exposure_date,
            "asset_id": asset_id, "regression_mode": "risk_only",
            "specific_return": value,
            "in_estimation_domain": in_estimation,
            "exclusion_reason": None if in_estimation else "limit_locked",
            "estimation_weight": estimation_weight,
            "huber_weight_multiplier": multiplier,
            "is_outlier_flagged": multiplier is not None and multiplier < 1.0,
        } for asset_id, value, in_estimation, estimation_weight, multiplier in zip(
            assets, residuals, estimation_flags, estimation_weights, huber_multipliers
        ))
        quality_rows.append({
            "trade_date": return_date, "exposure_date": exposure_date,
            "regression_mode": "risk_only", "status": result.status if result else "invalid",
            "sample_count": result.sample_count if result else len(assets),
            "label_sample_count": result.label_sample_count if result else len(assets),
            "estimation_excluded_count": (
                result.label_sample_count - result.sample_count if result else 0
            ),
            "excluded_count": len(result.excluded_assets) if result else 0,
            "factor_count": len(factor_columns), "constraint_count": len(blocks),
            "constraint_rank": design.constraint_rank,
            "expected_matrix_rank": design.expected_matrix_rank,
            "constraint_identification_passed": design.constraint_identification_passed,
            "matrix_rank": result.matrix_rank if result else design.matrix_rank,
            "stacked_rank": design.stacked_rank, "kkt_rank": design.kkt_rank,
            "condition_number": result.condition_number if result else float("inf"),
            "r_squared": result.r_squared if result else None,
            "full_label_r_squared": result.r_squared if result else None,
            "estimation_domain_r_squared": (
                result.estimation_domain_r_squared if result else None
            ),
            "full_label_unweighted_r_squared": (
                result.unweighted_r_squared if result else None
            ),
            "estimation_domain_unweighted_r_squared": (
                result.estimation_domain_unweighted_r_squared if result else None
            ),
            "huber_downweight_count": (
                sum(value is not None and value < 1.0 for value in result.huber_weight_multipliers)
                if result else 0
            ),
            "huber_downweight_rate": (
                sum(value is not None and value < 1.0 for value in result.huber_weight_multipliers)
                / result.sample_count if result and result.sample_count else None
            ),
            "r_squared_in_expected_range": result.r_squared_in_expected_range if result else False,
            "constraint_error": result.constraint_error if result else None,
            "regression_identity_error": result.regression_identity_error if result else None,
            "base_weight_alpha_risk_cross_gram_max": (
                result.base_weight_alpha_risk_cross_gram_max if result else None
            ),
            "effective_weight_alpha_risk_cross_gram_max": (
                result.effective_weight_alpha_risk_cross_gram_max if result else None
            ),
            "maximum_group_residual_correlation": None,
            "categorical_blocks_json": json.dumps(
                {name: list(columns) for name, columns in blocks.items()}, sort_keys=True
            ),
            "category_member_counts_json": json.dumps(design.category_member_counts, sort_keys=True),
            "category_constraint_weights_json": json.dumps(design.category_constraint_weights, sort_keys=True),
            "category_wls_weight_sums_json": json.dumps(design.category_wls_weight_sums, sort_keys=True),
            "industry_constraint_weights_json": json.dumps(design.industry_constraint_weights, sort_keys=True),
            "industry_wls_weight_sums_json": json.dumps(design.industry_wls_weight_sums, sort_keys=True),
            "warnings": "|".join((*design.warnings, failure_reason) if failure_reason else design.warnings),
        })
        valid = daily.filter("model_eligible")
        stats_rows.extend(cross_section_statistics(
            valid["realized_return"].drop_nulls().to_list(),
            valid.filter(pl.col("realized_return").is_not_null())["board_id"].to_list(),
            trade_date=return_date, regression_mode="risk_only",
        ))
    solve_seconds = perf_counter() - solve_start

    products = {
        "factor_returns_v1": pl.DataFrame(factor_rows),
        "specific_returns_v1": pl.DataFrame(specific_rows, infer_schema_length=None),
        "factor_regression_quality_v1": pl.DataFrame(quality_rows),
        "cross_section_stats_v1": pl.DataFrame(stats_rows),
    }
    inputs = {
        "l1_history_manifest": history_path,
        "l1_diagnostics_manifest": diagnostics_path,
        "risk_exposure_matrix_v1": exposure_path,
        "tradable_universe_v1": universe_path,
        "returns_daily": returns_path,
        "return_date_prices_daily": prices_path,
        "valuation_daily": valuation_path,
        "l2_config": l2_config_path,
        "l1_config": l1_config_path,
        "research_protocol": protocol_path,
        "weight_metric_contract": Path("config/weight_metric_contract_v1.json"),
    }
    if _base_weight_artifact_path is not None:
        inputs["base_weight_artifact"] = _base_weight_artifact_path
    identity = {
        "mode": "risk_only", "start": start.isoformat(), "end": end.isoformat(),
        "l1_run_id": history["run_id"], "risk_factor_set_id": history["risk_factor_set_id"],
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "input_hashes": _input_hashes_override or {
            name: file_sha256(path) for name, path in inputs.items()
        },
        "derived_params": derived, "code_sha": g3_calculation_code_hash(),
        "base_weight_scheme_id": _base_weight_scheme_id,
    }
    run_id = f"g3_risk_only_{end:%Y%m%d}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "gold" / "l2b_risk_only" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    write_start = perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Any] = {}
    for name, frame in products.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        outputs[name] = lake.artifact_record(path)
    quality = products["factor_regression_quality_v1"]
    invalid = quality.filter(pl.col("status") == "invalid").height
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g3_l2b_risk_only",
        "status": "passed" if invalid == 0 else "passed_with_invalid_dates",
        "publish_mode": "formal_current" if publish_current else "validation_only",
        "start": start.isoformat(), "end": end.isoformat(), "created_at": utc_now().isoformat(),
        "l1_run_id": history["run_id"], "l1_diagnostics_run_id": diagnostics["run_id"],
        "silver_version_id": history["silver_version_id"],
        "risk_factor_set_id": history["risk_factor_set_id"],
        "risk_factor_set_status": "candidate", "regression_mode": "risk_only",
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "base_weight_scheme_id": _base_weight_scheme_id,
        "gate_version": G3_GATE_VERSION,
        "gate_config_sha": file_sha256(G3_GATE_CONFIG_PATH),
        "input_hashes": identity["input_hashes"],
        "identity": identity, "counts": {
            "dates": quality.height, "invalid_dates": invalid,
            "factor_return_rows": products["factor_returns_v1"].height,
            "specific_return_rows": products["specific_returns_v1"].height,
        },
        "profiling": {
            "wall_seconds": perf_counter() - wall_start,
            "t_query_seconds": query_seconds,
            "t_pivot_seconds": pivot_seconds,
            "t_solve_seconds": solve_seconds,
            "t_write_seconds": perf_counter() - write_start,
        },
        "outputs": outputs,
    }
    lake.write_immutable_json(manifest_path, manifest)
    if publish_current:
        if invalid:
            raise RuntimeError("G3_CURRENT_PUBLICATION_REQUIRES_ZERO_INVALID_DATES")
        current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
        temporary = current_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "run_id": run_id, "manifest": str(manifest_path.relative_to(lake.root)),
            "regression_mode": "risk_only", "status": "active",
            "updated_at": utc_now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, current_path)
    return manifest_path
