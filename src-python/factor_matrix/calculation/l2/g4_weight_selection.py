"""Preregistered PIT out-of-sample selection among four G3 WLS schemes."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .config import load_return_decomposition_config
from .contracts import RegressionMode
from .design_matrix import build_daily_categorical_constrained_design
from .g3_runner import _active_factor_columns
from .return_decomposition import decompose_cross_section
from ...research_protocol import ResearchProtocol


IMPLEMENTATION_ID = "g4_1a_pit_weight_selection_v1"
DEFAULT_CONFIG = Path("config/g4_validation_protocol_v1.json")
L2_CONFIG = Path("config/l2_return_decomposition_v1.json")
RESEARCH_PROTOCOL = Path("config/research_protocol_v1.json")


def _q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _load_manifest(lake: DataLake, run_id: str) -> tuple[dict[str, Any], Path]:
    path = lake.root / "gold" / "l2b_risk_only" / f"run_id={run_id}" / "_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("run_id") != run_id:
        raise ValueError("G4_WEIGHT_BASELINE_INVALID")
    return payload, path


def _normalize_and_clip(raw: np.ndarray, estimation: np.ndarray, low: float, high: float) -> np.ndarray:
    median = float(np.median(raw[estimation]))
    if not math.isfinite(median) or median <= 0:
        raise ValueError("G4_WEIGHT_MEDIAN_INVALID")
    return np.clip(raw / median, low, high)


def _feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    board = frame["board_id"].to_list()
    return np.column_stack([
        np.ones(frame.height),
        np.log(frame["float_mkt_cap"].to_numpy()),
        frame["risk_liquidity"].to_numpy(),
        frame["risk_residual_volatility"].to_numpy(),
        np.asarray([value == "BSE" for value in board], dtype=float),
        np.asarray([value == "CHINEXT" for value in board], dtype=float),
        np.asarray([value == "STAR" for value in board], dtype=float),
    ])


def _fit_variance_models(
    training: pl.DataFrame, *, minimum_prior_observations: int,
) -> tuple[float, np.ndarray]:
    valid = training.filter(
        pl.col("in_estimation_domain")
        & (pl.col("n_prior_residuals") >= minimum_prior_observations)
        & pl.col("trailing_sigma").is_finite() & (pl.col("trailing_sigma") > 0)
        & pl.col("float_mkt_cap").is_finite() & (pl.col("float_mkt_cap") > 0)
        & pl.col("risk_liquidity").is_finite()
        & pl.col("risk_residual_volatility").is_finite()
    )
    if valid.height < 1000:
        raise ValueError("G4_WEIGHT_TRAINING_INSUFFICIENT")
    target = np.log(valid["trailing_sigma"].to_numpy())
    log_cap = np.log(valid["float_mkt_cap"].to_numpy())
    empirical = np.linalg.lstsq(
        np.column_stack([np.ones(valid.height), log_cap]), target, rcond=None
    )[0]
    structural = np.linalg.lstsq(_feature_matrix(valid), target, rcond=None)[0]
    return float(empirical[1]), structural


def _rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=int)
    result[order] = np.minimum(bins - 1, np.arange(values.size) * bins // values.size)
    return result + 1


def _cv(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    return float(array.std(ddof=0) / mean) if mean > 0 else float("inf")


def _inverse_variance_weight(
    predicted_log_sigma: np.ndarray,
    estimation: np.ndarray,
    sigma_floor_fraction: float,
) -> np.ndarray:
    predicted_sigma = np.exp(predicted_log_sigma)
    median_sigma = float(np.median(predicted_sigma[estimation]))
    floored_sigma = np.maximum(predicted_sigma, sigma_floor_fraction * median_sigma)
    return 1.0 / np.square(floored_sigma)


def _spearman_against_decile(values: np.ndarray) -> float:
    ranks = np.empty(values.size, dtype=float)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size, dtype=float)
    return float(np.corrcoef(np.arange(1, values.size + 1, dtype=float), ranks)[0, 1])


def _evaluation_dates(
    dates: list[Any], *, config: dict[str, Any], protocol: dict[str, Any]
) -> list[Any]:
    """Resolve the explicitly requested evaluation domain.

    The historical G4.1a protocol remains full rolling folds by default.  A
    frozen diagnostic sample is opt-in and must provide an immutable date
    artifact; merely declaring ``n_days`` and ``seed`` cannot silently alter
    the historical evaluation domain.
    """
    domain = protocol.get("evaluation_domain", "full_rolling_domain")
    sample = config.get("diagnostic_day_sample", {})
    if domain == "full_rolling_domain":
        return dates
    if domain != "frozen_diagnostic_sample":
        raise ValueError("G4_WEIGHT_EVALUATION_DOMAIN_UNSUPPORTED")
    date_artifact = sample.get("date_artifact")
    if not date_artifact:
        raise ValueError("G4_WEIGHT_FROZEN_SAMPLE_DATE_ARTIFACT_REQUIRED")
    payload = json.loads(Path(date_artifact).read_text(encoding="utf-8"))
    sampled_dates = payload.get("dates")
    if not isinstance(sampled_dates, list):
        raise ValueError("G4_WEIGHT_FROZEN_SAMPLE_DATES_INVALID")
    expected_days = int(sample["n_days"])
    if len(sampled_dates) != expected_days:
        raise ValueError("G4_WEIGHT_FROZEN_SAMPLE_SIZE_MISMATCH")
    date_set = {datetime.fromisoformat(value).date() for value in sampled_dates}
    resolved = [value for value in dates if value in date_set]
    if len(resolved) != expected_days:
        raise ValueError("G4_WEIGHT_FROZEN_SAMPLE_OUTSIDE_EVALUATION_DOMAIN")
    return resolved


def _daily_scores(
    frame: pl.DataFrame, residuals: np.ndarray, weights: np.ndarray,
    multipliers: tuple[float | None, ...],
) -> tuple[float, float, float, list[dict[str, Any]]]:
    estimation = frame["in_estimation_domain"].to_numpy().astype(bool)
    use = frame.filter(pl.Series(estimation))
    u = residuals[estimation]
    w = weights[estimation]
    wu2 = w * u * u
    liquidity_bins = _rank_bins(use["risk_liquidity"].to_numpy(), 10)
    size_bins = _rank_bins(use["float_mkt_cap"].to_numpy(), 10)
    vol_values = use["realized_volatility_21d"].fill_null(float("nan")).to_numpy()
    finite_vol = np.isfinite(vol_values)
    vol_bins = np.zeros(use.height, dtype=int)
    vol_bins[finite_vol] = _rank_bins(vol_values[finite_vol], 5)
    primary_means: list[float] = []
    for groups in (use["board_id"].to_numpy(), liquidity_bins, size_bins):
        for group in sorted(set(groups.tolist())):
            primary_means.append(float(wu2[groups == group].mean()))
    secondary_means = []
    industries = use["sw_l1_code"].to_numpy()
    for key in sorted(set(zip(industries[finite_vol].tolist(), vol_bins[finite_vol].tolist()))):
        mask = finite_vol & (industries == key[0]) & (vol_bins == key[1])
        if int(mask.sum()) >= 5:
            secondary_means.append(float(wu2[mask].mean()))
    range_ratio = max(primary_means) / min(primary_means)
    multiplier_array = np.asarray([float(value) for value in multipliers if value is not None])
    rows = []
    for decile in range(1, 11):
        mask = liquidity_bins == decile
        rows.append({
            "liquidity_decile": decile, "n_observations": int(mask.sum()),
            "n_downweighted": int((multiplier_array[mask] < 1.0).sum()),
        })
    return _cv(primary_means), _cv(secondary_means), range_ratio, rows


def run_g4_wls_weight_selection(
    lake: DataLake, *, config_path: Path = DEFAULT_CONFIG,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["g4_1a_wls_weight_protocol"]
    if protocol.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("G4_WEIGHT_IMPLEMENTATION_CONFIG_MISMATCH")
    baseline, baseline_manifest_path = _load_manifest(lake, protocol["baseline_run_id"])
    first_candidate_time = utc_now()
    superseded_at = datetime.fromisoformat(
        config["protocol_amendments"][0]["rule_supersession"]["superseded_at"]
    )
    if first_candidate_time <= superseded_at:
        raise RuntimeError("G4_WEIGHT_CANDIDATE_PREDATES_RULE_SUPERSESSION")
    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current_before = current_path.read_bytes()

    l1_current = json.loads((lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json").read_text())
    l1_manifest_path = lake.root / l1_current["manifest"]
    l1 = json.loads(l1_manifest_path.read_text())
    exposure_path = lake.root / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    specific_path = lake.root / baseline["outputs"]["specific_returns_v1"]["path"]
    valuation_path = lake.silver / "valuation_daily" / "data.parquet"
    returns_path = lake.silver / "returns_daily" / "data.parquet"
    schema = pl.read_parquet_schema(exposure_path)
    risk_columns = [name for name, dtype in schema.items() if name.startswith("risk_") and dtype.is_float()]

    identity = {
        "baseline_run_id": baseline["run_id"],
        "input_hashes": {
            "baseline_manifest": file_sha256(baseline_manifest_path),
            "l1_manifest": file_sha256(l1_manifest_path), "config": file_sha256(config_path),
            "returns": file_sha256(returns_path), "valuation": file_sha256(valuation_path),
        },
        "code_sha": source_tree_hash(), "implementation_id": IMPLEMENTATION_ID,
    }
    run_id = f"g4_wls_selection_{baseline['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_wls_selection" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    pit_path = run_dir / "pit_variance_training_v1.parquet"
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute(f"""
          COPY (
            WITH joined AS (
              SELECT s.trade_date,s.exposure_date,s.asset_id,s.specific_return,
                     s.in_estimation_domain,r.total_return AS realized_return,
                     e.board_id,e.sw_l1_code,e.risk_liquidity,
                     e.risk_residual_volatility,v.float_mkt_cap
              FROM read_parquet('{_q(specific_path)}') s
              JOIN read_parquet('{_q(exposure_path)}') e
                ON e.trade_date=s.exposure_date AND e.asset_id=s.asset_id
              JOIN read_parquet('{_q(valuation_path)}') v
                ON v.trade_date=s.exposure_date AND v.asset_id=s.asset_id
              JOIN read_parquet('{_q(returns_path)}') r
                ON r.trade_date=s.trade_date AND r.asset_id=s.asset_id
              WHERE s.specific_return IS NOT NULL AND v.float_mkt_cap>0
            )
            SELECT *,
              count(specific_return) OVER hist AS n_prior_residuals,
              quantile_cont(abs(specific_return),.5) OVER hist/0.6744897501960817 trailing_sigma,
              stddev_pop(realized_return) OVER volhist realized_volatility_21d
            FROM joined
            WINDOW hist AS (PARTITION BY asset_id ORDER BY exposure_date ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING),
                   volhist AS (PARTITION BY asset_id ORDER BY exposure_date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING)
          ) TO '{_q(pit_path)}' (FORMAT PARQUET,COMPRESSION ZSTD)
        """)
    finally:
        connection.close()

    decision_end = datetime.fromisoformat(protocol["evaluation"]["decision_sample_end_inclusive"]).date()
    dates = pl.scan_parquet(pit_path).filter(pl.col("trade_date") <= decision_end).select(
        "exposure_date"
    ).unique().sort("exposure_date").collect()["exposure_date"].to_list()
    dates = _evaluation_dates(dates, config=config, protocol=protocol)
    train_days = int(protocol["evaluation"]["fit_window_trading_days"])
    test_days = int(protocol["evaluation"]["test_window_trading_days"])
    folds = [
        (index, dates[index-train_days:index], dates[index:index+test_days])
        for index in range(train_days, len(dates)-test_days+1, test_days)
    ]
    derived = ResearchProtocol.load(RESEARCH_PROTOCOL).derive(
        factor_count=len(risk_columns), maximum_evaluation_horizon_days=20,
        cross_section_size=5000, available_estimation_days=len(dates),
    )["values"]
    decomposition_config = load_return_decomposition_config(L2_CONFIG, derived_params=derived)
    low, high = (float(value) for value in protocol["pit_policy"]["candidate_weight_ratio_clip"])
    sigma_floor_fraction = float(
        protocol["pit_policy"]["sigma_floor_relative_to_cross_section_median"]
    )
    minimum_prior_observations = int(
        protocol["pit_policy"]["minimum_prior_residual_observations"]
    )
    candidates = [item["id"] for item in protocol["candidates"]]
    fold_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    liquidity_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for fold_number, (_, training_dates, test_dates) in enumerate(folds, start=1):
        training = pl.scan_parquet(pit_path).filter(
            pl.col("exposure_date").is_in(training_dates)
        ).collect()
        beta, structural_coef = _fit_variance_models(
            training, minimum_prior_observations=minimum_prior_observations
        )
        coefficient_rows.append({
            "fold": fold_number, "training_start": training_dates[0],
            "training_end": training_dates[-1], "empirical_cap_beta": beta,
            "structural_coefficients_json": json.dumps(structural_coef.tolist()),
        })
        pit_test = pl.scan_parquet(pit_path).filter(
            pl.col("exposure_date").is_in(test_dates)
        ).collect()
        exposure_test = pl.scan_parquet(exposure_path).filter(
            pl.col("trade_date").is_in(test_dates) & pl.col("is_valid")
        ).select("trade_date", "asset_id", *risk_columns).collect().rename({"trade_date": "exposure_date"})
        joined = pit_test.join(exposure_test, on=("exposure_date", "asset_id"), how="inner")
        fold_scores = {candidate: [] for candidate in candidates}
        fold_guards = {candidate: [] for candidate in candidates}
        for exposure_date, daily in joined.group_by("exposure_date", maintain_order=True):
            exposure_value = exposure_date[0] if isinstance(exposure_date, tuple) else exposure_date
            daily = daily.sort("asset_id").with_columns(pl.lit(True).alias("model_eligible"))
            factor_columns, blocks = _active_factor_columns(daily, risk_columns)
            estimation = daily["in_estimation_domain"].to_numpy().astype(bool)
            cap = daily["float_mkt_cap"].to_numpy()
            feature = _feature_matrix(daily)
            log_cap = np.log(cap)
            raw_by_candidate = {
                "sqrt_cap": np.sqrt(cap),
                "equal": np.ones(daily.height),
                "empirical_cap": _inverse_variance_weight(
                    beta * log_cap, estimation, sigma_floor_fraction
                ),
                "structural": _inverse_variance_weight(
                    feature @ structural_coef, estimation, sigma_floor_fraction
                ),
            }
            for candidate in candidates:
                weight = _normalize_and_clip(raw_by_candidate[candidate], estimation, low, high)
                candidate_daily = daily.with_columns(pl.Series("candidate_weight", weight))
                design = build_daily_categorical_constrained_design(
                    candidate_daily, exposure_columns=factor_columns,
                    family_by_column={column: "risk" for column in factor_columns},
                    categorical_blocks=blocks, estimation_column="in_estimation_domain",
                    return_column="realized_return", base_weight_column="candidate_weight",
                )
                result = decompose_cross_section(
                    design.regression_input, mode=RegressionMode.RISK_ONLY,
                    config=decomposition_config,
                )
                residual = np.asarray(result.specific_returns)
                primary, guard, range_ratio, liquidity = _daily_scores(
                    candidate_daily, residual, weight, result.huber_weight_multipliers,
                )
                fold_scores[candidate].append(primary)
                fold_guards[candidate].append(guard)
                daily_rows.append({
                    "fold": fold_number, "candidate": candidate,
                    "trade_date": daily["trade_date"][0], "exposure_date": exposure_value,
                    "primary_score": primary, "secondary_guard_score": guard,
                    "primary_range_ratio": range_ratio,
                    "r_squared": result.r_squared,
                    "unweighted_r_squared": result.unweighted_r_squared,
                    "condition_number": result.condition_number,
                    "downweight_rate": sum(value < 1.0 for value in result.huber_weight_multipliers if value is not None)/result.sample_count,
                })
                factor_rows.extend({
                    "fold": fold_number, "candidate": candidate,
                    "trade_date": daily["trade_date"][0], "factor_id": factor_id,
                    "factor_return": value,
                } for factor_id, value in zip(result.factor_ids if hasattr(result, "factor_ids") else design.regression_input.factor_ids, result.factor_returns))
                liquidity_rows.extend({
                    "fold": fold_number, "candidate": candidate,
                    "trade_date": daily["trade_date"][0], **row,
                } for row in liquidity)
        for candidate in candidates:
            fold_rows.append({
                "fold": fold_number, "candidate": candidate,
                "training_start": training_dates[0], "training_end": training_dates[-1],
                "test_start": test_dates[0], "test_end": test_dates[-1],
                "primary_score": float(np.mean(fold_scores[candidate])),
                "secondary_guard_score": float(np.mean(fold_guards[candidate])),
            })

    folds_frame = pl.DataFrame(fold_rows)
    candidate_summary = folds_frame.group_by("candidate").agg(
        pl.len().alias("fold_count"), pl.col("primary_score").mean().alias("mean_primary_score"),
        (pl.col("primary_score").std()/pl.len().sqrt()).alias("standard_error"),
        pl.col("secondary_guard_score").mean().alias("mean_secondary_guard_score"),
    )
    minimum_row = candidate_summary.sort("mean_primary_score").row(0, named=True)
    one_se_limit = minimum_row["mean_primary_score"] + minimum_row["standard_error"]
    eligible = set(candidate_summary.filter(pl.col("mean_primary_score") <= one_se_limit)["candidate"])
    winner = next(candidate for candidate in protocol["simplicity_order"] if candidate in eligible)
    ordered = candidate_summary.sort("mean_primary_score")
    runner_up = ordered["candidate"][1]
    winner_guard = float(candidate_summary.filter(pl.col("candidate") == winner)["mean_secondary_guard_score"][0])
    runner_guard = float(candidate_summary.filter(pl.col("candidate") == runner_up)["mean_secondary_guard_score"][0])
    known_residuals = []
    if winner_guard > runner_guard:
        known_residuals.append("winner_worse_than_runner_up_on_secondary_guard")
    daily_frame = pl.DataFrame(daily_rows)
    winner_range = float(daily_frame.filter(pl.col("candidate") == winner)["primary_range_ratio"].median())
    if winner_range > float(protocol["known_residual_policy"]["maximum_group_mean_Wu2_range_ratio"]):
        known_residuals.append("winner_primary_group_range_ratio_above_1.5")
    liquidity_frame = pl.DataFrame(liquidity_rows)
    liquidity_summary = liquidity_frame.group_by("candidate", "liquidity_decile").agg(
        pl.col("n_observations").sum(), pl.col("n_downweighted").sum(),
    ).with_columns(
        (pl.col("n_downweighted")/pl.col("n_observations")).alias("downweight_rate")
    ).sort("candidate", "liquidity_decile")
    for candidate in candidates:
        values = liquidity_summary.filter(pl.col("candidate") == candidate)["downweight_rate"].to_numpy()
        rho = _spearman_against_decile(values)
        candidate_summary = candidate_summary.with_columns(
            pl.when(pl.col("candidate") == candidate).then(pl.lit(rho))
            .otherwise(pl.col("liquidity_downweight_spearman") if "liquidity_downweight_spearman" in candidate_summary.columns else pl.lit(None))
            .alias("liquidity_downweight_spearman")
        )

    winner_liquidity = liquidity_summary.filter(pl.col("candidate") == winner).sort(
        "liquidity_decile"
    )
    winner_rates = winner_liquidity["downweight_rate"].to_numpy()
    winner_rho = _spearman_against_decile(winner_rates)
    acceptance_config = protocol["acceptance_diagnostics"][
        "successful_structural_variance_absorption"
    ]
    acceptance_passed = bool(
        winner_rho <= float(acceptance_config["liquidity_decile_downweight_spearman_maximum"])
        and winner_rates[0] >= winner_rates[-1]
    )
    if not acceptance_passed:
        known_residuals.append(
            "winner_failed_preregistered_liquidity_downweight_direction"
        )

    outputs = {
        "candidate_fold_scores_v1": folds_frame,
        "candidate_summary_v1": candidate_summary,
        "candidate_daily_diagnostics_v1": daily_frame,
        "candidate_factor_returns_v1": pl.DataFrame(factor_rows),
        "candidate_liquidity_downweight_v1": liquidity_summary,
        "variance_model_coefficients_v1": pl.DataFrame(coefficient_rows),
    }
    output_records = {}
    output_records["pit_variance_training_v1"] = lake.artifact_record(pit_path)
    for name, frame in outputs.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        output_records[name] = lake.artifact_record(path)
    if current_path.read_bytes() != current_before:
        raise RuntimeError("G4_WEIGHT_CANDIDATE_MUTATED_CURRENT")
    manifest = {
        "schema_version": 1, "run_id": run_id, "job": "g4_1a_wls_weight_selection",
        "status": "passed", "created_at": first_candidate_time.isoformat(),
        "baseline_run_id": baseline["run_id"], "identity": identity,
        "decision_sample_end": decision_end.isoformat(), "fold_count": len(folds),
        "winner": winner, "minimum_score_candidate": minimum_row["candidate"],
        "one_standard_error_limit": one_se_limit, "runner_up": runner_up,
        "known_residuals": known_residuals, "current_pointer_mutated": False,
        "acceptance_diagnostic_passed": acceptance_passed,
        "acceptance_diagnostic": {
            "liquidity_downweight_spearman": winner_rho,
            "decile_1_downweight_rate": float(winner_rates[0]),
            "decile_10_downweight_rate": float(winner_rates[-1]),
            "decision_role": acceptance_config["decision_role"],
        },
        "protocol": protocol, "outputs": output_records,
    }
    return lake.write_immutable_json(manifest_path, manifest)
