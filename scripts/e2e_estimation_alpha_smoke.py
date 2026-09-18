#!/usr/bin/env python3
"""Standalone estimation-only alpha vertical-slice smoke test.

This file is deliberately outside ``backend/factor_matrix`` and is not
imported by production code.  It reads existing parquet files only and does
not touch the registry, attempt ledger, manifests, or any ``_CURRENT`` file.

The slice is intentionally rough:

    price data -> seal proxy -> country/industry/size/beta neutralization
                 -> IC term structure + Newey-West inference
                 -> required negative controls
                 -> quintile spread -> simple top-decile portfolio
                 -> turnover/cost sensitivity

The hard data cutoff is the frozen estimation cutoff 2025-02-28.  Any attempt
to pass a later end date is rejected so this smoke test cannot touch holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.estimation_evaluation import (
    EvaluationContract,
    centered_peak_gate,
    permutation_null_gate,
    placebo_zero_gate,
    rolling_mean,
    serial_autocorrelation,
    shifted_exposure_diagnostic,
    summarize_ic,
)
from factor_matrix.calculation.l2.estimation_probe import (
    PermutationDay,
    repeated_within_asset_time_shuffle_mean_ics,
)
from factor_matrix.calculation.l2.estimation_panel import (
    ALIGNMENT_SHIFT_OFFSETS,
    DEFAULT_CONTRACT,
    DEFAULT_END,
    DEFAULT_START,
    _load_panel,
    _residualize,
    _shift_key,
    _signal_column,
    _spearman,
)


def run(
    data_root: Path, start: date, end: date,
    *, contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    contract = EvaluationContract.from_json(contract_path)
    if end >= contract.holdout_start:
        raise ValueError(
            f"E2E_SMOKE_HOLDOUT_LOCKED end={end} holdout_start={contract.holdout_start}"
        )
    if start >= end:
        raise ValueError("E2E_SMOKE_DATE_RANGE_INVALID")

    panel, risk_columns, safe_end = _load_panel(data_root, start, end, contract)
    if contract.alignment_shift_offsets != ALIGNMENT_SHIFT_OFFSETS:
        raise ValueError("EVALUATION_ALIGNMENT_SHIFT_OFFSETS_PANEL_MISMATCH")
    signal_column = _signal_column(contract.signal_id)
    daily: list[dict[str, Any]] = []
    permutation_days: list[PermutationDay] = []
    previous_weights: dict[str, float] = {}
    rng = np.random.default_rng(contract.placebo_seed)
    horizons = contract.reported_horizons
    for frame in panel.partition_by("trade_date", maintain_order=True):
        frame = frame.filter(
            pl.col(signal_column).is_finite()
            & pl.all_horizontal([
                pl.col(f"forward_return_{horizon}").is_finite()
                & pl.col(f"time_shuffle_{horizon}").is_finite()
                & pl.col(f"independent_time_shuffle_{horizon}").is_finite()
                & pl.col(f"independent_demeaned_time_shuffle_{horizon}").is_finite()
                for horizon in horizons
            ])
            & pl.all_horizontal([pl.col(column).is_finite() for column in risk_columns])
            & pl.all_horizontal([
                pl.col(f"shifted_{column}").is_finite() for column in risk_columns
            ])
            & pl.all_horizontal([
                pl.col(f"raw_signal_offset_{_shift_key(offset)}").is_finite()
                & pl.col(
                    f"label_window_offset_{_shift_key(offset)}_{contract.primary_horizon}"
                ).is_finite()
                for offset in ALIGNMENT_SHIFT_OFFSETS
            ])
        )
        if frame.height < max(50, len(risk_columns) + 5):
            continue
        x = frame.select(risk_columns).to_numpy()
        x_shifted = frame.select([f"shifted_{column}" for column in risk_columns]).to_numpy()
        signal = frame[signal_column].to_numpy()
        signal_resid, _, signal_condition = _residualize(signal, x)
        signal_resid_shifted, _, shifted_condition = _residualize(signal, x_shifted)
        row: dict[str, Any] = {
            "trade_date": str(frame["trade_date"][0]),
            "n": frame.height,
            "active_signal_count": int(np.sum(np.abs(signal) > 1e-12)),
            "seal_events": int(np.sum(frame["seal_proxy"].to_numpy() > 0)),
        }
        risk_r2: dict[int, float] = {}
        label_residuals: dict[int, np.ndarray] = {}
        label_conditions: list[float] = []
        valid = True
        for horizon in horizons:
            label = frame[f"forward_return_{horizon}"].to_numpy()
            time_label = frame[f"time_shuffle_{horizon}"].to_numpy()
            independent_time_label = frame[f"independent_time_shuffle_{horizon}"].to_numpy()
            independent_demeaned_time_label = frame[
                f"independent_demeaned_time_shuffle_{horizon}"
            ].to_numpy()
            label_resid, r2, label_condition = _residualize(label, x)
            shuffled_label_resid, _, _ = _residualize(rng.permutation(label), x)
            time_label_resid, _, _ = _residualize(time_label, x)
            independent_time_label_resid, _, _ = _residualize(independent_time_label, x)
            independent_demeaned_time_label_resid, _, _ = _residualize(
                independent_demeaned_time_label, x
            )
            shifted_label_resid, _, _ = _residualize(label, x_shifted)
            metrics = {
                f"ic_{horizon}": _spearman(signal_resid, label_resid),
                f"cross_section_shuffle_ic_{horizon}": _spearman(
                    signal_resid, shuffled_label_resid
                ),
                f"time_shuffle_ic_{horizon}": _spearman(signal_resid, time_label_resid),
                f"independent_time_shuffle_ic_{horizon}": _spearman(
                    signal_resid, independent_time_label_resid
                ),
                f"independent_demeaned_time_shuffle_ic_{horizon}": _spearman(
                    signal_resid, independent_demeaned_time_label_resid
                ),
                f"shifted_exposure_ic_{horizon}": _spearman(
                    signal_resid_shifted, shifted_label_resid
                ),
            }
            if any(value is None for value in metrics.values()):
                valid = False
                break
            row.update(metrics)
            risk_r2[horizon] = r2
            label_residuals[horizon] = label_resid
            label_conditions.append(label_condition)
        if not valid:
            continue

        primary_label = frame[f"forward_return_{contract.primary_horizon}"].to_numpy()
        for offset in ALIGNMENT_SHIFT_OFFSETS:
            key = _shift_key(offset)
            shifted_label = frame[
                f"label_window_offset_{key}_{contract.primary_horizon}"
            ].to_numpy()
            shifted_raw_signal = frame[f"raw_signal_offset_{key}"].to_numpy()
            shifted_label_residual, _, _ = _residualize(shifted_label, x)
            shifted_raw_signal_residual, _, _ = _residualize(shifted_raw_signal, x)
            label_shift_ic = _spearman(signal_resid, shifted_label_residual)
            raw_signal_shift_ic = _spearman(
                shifted_raw_signal_residual,
                label_residuals[contract.primary_horizon],
            )
            if label_shift_ic is None or raw_signal_shift_ic is None:
                valid = False
                break
            row[f"label_window_offset_{offset}_ic"] = label_shift_ic
            row[f"raw_signal_offset_{offset}_ic"] = raw_signal_shift_ic
        if not valid:
            continue
        permutation_days.append(PermutationDay(
            asset_ids=np.asarray(frame["asset_id"].to_list(), dtype=str),
            labels=np.asarray(primary_label, dtype=float),
            exposures=np.asarray(x, dtype=float),
            signal_residual=np.asarray(signal_resid, dtype=float),
        ))

        order = np.argsort(signal_resid, kind="stable")
        bucket = max(1, frame.height // 5)
        low, high = order[:bucket], order[-bucket:]
        label_1 = frame["forward_return_1"].to_numpy()
        label_primary = frame[f"forward_return_{contract.primary_horizon}"].to_numpy()
        spread_1 = float(np.mean(label_1[high]) - np.mean(label_1[low]))
        spread_primary = float(np.mean(label_primary[high]) - np.mean(label_primary[low]))

        top_count = max(1, int(np.ceil(frame.height * 0.10)))
        top_indices = order[-top_count:]
        top_assets = set(frame["asset_id"][top_indices].to_list())
        current_weights = {asset: 1.0 / len(top_assets) for asset in top_assets}
        turnover = 0.5 * sum(
            abs(current_weights.get(asset, 0.0) - previous_weights.get(asset, 0.0))
            for asset in set(current_weights) | set(previous_weights)
        )
        previous_weights = current_weights
        row.update({
            "quintile_spread_1": spread_1,
            f"quintile_spread_{contract.primary_horizon}": spread_primary,
            "top_decile_gross_1": float(np.mean(label_1[top_indices])),
            "turnover": turnover,
            **{f"rough_risk_r2_{horizon}": value for horizon, value in risk_r2.items()},
            "rough_risk_condition": max(
                signal_condition, shifted_condition, *label_conditions
            ),
        })
        daily.append(row)

    if not daily:
        raise ValueError("E2E_SMOKE_NO_VALID_DAILY_CROSS_SECTIONS")
    daily_frame = pl.DataFrame(daily)
    gross = daily_frame["top_decile_gross_1"].to_numpy()
    turnover = daily_frame["turnover"].to_numpy()
    costs = {}
    for bps in (10, 25, 50):
        net = gross - turnover * bps / 10_000.0
        costs[str(bps)] = {
            "mean_daily_net_return": float(np.mean(net)),
            "cumulative_net_return": float(np.prod(1.0 + net) - 1.0),
        }

    ic_summaries = {
        str(horizon): summarize_ic(
            daily_frame[f"ic_{horizon}"].to_list(), horizon_days=horizon
        )
        for horizon in horizons
    }
    cross_section_controls = {
        str(horizon): placebo_zero_gate(
            daily_frame[f"cross_section_shuffle_ic_{horizon}"].to_list(),
            horizon_days=horizon,
            threshold_standard_errors=contract.zero_threshold_standard_errors,
        )
        for horizon in horizons
    }
    time_controls = {
        str(horizon): placebo_zero_gate(
            daily_frame[f"time_shuffle_ic_{horizon}"].to_list(),
            horizon_days=horizon,
            threshold_standard_errors=contract.zero_threshold_standard_errors,
        )
        for horizon in horizons
    }
    exposure_controls = {
        str(horizon): shifted_exposure_diagnostic(
            daily_frame[f"ic_{horizon}"].to_list(),
            daily_frame[f"shifted_exposure_ic_{horizon}"].to_list(),
            horizon_days=horizon,
        )
        for horizon in horizons
    }
    independent_time_controls = {
        str(horizon): placebo_zero_gate(
            daily_frame[f"independent_time_shuffle_ic_{horizon}"].to_list(),
            horizon_days=horizon,
            threshold_standard_errors=contract.zero_threshold_standard_errors,
        )
        for horizon in horizons
    }
    independent_demeaned_time_controls = {
        str(horizon): placebo_zero_gate(
            daily_frame[f"independent_demeaned_time_shuffle_ic_{horizon}"].to_list(),
            horizon_days=horizon,
            threshold_standard_errors=contract.zero_threshold_standard_errors,
        )
        for horizon in horizons
    }
    alignment_controls: dict[str, Any] = {}
    for control_name, prefix in (
        ("label_window_shift", "label_window_offset"),
        ("raw_signal_shift", "raw_signal_offset"),
    ):
        points = []
        point_means: dict[int, float] = {}
        for offset in ALIGNMENT_SHIFT_OFFSETS:
            summary = summarize_ic(
                daily_frame[f"{prefix}_{offset}_ic"].to_list(),
                horizon_days=contract.primary_horizon,
            )
            point_means[offset] = float(summary["mean"])
            points.append({"offset_trading_days": offset, **summary})
        alignment_controls[control_name] = {
            **centered_peak_gate(point_means),
            "points": points,
        }

    def permutation_progress(label: str):
        def report(done: int, total: int) -> None:
            if done == 1 or done % 10 == 0 or done == total:
                print(
                    f"{label} time-shuffle permutations {done}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
        return report

    permutation_mean_ics = repeated_within_asset_time_shuffle_mean_ics(
        permutation_days,
        repetitions=contract.time_shuffle_repetitions,
        seed=contract.placebo_seed + 100,
        progress=permutation_progress("raw"),
    )
    raw_permutation_distribution = permutation_null_gate(permutation_mean_ics)
    demeaned_permutation_mean_ics = repeated_within_asset_time_shuffle_mean_ics(
        permutation_days,
        repetitions=contract.time_shuffle_repetitions,
        seed=contract.placebo_seed + 200,
        demean_within_asset=True,
        progress=permutation_progress("asset-demeaned"),
    )
    permutation_distribution = permutation_null_gate(demeaned_permutation_mean_ics)
    negative_control_passed = (
        all(bool(item["passed"]) for item in cross_section_controls.values())
        and bool(alignment_controls["label_window_shift"]["passed"])
        and bool(alignment_controls["raw_signal_shift"]["passed"])
        and bool(permutation_distribution["passed"])
    )
    rolling_by_horizon = {
        str(horizon): rolling_mean(
            daily_frame[f"ic_{horizon}"].to_list(),
            window=contract.rolling_window_trading_days,
        )
        for horizon in horizons
    }
    result = {
        "script": "e2e_estimation_alpha_smoke",
        "schema_version": 1,
        "framework_id": contract.framework_id,
        "status": "completed" if negative_control_passed else "blocked_negative_control",
        "scope": {
            "start": str(start), "end": str(end),
            "holdout_start": str(contract.holdout_start),
            "safe_feature_end": str(safe_end),
            "horizon_unit": contract.horizon_unit,
            "reported_horizons": list(horizons),
            "primary_horizon": contract.primary_horizon,
            "embargo_trading_days": contract.embargo_trading_days,
            "purge_gap_trading_days": contract.purge_gap_trading_days,
            "risk_model": ["country", "industry_L1", "size", "beta"],
            "alpha_proxy": contract.signal_id,
            "uses_statistics_factors": False,
            "uses_g4_2": False,
            "uses_attempt_ledger": False,
            "writes_current_pointer": False,
        },
        "data_quality": {
            "panel_rows": panel.height,
            "valid_daily_cross_sections": daily_frame.height,
            "median_cross_section": float(daily_frame["n"].median()),
            "cross_section_quantiles": {
                key: float(np.quantile(daily_frame["n"].to_numpy(), probability))
                for key, probability in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
            },
            "active_signal_count_quantiles": {
                key: float(np.quantile(
                    daily_frame["active_signal_count"].to_numpy(), probability
                ))
                for key, probability in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
            },
            "seal_event_count_quantiles": {
                key: float(np.quantile(
                    daily_frame["seal_events"].to_numpy(), probability
                ))
                for key, probability in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
            },
            "total_seal_events": int(daily_frame["seal_events"].sum()),
            "median_rough_risk_r2_1": float(daily_frame["rough_risk_r2_1"].median()),
            f"median_rough_risk_r2_{contract.primary_horizon}": float(
                daily_frame[f"rough_risk_r2_{contract.primary_horizon}"].median()
            ),
            "p99_rough_risk_condition": float(daily_frame["rough_risk_condition"].quantile(0.99)),
        },
        "ic": {
            "by_horizon": ic_summaries,
            "decay_curve": [
                {"horizon_trading_days": horizon, **ic_summaries[str(horizon)]}
                for horizon in horizons
            ],
            "daily_series": [
                {
                    "trade_date": row["trade_date"],
                    "horizons": {
                        str(horizon): {
                            "ic": row[f"ic_{horizon}"],
                            "rolling_12m_mean": rolling_by_horizon[str(horizon)][index],
                            "whole_cross_section_time_shuffle_ic": row[
                                f"time_shuffle_ic_{horizon}"
                            ],
                            "within_asset_independent_time_shuffle_ic": row[
                                f"independent_time_shuffle_ic_{horizon}"
                            ],
                            "within_asset_independent_demeaned_time_shuffle_ic": row[
                                f"independent_demeaned_time_shuffle_ic_{horizon}"
                            ],
                        }
                        for horizon in horizons
                    },
                }
                for index, row in enumerate(daily)
            ],
        },
        "negative_controls": {
            "passed": negative_control_passed,
            "seed": contract.placebo_seed,
            "within_date_cross_sectional_return_shuffle": cross_section_controls,
            "whole_cross_section_time_axis_shuffle": time_controls,
            "within_asset_independent_time_shuffle": independent_time_controls,
            "within_asset_independent_time_shuffle_asset_demeaned_diagnostic": (
                independent_demeaned_time_controls
            ),
            "label_window_shift_peak_at_zero": alignment_controls[
                "label_window_shift"
            ],
            "raw_signal_shift_peak_at_zero": alignment_controls["raw_signal_shift"],
            "within_asset_time_shuffle_100_repetition_raw_diagnostic": (
                raw_permutation_distribution
            ),
            "within_asset_time_shuffle_100_repetition_asset_demeaned_null": (
                permutation_distribution
            ),
            "primary_horizon_autocorrelation": {
                "whole_cross_section_time_axis_shuffle": serial_autocorrelation(
                    daily_frame[f"time_shuffle_ic_{contract.primary_horizon}"].to_list(),
                    lags=(1, 2, 5, 10, 20),
                ),
                "within_asset_independent_time_shuffle": serial_autocorrelation(
                    daily_frame[
                        f"independent_time_shuffle_ic_{contract.primary_horizon}"
                    ].to_list(),
                    lags=(1, 2, 5, 10, 20),
                ),
            },
            "risk_exposure_matrix_shift_one_trading_day_sensitivity_only": (
                exposure_controls
            ),
        },
        "backtest": {
            "mean_quintile_spread_1": float(daily_frame["quintile_spread_1"].mean()),
            f"mean_quintile_spread_{contract.primary_horizon}": float(
                daily_frame[f"quintile_spread_{contract.primary_horizon}"].mean()
            ),
            "mean_top_decile_gross_1": float(daily_frame["top_decile_gross_1"].mean()),
            "mean_turnover": float(daily_frame["turnover"].mean()),
            "cost_sensitivity_bps_per_turnover": costs,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/metadata/evaluation-framework-summary.json"),
    )
    args = parser.parse_args()
    result = run(args.data_root, args.start, args.end, contract_path=args.contract)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "alpha_proxy": result["scope"]["alpha_proxy"],
        "negative_controls_passed": result["negative_controls"]["passed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
