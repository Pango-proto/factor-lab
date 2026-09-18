#!/usr/bin/env python3
"""Audit estimation alignment and time-shuffle semantics without touching holdout.

The exposure lead/lag curve is deliberately reported as a neutralization-date
sensitivity diagnostic.  Risk exposures are controls applied to both signal
and label, not the alpha signal itself, so the curve's maximum cannot by itself
prove which date is the economically correct feature date.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.estimation_evaluation import (
    EvaluationContract,
    placebo_zero_gate,
    serial_autocorrelation,
    summarize_ic,
)
from factor_matrix.calculation.l2.estimation_panel import (
    DEFAULT_CONTRACT,
    DEFAULT_END,
    DEFAULT_START,
    EXPOSURE_SHIFT_OFFSETS,
    _load_panel,
    _residualize,
    _shift_key,
    _spearman,
)


DEFAULT_OUTPUT = Path(
    "data/diagnostics/evaluation_alignment/alignment_diagnostic_20260818.json"
)
AUTOCORRELATION_LAGS = (1, 2, 5, 10, 20)


def _alignment_examples(
    panel: pl.DataFrame, *, horizon: int, count: int = 5,
) -> list[dict[str, Any]]:
    candidates = panel.filter(
        (pl.col("seal_proxy") > 0)
        & pl.col(f"label_start_date_{horizon}").is_not_null()
        & pl.col(f"label_date_{horizon}").is_not_null()
        & pl.all_horizontal([
            pl.col(f"exposure_date_{_shift_key(offset)}").is_not_null()
            for offset in EXPOSURE_SHIFT_OFFSETS
        ])
    )
    dates = candidates["trade_date"].unique().sort().to_list()
    if len(dates) < count:
        raise ValueError("ALIGNMENT_DIAGNOSTIC_INSUFFICIENT_EXAMPLE_DATES")
    positions = np.linspace(0, len(dates) - 1, count, dtype=int)
    output = []
    for position in positions:
        selected = (
            candidates.filter(pl.col("trade_date") == dates[position])
            .sort("seal_proxy", descending=True)
            .row(0, named=True)
        )
        signal_date = selected["trade_date"]
        label_start = selected[f"label_start_date_{horizon}"]
        label_end = selected[f"label_date_{horizon}"]
        output.append({
            "asset_id": selected["asset_id"],
            "signal_trade_date": str(signal_date),
            "signal_uses_price_fields_from": str(signal_date),
            "baseline_exposure_date": str(selected["exposure_date_0"]),
            "exposure_dates_by_offset": {
                str(offset): str(selected[f"exposure_date_{_shift_key(offset)}"])
                for offset in EXPOSURE_SHIFT_OFFSETS
            },
            "forward_return_window_start": str(label_start),
            "forward_return_window_end": str(label_end),
            "forward_window_trading_days": horizon,
            "forward_window_includes_signal_day": label_start <= signal_date <= label_end,
            "baseline_exposure_equals_signal_day": selected["exposure_date_0"] == signal_date,
        })
    return output


def run(
    data_root: Path, *, start: date, end: date, contract_path: Path,
) -> dict[str, Any]:
    contract = EvaluationContract.from_json(contract_path)
    horizon = contract.primary_horizon
    panel, risk_columns, safe_end = _load_panel(data_root, start, end, contract)
    examples = _alignment_examples(panel, horizon=horizon)

    daily: list[dict[str, Any]] = []
    for frame in panel.partition_by("trade_date", maintain_order=True):
        required = (
            pl.col("seal_proxy").is_finite()
            & pl.col(f"forward_return_{horizon}").is_finite()
            & pl.col(f"time_shuffle_{horizon}").is_finite()
            & pl.col(f"independent_time_shuffle_{horizon}").is_finite()
            & pl.col(f"independent_demeaned_time_shuffle_{horizon}").is_finite()
            & pl.all_horizontal([
                pl.col(f"exposure_{_shift_key(offset)}_{column}").is_finite()
                for offset in EXPOSURE_SHIFT_OFFSETS
                for column in risk_columns
            ])
        )
        frame = frame.filter(required)
        if frame.height < max(50, len(risk_columns) + 5):
            continue
        signal = frame["seal_proxy"].to_numpy()
        label = frame[f"forward_return_{horizon}"].to_numpy()
        row: dict[str, Any] = {
            "trade_date": str(frame["trade_date"][0]),
            "n": frame.height,
        }
        signal_residuals: dict[int, np.ndarray] = {}
        label_residuals: dict[int, np.ndarray] = {}
        baseline_signal_residual: np.ndarray | None = None
        baseline_x: np.ndarray | None = None
        for offset in EXPOSURE_SHIFT_OFFSETS:
            key = _shift_key(offset)
            x = frame.select([
                f"exposure_{key}_{column}" for column in risk_columns
            ]).to_numpy()
            signal_residual, _, _ = _residualize(signal, x)
            label_residual, _, _ = _residualize(label, x)
            value = _spearman(signal_residual, label_residual)
            if value is None:
                break
            signal_residuals[offset] = signal_residual
            label_residuals[offset] = label_residual
            row[f"exposure_offset_{offset}_ic"] = value
            if offset == 0:
                baseline_signal_residual = signal_residual
                baseline_x = x
        else:
            if baseline_signal_residual is None or baseline_x is None:
                raise RuntimeError("ALIGNMENT_DIAGNOSTIC_BASELINE_MISSING")
            for offset in EXPOSURE_SHIFT_OFFSETS:
                signal_only = _spearman(signal_residuals[offset], label_residuals[0])
                label_only = _spearman(signal_residuals[0], label_residuals[offset])
                if signal_only is None or label_only is None:
                    break
                row[f"signal_neutralizer_offset_{offset}_ic"] = signal_only
                row[f"label_neutralizer_offset_{offset}_ic"] = label_only
            else:
                pass
            if any(
                f"signal_neutralizer_offset_{offset}_ic" not in row
                for offset in EXPOSURE_SHIFT_OFFSETS
            ):
                continue
            for variant, column in (
                ("whole_cross_section_date_permutation", f"time_shuffle_{horizon}"),
                ("within_asset_independent_permutation", f"independent_time_shuffle_{horizon}"),
                (
                    "within_asset_independent_permutation_asset_demeaned",
                    f"independent_demeaned_time_shuffle_{horizon}",
                ),
            ):
                shuffled = frame[column].to_numpy()
                shuffled_residual, _, _ = _residualize(shuffled, baseline_x)
                value = _spearman(baseline_signal_residual, shuffled_residual)
                if value is None:
                    break
                row[f"{variant}_ic"] = value
            else:
                daily.append(row)

    if len(daily) < 2:
        raise ValueError("ALIGNMENT_DIAGNOSTIC_NO_VALID_DAILY_SERIES")
    frame = pl.DataFrame(daily)
    baseline = frame["exposure_offset_0_ic"].to_list()
    curve = []
    for offset in EXPOSURE_SHIFT_OFFSETS:
        values = frame[f"exposure_offset_{offset}_ic"].to_list()
        summary = summarize_ic(values, horizon_days=horizon)
        delta = [value - base for value, base in zip(values, baseline)]
        delta_summary = summarize_ic(delta, horizon_days=horizon)
        curve.append({
            "exposure_offset_trading_days": offset,
            **summary,
            "delta_vs_offset_0_mean": delta_summary["mean"],
            "delta_vs_offset_0_newey_west_t": delta_summary["newey_west_t"],
        })
    peak = max(curve, key=lambda item: float(item["mean"]))
    means = [float(item["mean"]) for item in curve]
    peak_index = curve.index(peak)
    peak_is_internal = 0 < peak_index < len(curve) - 1
    single_peak_monotone = (
        all(means[index] <= means[index + 1] for index in range(peak_index))
        and all(means[index] >= means[index + 1] for index in range(peak_index, len(means) - 1))
    )
    neutralization_decomposition: dict[str, list[dict[str, Any]]] = {}
    for arm, prefix in (
        ("shift_signal_neutralizer_only", "signal_neutralizer_offset"),
        ("shift_label_neutralizer_only", "label_neutralizer_offset"),
    ):
        points = []
        for offset in EXPOSURE_SHIFT_OFFSETS:
            summary = summarize_ic(
                frame[f"{prefix}_{offset}_ic"].to_list(), horizon_days=horizon
            )
            points.append({"exposure_offset_trading_days": offset, **summary})
        neutralization_decomposition[arm] = points

    time_variants: dict[str, Any] = {}
    for variant in (
        "whole_cross_section_date_permutation",
        "within_asset_independent_permutation",
        "within_asset_independent_permutation_asset_demeaned",
    ):
        values = frame[f"{variant}_ic"].to_list()
        time_variants[variant] = {
            "zero_gate": placebo_zero_gate(
                values,
                horizon_days=horizon,
                threshold_standard_errors=contract.zero_threshold_standard_errors,
            ),
            "autocorrelation": serial_autocorrelation(
                values, lags=AUTOCORRELATION_LAGS
            ),
            "daily_series": [
                {"trade_date": day, "ic": value}
                for day, value in zip(frame["trade_date"].to_list(), values)
            ],
        }

    return {
        "schema_version": 1,
        "diagnostic_id": "evaluation_alignment_diagnostic_v1",
        "status": "completed_neutralization_sensitivity_only",
        "scope": {
            "start": str(start),
            "end": str(end),
            "safe_feature_end": str(safe_end),
            "holdout_start": str(contract.holdout_start),
            "holdout_opened": False,
            "primary_horizon_trading_days": horizon,
            "valid_daily_cross_sections": frame.height,
        },
        "source_semantics": {
            "signal": "seal_proxy uses price fields dated t",
            "baseline_risk_exposure": (
                "risk exposure row dated t; builder uses valuation t and trailing returns "
                "through t (all source filters are <= as_of=t)"
            ),
            "forward_return": "sum of total_return rows t+1 through t+h; t is excluded",
            "offset_convention": "-1 means use t-1 exposure; +1 means diagnostic-only future t+1 exposure",
            "lead_offsets_are_production_eligible": False,
            "exposure_shift_is_blocking_gate": False,
        },
        "alignment_examples": examples,
        "exposure_shift_curve": {
            "points": curve,
            "peak_offset_trading_days": peak["exposure_offset_trading_days"],
            "peak_is_internal": peak_is_internal,
            "single_peak_monotone": single_peak_monotone,
            "interpretation_limit": (
                "Exposures are neutralization controls on both signal and label. A higher IC "
                "under stale or future controls does not by itself identify the correct signal date."
            ),
            "neutralization_decomposition": neutralization_decomposition,
        },
        "time_shuffle": {
            "implementation": {
                "whole_cross_section_date_permutation": (
                    "one global permutation of dated cross-sections; source-day asset IDs and "
                    "cross-sectional covariance are retained, then joined to target date by asset_id"
                ),
                "within_asset_independent_permutation": (
                    "each asset's label history is independently permuted using an asset/date hash; "
                    "contemporaneous cross-sectional covariance is destroyed"
                ),
                "within_asset_independent_permutation_asset_demeaned": (
                    "same independent permutation after subtracting each asset's full-sample mean label"
                ),
            },
            "variants": time_variants,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.data_root, start=args.start, end=args.end, contract_path=args.contract
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "peak_offset_trading_days": result["exposure_shift_curve"]["peak_offset_trading_days"],
        "single_peak_monotone": result["exposure_shift_curve"]["single_peak_monotone"],
        "blocking_gate": False,
        "time_shuffle": {
            key: {
                "mean": value["zero_gate"]["mean"],
                "newey_west_t": value["zero_gate"]["newey_west_t"],
                "passed": value["zero_gate"]["passed"],
            }
            for key, value in result["time_shuffle"]["variants"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
