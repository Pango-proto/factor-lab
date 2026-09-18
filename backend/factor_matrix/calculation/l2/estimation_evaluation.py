"""Registered estimation-only evaluation helpers.

This module contains the statistical contract shared by the rough vertical
slice and later L2 evaluators.  It has no data-lake writes and never opens the
holdout.  Data loading remains in the calling runner so these functions stay
small and directly testable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .predictivity import newey_west_mean

Disposition = Literal["gate", "sensitivity_only", "diagnostic"]
VALID_DISPOSITIONS: frozenset[str] = frozenset(
    {"gate", "sensitivity_only", "diagnostic"}
)
REGISTERED_CONTROLS: tuple[str, ...] = (
    "within_date_cross_sectional_return_shuffle",
    "within_asset_time_shuffle_raw",
    "within_asset_time_shuffle_demeaned",
    "date_axis_shuffle",
    "label_window_shift",
    "raw_signal_shift",
    "label_window_shift_wide",
    "raw_signal_shift_wide",
    "shifted_exposure",
)


@dataclass(frozen=True)
class EvaluationContract:
    framework_id: str
    development_start: date
    holdout_start: date
    horizon_unit: str
    reported_horizons: tuple[int, ...]
    primary_horizon: int
    embargo_trading_days: int
    purge_gap_trading_days: int
    placebo_seed: int
    time_shuffle_repetitions: int
    alignment_shift_offsets: tuple[int, ...]
    signal_id: str
    zero_threshold_standard_errors: float
    rolling_window_trading_days: int
    variant_dispositions: Mapping[str, str] = MappingProxyType({})
    signal_lookback_trading_days: int = 0
    alignment_decay_side_span: int = 2
    neutralization_weight_metric: str = "equal_weight_ols"
    risk_exposure_column_set_id: str = ""
    risk_set_version: int = 0
    risk_set_status: str = ""
    singular_value_rcond: float = 1e-10
    declared_direction: int = 1
    ic_statistic: str = "spearman"
    ic_neutralization_sides: str = "signal_only"

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationContract":
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = payload["sample"]
        controls = payload["negative_controls"]
        inference = payload["inference"]
        channel = payload.get("evaluation_channel", {})
        horizons = tuple(int(value) for value in sample["reported_horizons"])
        contract = cls(
            framework_id=str(payload["framework_id"]),
            development_start=date.fromisoformat(sample["development_start"]),
            holdout_start=date.fromisoformat(sample["holdout_start"]),
            horizon_unit=str(sample["horizon_unit"]),
            reported_horizons=horizons,
            primary_horizon=int(sample["primary_horizon"]),
            embargo_trading_days=int(sample["embargo_trading_days"]),
            purge_gap_trading_days=int(sample["purge_gap_trading_days"]),
            placebo_seed=int(controls["seed"]),
            time_shuffle_repetitions=int(controls.get("time_shuffle_repetitions", 1)),
            alignment_shift_offsets=tuple(int(value) for value in controls.get(
                "alignment_shift_offsets_trading_days", (-2, -1, 0, 1, 2)
            )),
            signal_id=str(payload.get("signal", {}).get("signal_id", "seal_proxy_v0")),
            zero_threshold_standard_errors=float(controls["zero_threshold_standard_errors"]),
            rolling_window_trading_days=int(inference["rolling_window_trading_days"]),
            variant_dispositions=MappingProxyType(dict(channel.get("variant_dispositions", {}))),
            signal_lookback_trading_days=int(channel.get("signal_lookback_trading_days", payload.get("signal", {}).get("lookback_trading_days", 0))),
            alignment_decay_side_span=int(channel.get("alignment_decay_side_span", 2)),
            neutralization_weight_metric=str(channel.get("neutralization_weight_metric", "equal_weight_ols")),
            risk_exposure_column_set_id=str(channel.get("risk_exposure_column_set_id", "")),
            risk_set_version=int(channel.get("risk_set_version", 0)),
            risk_set_status=str(channel.get("risk_set_status", "")),
            singular_value_rcond=float(channel.get("singular_value_rcond", 1e-10)),
            declared_direction=int(channel.get("declared_direction", 1)),
            ic_statistic=str(channel.get("ic_statistic", "spearman")),
            ic_neutralization_sides=str(channel.get("ic_neutralization_sides", "signal_only")),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.horizon_unit != "trading_days":
            raise ValueError("EVALUATION_HORIZON_UNIT_MUST_BE_TRADING_DAYS")
        if not self.reported_horizons or any(value < 1 for value in self.reported_horizons):
            raise ValueError("EVALUATION_HORIZONS_INVALID")
        if self.primary_horizon not in self.reported_horizons:
            raise ValueError("EVALUATION_PRIMARY_HORIZON_UNREGISTERED")
        derived_gap = max(self.reported_horizons) + self.embargo_trading_days
        if self.purge_gap_trading_days != derived_gap:
            raise ValueError("EVALUATION_PURGE_GAP_NOT_DERIVED")
        if self.development_start >= self.holdout_start:
            raise ValueError("EVALUATION_SAMPLE_BOUNDARY_INVALID")
        if self.zero_threshold_standard_errors <= 0:
            raise ValueError("EVALUATION_PLACEBO_THRESHOLD_INVALID")
        if self.time_shuffle_repetitions < 1:
            raise ValueError("EVALUATION_TIME_SHUFFLE_REPETITIONS_INVALID")
        if 0 not in self.alignment_shift_offsets or len(self.alignment_shift_offsets) < 3:
            raise ValueError("EVALUATION_ALIGNMENT_SHIFT_OFFSETS_INVALID")
        if tuple(sorted(set(self.alignment_shift_offsets))) != self.alignment_shift_offsets:
            raise ValueError("EVALUATION_ALIGNMENT_SHIFT_OFFSETS_NOT_STRICTLY_SORTED")
        keys = frozenset(self.variant_dispositions)
        registered = frozenset(REGISTERED_CONTROLS)
        if keys != registered:
            raise ValueError(
                "EVALUATION_DISPOSITION_COVERAGE "
                f"missing={sorted(registered - keys)} extra={sorted(keys - registered)}"
            )
        invalid = {
            key: value for key, value in self.variant_dispositions.items()
            if value not in VALID_DISPOSITIONS
        }
        if invalid:
            raise ValueError(f"EVALUATION_DISPOSITION_VALUE {sorted(invalid)}")
        if not any(value == "gate" for value in self.variant_dispositions.values()):
            raise ValueError("EVALUATION_NO_GATE_DECLARED")
        if self.signal_lookback_trading_days <= 1:
            raise ValueError("EVALUATION_LOOKBACK_UNSET")
        if self.alignment_decay_side_span < 1:
            raise ValueError("EVALUATION_DECAY_SPAN_INVALID")
        if self.neutralization_weight_metric != "equal_weight_ols":
            raise ValueError("EVALUATION_NEUTRALIZATION_WEIGHT_UNSUPPORTED")
        if self.ic_neutralization_sides not in {"signal_only", "both"}:
            raise ValueError("EVALUATION_IC_NEUTRALIZATION_SIDES_INVALID")
        if self.singular_value_rcond <= 0:
            raise ValueError("EVALUATION_RCOND_INVALID")
        for horizon in self.reported_horizons:
            for side in ("label", "signal"):
                grid = self.gate_alignment_offsets(horizon, side)
                if 0 not in grid or len(grid) < 3:
                    raise ValueError(
                        f"EVALUATION_DERIVED_GRID_INVALID h={horizon} side={side}"
                    )

    def gate_alignment_offsets(
        self, horizon_days: int, side: Literal["label", "signal"]
    ) -> tuple[int, ...]:
        if horizon_days <= 0 or self.signal_lookback_trading_days <= 1:
            raise ValueError("EVALUATION_GRID_INPUT_INVALID")
        overlap = min(horizon_days, self.signal_lookback_trading_days)
        decay = self.alignment_decay_side_span
        if side == "label":
            return tuple(range(-overlap, decay + 1))
        if side == "signal":
            return tuple(range(-decay, overlap + 1))
        raise ValueError("EVALUATION_GRID_SIDE_INVALID")

    def sensitivity_alignment_offsets(
        self, horizon_days: int, side: Literal["label", "signal"]
    ) -> tuple[int, ...]:
        if horizon_days <= 0 or self.signal_lookback_trading_days <= 1:
            raise ValueError("EVALUATION_GRID_INPUT_INVALID")
        span = self.signal_lookback_trading_days + 5
        if side == "label":
            return tuple(range(-span, self.alignment_decay_side_span + 1))
        if side == "signal":
            return tuple(range(-self.alignment_decay_side_span, span + 1))
        raise ValueError("EVALUATION_GRID_SIDE_INVALID")


def summarize_ic(values: Sequence[float], *, horizon_days: int) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        raise ValueError("EVALUATION_IC_SERIES_TOO_SHORT")
    count = len(finite)
    mean = sum(finite) / count
    variance = sum((value - mean) ** 2 for value in finite) / (count - 1)
    standard_deviation = math.sqrt(max(variance, 0.0))
    ordered = sorted(finite)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    # The planning contract explicitly requires lag >= N for overlapping
    # N-day labels.  Use N, not N-1, so callers cannot silently understate it.
    nw_mean, nw_standard_error, nw_t = newey_west_mean(finite, lag=horizon_days)
    return {
        "observations": count,
        "mean": mean,
        "median": median,
        "standard_deviation": standard_deviation,
        "ic_ir": mean / standard_deviation if standard_deviation > 0 else 0.0,
        "newey_west_lag": horizon_days,
        "newey_west_mean": nw_mean,
        "newey_west_standard_error": nw_standard_error,
        "newey_west_t": nw_t,
    }


def placebo_zero_gate(
    values: Sequence[float], *, horizon_days: int, threshold_standard_errors: float,
) -> dict[str, Any]:
    summary = summarize_ic(values, horizon_days=horizon_days)
    limit = threshold_standard_errors * float(summary["newey_west_standard_error"])
    distance = abs(float(summary["mean"]))
    return {
        **summary,
        "threshold_standard_errors": threshold_standard_errors,
        "absolute_mean_limit": limit,
        "absolute_mean": distance,
        "passed": distance < limit if limit > 0 else distance == 0,
    }


def rolling_mean(values: Sequence[float], *, window: int) -> list[float | None]:
    if window < 1:
        raise ValueError("EVALUATION_ROLLING_WINDOW_INVALID")
    output: list[float | None] = []
    running = 0.0
    finite_values: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("EVALUATION_ROLLING_SERIES_NONFINITE")
        finite_values.append(value)
        running += value
        if index >= window:
            running -= finite_values[index - window]
        output.append(running / window if index + 1 >= window else None)
    return output


def serial_autocorrelation(values: Sequence[float], *, lags: Sequence[int]) -> dict[str, float | None]:
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("EVALUATION_AUTOCORRELATION_NONFINITE")
    output: dict[str, float | None] = {}
    for lag in lags:
        if lag < 1:
            raise ValueError("EVALUATION_AUTOCORRELATION_LAG_INVALID")
        if len(finite) <= lag + 1:
            output[str(lag)] = None
            continue
        left = finite[lag:]
        right = finite[:-lag]
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
        )
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        output[str(lag)] = numerator / denominator if denominator > 0 else None
    return output


def centered_peak_gate(points: Mapping[int, float]) -> dict[str, Any]:
    """Require a unique center maximum with non-decreasing/decreasing sides."""
    if 0 not in points or len(points) < 3:
        raise ValueError("EVALUATION_CENTER_PEAK_POINTS_INVALID")
    offsets = sorted(points)
    values = [float(points[offset]) for offset in offsets]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("EVALUATION_CENTER_PEAK_NONFINITE")
    zero_index = offsets.index(0)
    left_monotone = all(
        values[index] <= values[index + 1] for index in range(zero_index)
    )
    right_monotone = all(
        values[index] >= values[index + 1]
        for index in range(zero_index, len(values) - 1)
    )
    maximum = max(values)
    peak_offsets = [
        offset for offset, value in zip(offsets, values) if value == maximum
    ]
    passed = peak_offsets == [0] and left_monotone and right_monotone
    return {
        "passed": passed,
        "peak_offsets_trading_days": peak_offsets,
        "left_monotone_toward_zero": left_monotone,
        "right_monotone_away_from_zero": right_monotone,
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("EVALUATION_QUANTILE_PROBABILITY_INVALID")
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("EVALUATION_QUANTILE_VALUES_INVALID")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def permutation_null_gate(
    permutation_mean_ics: Sequence[float], *, interval_probability: float = 0.95,
) -> dict[str, Any]:
    """Gate on whether the empirical permutation interval contains zero."""
    if not 0.0 < interval_probability < 1.0:
        raise ValueError("EVALUATION_NULL_INTERVAL_PROBABILITY_INVALID")
    values = [float(value) for value in permutation_mean_ics]
    if len(values) < 20:
        raise ValueError("EVALUATION_NULL_REPETITIONS_INSUFFICIENT")
    tail = (1.0 - interval_probability) / 2.0
    lower = _linear_quantile(values, tail)
    upper = _linear_quantile(values, 1.0 - tail)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {
        "passed": lower <= 0.0 <= upper,
        "repetitions": len(values),
        "interval_probability": interval_probability,
        "lower": lower,
        "upper": upper,
        "mean": mean,
        "standard_deviation": math.sqrt(max(variance, 0.0)),
        "median": _linear_quantile(values, 0.5),
        "permutation_mean_ics": values,
    }


def shifted_exposure_diagnostic(
    baseline: Sequence[float], shifted: Sequence[float], *, horizon_days: int,
) -> dict[str, Any]:
    if len(baseline) != len(shifted):
        raise ValueError("EVALUATION_SHIFTED_EXPOSURE_AXIS_MISMATCH")
    baseline_summary = summarize_ic(baseline, horizon_days=horizon_days)
    shifted_summary = summarize_ic(shifted, horizon_days=horizon_days)
    baseline_mean = float(baseline_summary["mean"])
    shifted_mean = float(shifted_summary["mean"])
    decline = abs(baseline_mean) - abs(shifted_mean)
    direction = 1.0 if baseline_mean >= 0 else -1.0
    oriented_daily_decline = [
        direction * (float(base) - float(moved))
        for base, moved in zip(baseline, shifted)
    ]
    _, decline_standard_error, decline_t = newey_west_mean(
        oriented_daily_decline, lag=horizon_days
    )
    significant_decline = decline > 0 and decline_t > 2.0
    return {
        "baseline_mean_ic": baseline_mean,
        "shifted_mean_ic": shifted_mean,
        "absolute_ic_decline": decline,
        "decline_standard_error": decline_standard_error,
        "decline_t": decline_t,
        "declined": significant_decline,
        "status": (
            "passed_significant_decline"
            if significant_decline
            else "alignment_review_required"
        ),
    }
