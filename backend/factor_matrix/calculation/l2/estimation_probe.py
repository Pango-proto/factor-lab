"""Dense probe alignment and repeated time-permutation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .estimation_panel import _centered_rank, _rank


@dataclass(frozen=True)
class PermutationDay:
    asset_ids: np.ndarray
    labels: np.ndarray
    exposures: np.ndarray
    signal_residual: np.ndarray
    exposure_columns: tuple[str, ...] = ()


def _projection_basis(exposures: np.ndarray) -> np.ndarray:
    finite_columns = np.std(exposures, axis=0) > 1e-12
    design = exposures[:, finite_columns]
    if design.shape[1] == 0:
        return np.empty((exposures.shape[0], 0), dtype=float)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = 1e-10 * singular[0]
    rank = int(np.sum(singular > tolerance))
    return left[:, :rank]


def _derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("EVALUATION_DATE_AXIS_TOO_FEW_DATES")
    for _ in range(1000):
        candidate = rng.permutation(size)
        if not np.any(candidate == np.arange(size)):
            return candidate
    raise RuntimeError("EVALUATION_DATE_AXIS_DERANGEMENT_FAILED")


def repeated_within_asset_time_shuffle_mean_ics(
    days: Sequence[PermutationDay],
    *,
    repetitions: int,
    seed: int,
    demean_within_asset: bool = False,
    strategy: str = "within_asset_time",
    progress: Callable[[int, int], None] | None = None,
    expected_exposure_columns: Sequence[str] | None = None,
) -> list[float]:
    """Independently permute each asset's label history for every repetition."""
    if repetitions < 1 or not days:
        raise ValueError("EVALUATION_TIME_SHUFFLE_INPUT_INVALID")
    if strategy not in {
        "within_asset_time", "within_date_cross_section", "date_axis"
    }:
        raise ValueError(f"EVALUATION_TIME_SHUFFLE_STRATEGY_INVALID {strategy}")
    if demean_within_asset and strategy != "within_asset_time":
        raise ValueError("EVALUATION_DEMEAN_STRATEGY_INCOMPATIBLE")
    sizes = [day.labels.size for day in days]
    if any(
        size < 3
        or day.asset_ids.size != size
        or day.exposures.shape[0] != size
        or day.signal_residual.size != size
        for day, size in zip(days, sizes)
    ):
        raise ValueError("EVALUATION_TIME_SHUFFLE_DAY_AXIS_MISMATCH")
    if expected_exposure_columns is not None:
        expected = tuple(expected_exposure_columns)
        if any(day.exposure_columns and day.exposure_columns != expected for day in days):
            raise ValueError("EVALUATION_PROBE_EXPOSURE_COLUMNS_MISMATCH")

    labels = np.concatenate([np.asarray(day.labels, dtype=float) for day in days])
    assets = np.concatenate([np.asarray(day.asset_ids, dtype=str) for day in days])
    order = np.argsort(assets, kind="stable")
    sorted_assets = assets[order]
    boundaries = np.flatnonzero(sorted_assets[1:] != sorted_assets[:-1]) + 1
    asset_groups = np.split(order, boundaries)
    source_labels = labels.copy()
    if demean_within_asset:
        for indices in asset_groups:
            source_labels[indices] -= source_labels[indices].mean()

    spans: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    cursor = 0
    for day, size in zip(days, sizes):
        basis = _projection_basis(np.asarray(day.exposures, dtype=float))
        signal_rank = _centered_rank(np.asarray(day.signal_residual, dtype=float))
        signal_norm = float(np.sqrt(np.dot(signal_rank, signal_rank)))
        if signal_norm <= 0:
            raise ValueError("EVALUATION_TIME_SHUFFLE_SIGNAL_RANK_DEGENERATE")
        spans.append((cursor, cursor + size, basis, signal_rank, signal_norm))
        cursor += size

    rng = np.random.default_rng(seed)
    shuffled = np.empty(labels.size, dtype=float)
    output: list[float] = []
    for repetition in range(1, repetitions + 1):
        if strategy == "within_asset_time":
            for indices in asset_groups:
                shuffled[indices] = source_labels[indices][rng.permutation(indices.size)]
        elif strategy == "within_date_cross_section":
            for start, stop, _, _, _ in spans:
                shuffled[start:stop] = source_labels[start:stop][
                    rng.permutation(stop - start)
                ]
        else:
            day_order = _derangement(rng, len(spans))
            shuffled.fill(np.nan)
            for target, source in enumerate(day_order):
                target_start, target_stop, _, _, _ = spans[target]
                source_start, source_stop, _, _, _ = spans[source]
                source_assets = assets[source_start:source_stop]
                source_values = source_labels[source_start:source_stop]
                source_by_asset = dict(zip(source_assets, source_values))
                target_assets = assets[target_start:target_stop]
                shuffled[target_start:target_stop] = np.asarray(
                    [source_by_asset.get(asset, np.nan) for asset in target_assets],
                    dtype=float,
                )
        daily_ics = []
        for day_index, (start, stop, basis, signal_rank, signal_norm) in enumerate(spans):
            values = shuffled[start:stop]
            if strategy == "date_axis":
                valid = np.isfinite(values)
                if valid.sum() < 3:
                    continue
                target_exposures = np.asarray(
                    days[day_index].exposures, dtype=float
                )[valid]
                target_signal = np.asarray(
                    days[day_index].signal_residual, dtype=float
                )[valid]
                basis = _projection_basis(target_exposures)
                if basis.shape[1]:
                    values = values[valid] - basis @ (basis.T @ values[valid])
                else:
                    values = values[valid] - values[valid].mean()
                signal_rank = _centered_rank(target_signal)
                signal_norm = float(np.sqrt(np.dot(signal_rank, signal_rank)))
            elif basis.shape[1]:
                values = values - basis @ (basis.T @ values)
            else:
                values = values - values.mean()
            label_rank = _centered_rank(values)
            denominator = signal_norm * float(np.sqrt(np.dot(label_rank, label_rank)))
            if denominator <= 0:
                raise ValueError("EVALUATION_TIME_SHUFFLE_LABEL_RANK_DEGENERATE")
            daily_ics.append(float(np.dot(signal_rank, label_rank) / denominator))
        output.append(float(np.mean(daily_ics)))
        if progress is not None:
            progress(repetition, repetitions)
    return output
