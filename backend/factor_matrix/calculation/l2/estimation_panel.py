"""Shared estimation-panel construction and alignment helpers.

This module centralizes the formal panel semantics used by both the alpha
evaluation runner and its focused alignment diagnostic.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.estimation_evaluation import EvaluationContract


DEFAULT_START = date(2019, 10, 8)
DEFAULT_END = date(2025, 1, 16)
DEFAULT_CONTRACT = Path("config/evaluation_framework_v1.json")
TRADABLE_UNIVERSE_PATH = Path(
    "gold/tradable_universe/artifact_version=1/"
    "run_id=tradable_universe_v1_20260814_f3cbb68ab636/tradable_universe.parquet"
)
RISK_PREFIXES = ("risk_country", "risk_industry_", "risk_board_")
RISK_TAIL = (
    "risk_size",
    "risk_beta",
    "risk_residual_volatility",
    "risk_liquidity",
    "risk_nonlinear_size",
    "risk_listing_age",
)
EXPOSURE_SHIFT_OFFSETS = (-2, -1, 0, 1, 2)
ALIGNMENT_SHIFT_OFFSETS = (-2, -1, 0, 1, 2)
_RESIDUALIZATION_CACHE: OrderedDict[
    int, tuple[np.ndarray, np.ndarray, float]
] = OrderedDict()


@dataclass(frozen=True)
class DailyICContext:
    """Residualized daily panel reused by every alignment offset."""

    dates: tuple[date, ...]
    assets: tuple[str, ...]
    lookup: np.ndarray
    signal_residual: np.ndarray
    diagnostics: dict[int, dict[str, object]]
    exposure_offset: int
    policy_fingerprint: tuple[object, ...]


def _shift_key(offset: int) -> str:
    return f"m{abs(offset)}" if offset < 0 else f"p{offset}" if offset > 0 else "0"


def _signal_column(signal_id: str) -> str:
    columns = {
        "reversal_20d_v1": "reversal_20d",
        "seal_proxy_v0": "seal_proxy",
    }
    try:
        return columns[signal_id]
    except KeyError as error:
        raise ValueError(f"EVALUATION_SIGNAL_UNSUPPORTED {signal_id}") from error


def _global_exposure_date_mapping(
    exposure_dates: list[date], *, start: date, end: date, offset: int,
) -> pl.DataFrame:
    """Map every target date to one market-wide source date.

    This must never be implemented as ``shift().over('asset_id')`` because an
    asset missing one exposure row would then jump by more than one market day.
    """
    ordered = sorted(set(exposure_dates))
    positions = {value: index for index, value in enumerate(ordered)}
    key = _shift_key(offset)
    rows = []
    for target_date in ordered:
        if not start <= target_date <= end:
            continue
        source_position = positions[target_date] + offset
        if 0 <= source_position < len(ordered):
            rows.append({
                "trade_date": target_date,
                f"exposure_date_{key}": ordered[source_position],
            })
    return pl.DataFrame(rows)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [values.size]))
    average_ranks = (starts + stops - 1) / 2.0
    ranks[order] = np.repeat(average_ranks, stops - starts)
    return ranks


def _centered_rank(values: np.ndarray) -> np.ndarray:
    ranks = _rank(values)
    return ranks - ranks.mean()


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    x, y = _centered_rank(left), _centered_rank(right)
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def prepare_daily_ic_context(
    panel: pl.DataFrame,
    risk_columns: Sequence[str],
    *,
    horizon_days: int | None = None,
    exposure_offset: int = 0,
) -> DailyICContext:
    """Build the residual table once; alignment offsets are integer gathers."""
    if horizon_days is not None and horizon_days < 1:
        raise ValueError("EVALUATION_HORIZON_INVALID")
    if abs(exposure_offset) > 10000:
        raise ValueError("EVALUATION_OFFSET_OUT_OF_RANGE")
    signal_column = "reversal_20d"
    exposure_columns = list(risk_columns)
    required = ["trade_date", "asset_id", signal_column, *exposure_columns]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"EVALUATION_PANEL_COLUMNS_MISSING {missing}")
    data = panel.select(required).sort(["trade_date", "asset_id"])
    dates = data.get_column("trade_date").unique(maintain_order=True).to_list()
    assets = data.get_column("asset_id").unique(maintain_order=True).to_list()
    date_idx = {value: index for index, value in enumerate(dates)}
    asset_idx = {value: index for index, value in enumerate(assets)}
    lookup = np.full((len(dates), len(assets)), -1, dtype=np.int32)
    row_asset_idx = np.full(len(data), -1, dtype=np.int32)
    rows_by_date: list[list[int]] = [[] for _ in dates]
    trade_dates = data.get_column("trade_date").to_list()
    asset_ids = data.get_column("asset_id").to_list()
    for row_index, (trade_date, asset_id) in enumerate(zip(trade_dates, asset_ids)):
        d, a = date_idx[trade_date], asset_idx[asset_id]
        if lookup[d, a] != -1:
            raise ValueError("EVALUATION_PANEL_DUPLICATE_DATE_ASSET")
        lookup[d, a] = row_index
        row_asset_idx[row_index] = a
        rows_by_date[d].append(row_index)

    values = data.get_column(signal_column).to_numpy()
    signal_residual = np.full(len(values), np.nan)
    label_residual = np.full(len(values), np.nan)
    diagnostics: dict[int, dict[str, object]] = {}
    for d, row_indices in enumerate(rows_by_date):
        ix = np.asarray(row_indices, dtype=np.int64)
        x = data.select(exposure_columns).gather(ix).to_numpy().astype(float)
        if exposure_offset:
            source_date = d + exposure_offset
            if not 0 <= source_date < len(dates):
                continue
            source_rows = lookup[source_date, row_asset_idx[ix]]
            x = np.full((ix.size, len(exposure_columns)), np.nan)
            present = source_rows >= 0
            source_ix = source_rows[present].astype(np.uint32)
            x[present] = data.select(exposure_columns).gather(source_ix).to_numpy().astype(float)
        physical_columns = np.any(np.isfinite(x), axis=0)
        x = x[:, physical_columns]
        valid = (
            np.isfinite(values[ix])
            & np.all(np.isfinite(x), axis=1)
        )
        if valid.sum() < max(3, x.shape[1] + 2):
            continue
        valid_ix = ix[valid]
        singular = np.linalg.svd(x[valid], compute_uv=False)
        rank = int(np.sum(singular > 1e-10 * singular[0]))
        signal_residual[valid_ix] = _residualize(values[valid_ix], x[valid])[0]
        diagnostics[d] = {
            "n": int(valid.sum()),
            "n_configured": int(len(exposure_columns)),
            "n_physical_retained": int(x.shape[1]),
            "exposure_rank": rank,
            "constraint_block_rank": int(x.shape[1] - rank),
            "sigma_min_true_ratio": float(singular[rank - 1] / singular[0]),
            "sigma_max_noise_ratio": float(singular[rank] / singular[0]) if rank < singular.size else 0.0,
            "condition_number": float(singular[0] / singular[rank - 1]),
            "rcond_used": 1e-10,
        }

    return DailyICContext(
        dates=tuple(dates),
        assets=tuple(str(value) for value in assets),
        lookup=lookup,
        signal_residual=signal_residual,
        diagnostics=diagnostics,
        exposure_offset=exposure_offset,
        policy_fingerprint=(
            tuple(risk_columns), 1e-10, "equal_weight_ols", "reversal_20d_v1",
            "signal_only",
        ),
    )


def compute_daily_ic_series(
    panel: pl.DataFrame,
    risk_columns: Sequence[str],
    *,
    horizon_days: int,
    label_offset: int = 0,
    signal_offset: int = 0,
    exposure_offset: int = 0,
    context: DailyICContext | None = None,
) -> list[dict[str, object]]:
    """Compute one daily Spearman IC series from a reusable residual table."""
    if any(abs(offset) > 10000 for offset in (label_offset, signal_offset, exposure_offset)):
        raise ValueError("EVALUATION_OFFSET_OUT_OF_RANGE")
    label_column = f"forward_return_{horizon_days}"
    if context is None or context.exposure_offset != exposure_offset:
        context = prepare_daily_ic_context(
            panel, risk_columns, horizon_days=horizon_days,
            exposure_offset=exposure_offset,
        )
    elif context.policy_fingerprint[0] != tuple(risk_columns):
        raise ValueError("EVALUATION_CONTEXT_POLICY_FINGERPRINT_MISMATCH")
    dates = context.dates
    lookup = context.lookup
    signal_residual = context.signal_residual
    label_data = panel.select(["trade_date", "asset_id", label_column]).sort(
        ["trade_date", "asset_id"]
    )
    if tuple(label_data["trade_date"].unique(maintain_order=True).to_list()) != dates:
        raise ValueError("EVALUATION_CONTEXT_DATE_AXIS_MISMATCH")
    labels = label_data.get_column(label_column).to_numpy()
    rows: list[dict[str, object]] = []
    for d, trade_date in enumerate(dates):
        signal_date = d + signal_offset
        label_date = d + label_offset
        if not (0 <= signal_date < len(dates) and 0 <= label_date < len(dates)):
            continue
        signal_rows = lookup[signal_date]
        label_rows = lookup[label_date]
        valid_rows = (
            (signal_rows >= 0)
            & (label_rows >= 0)
            & np.isfinite(signal_residual[np.maximum(signal_rows, 0)])
            & np.isfinite(labels[np.maximum(label_rows, 0)])
        )
        if int(valid_rows.sum()) < 3:
            continue
        sr = signal_residual[signal_rows[valid_rows]]
        lr = labels[label_rows[valid_rows]]
        ic = _spearman(sr, lr)
        if ic is None or signal_date not in context.diagnostics:
            continue
        rows.append({
            "trade_date": trade_date,
            "ic": float(ic),
            **context.diagnostics[signal_date],
            "label_offset": int(label_offset),
            "signal_offset": int(signal_offset),
            "exposure_offset": int(exposure_offset),
            "signal_trade_date": dates[signal_date],
        })
    return rows

def _residualize(values: np.ndarray, exposures: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Equal-weight OLS residualization for this diagnostic only."""
    if values.size == 0:
        raise ValueError("EMPTY_RESIDUALIZATION_SAMPLE")
    finite_columns = np.std(exposures, axis=0) > 1e-12
    design = exposures[:, finite_columns]
    if design.shape[1] == 0:
        return values - values.mean(), 0.0, float("nan")
    cache_key = id(exposures)
    cached = _RESIDUALIZATION_CACHE.get(cache_key)
    if cached is None or cached[0] is not exposures:
        pseudo_inverse = np.linalg.pinv(design, rcond=1e-10)
        condition = float(np.linalg.cond(design))
        _RESIDUALIZATION_CACHE[cache_key] = (exposures, pseudo_inverse, condition)
        _RESIDUALIZATION_CACHE.move_to_end(cache_key)
        while len(_RESIDUALIZATION_CACHE) > 4:
            _RESIDUALIZATION_CACHE.popitem(last=False)
    else:
        _, pseudo_inverse, condition = cached
        _RESIDUALIZATION_CACHE.move_to_end(cache_key)
    coefficients = pseudo_inverse @ values
    fitted = design @ coefficients
    residual = values - fitted
    centered = values - values.mean()
    total = float(np.dot(centered, centered))
    r_squared = 1.0 - float(np.dot(residual, residual)) / total if total > 0 else 0.0
    return residual, r_squared, condition


def _safe_feature_end(returns_path: Path, contract: EvaluationContract) -> date:
    dates = (
        pl.scan_parquet(returns_path)
        .filter(pl.col("trade_date") < contract.holdout_start)
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()["trade_date"]
        .to_list()
    )
    if len(dates) <= contract.purge_gap_trading_days:
        raise ValueError("EVALUATION_INSUFFICIENT_DATES_FOR_PURGE_GAP")
    return dates[-contract.purge_gap_trading_days - 1]


def _load_panel(
    data_root: Path, start: date, end: date, contract: EvaluationContract,
) -> tuple[pl.DataFrame, list[str], date]:
    prices_path = data_root / "silver" / "prices_daily" / "data.parquet"
    returns_path = data_root / "silver" / "returns_daily" / "data.parquet"
    exposure_path = (
        data_root / "gold" / "risk_exposure_matrix"
        / "run_id=l1_risk_exposure_history_20260814_e3b192fa61031760"
        / "risk_exposure_matrix_v1.parquet"
    )
    if not all(path.exists() for path in (prices_path, returns_path, exposure_path)):
        raise FileNotFoundError("E2E_SMOKE_REQUIRED_PARQUET_MISSING")

    safe_end = _safe_feature_end(returns_path, contract)
    if end > safe_end:
        raise ValueError(
            f"EVALUATION_PURGE_GAP_VIOLATION end={end} safe_end={safe_end} "
            f"gap={contract.purge_gap_trading_days}_trading_days"
        )

    prices = (
        pl.scan_parquet(prices_path)
        .filter(pl.col("trade_date").is_between(start, end, closed="both"))
        .select(
            "trade_date", "asset_id", "limit_up", "raw_high", "raw_low",
            "raw_close", "amount",
        )
        .with_columns(
            (
                pl.col("limit_up").is_not_null()
                & pl.col("raw_high").is_not_null()
                & pl.col("raw_low").is_not_null()
                & pl.col("raw_close").is_not_null()
                & ((pl.col("raw_high") - pl.col("limit_up")).abs() <= 1e-6)
                & ((pl.col("raw_low") - pl.col("limit_up")).abs() <= 1e-6)
                & ((pl.col("raw_close") - pl.col("limit_up")).abs() <= 1e-6)
            ).cast(pl.Float64).alias("one_word_up"),
        )
        .with_columns(
            pl.when(pl.col("amount").is_not_null())
            .then(
                1.0
                - (
                    pl.col("amount").rank("average").over("trade_date")
                    / pl.col("amount").count().over("trade_date")
                )
            )
            .otherwise(1.0)
            .alias("relative_illiquidity"),
        )
        .with_columns(
            (pl.col("one_word_up") * pl.col("relative_illiquidity")).alias("seal_proxy")
        )
        .collect()
    )

    # Labels are built by shifting within each asset.  The source is capped
    # strictly before holdout and the feature end is separately purged above.
    horizons = contract.reported_horizons
    relative_offsets = range(
        min(ALIGNMENT_SHIFT_OFFSETS),
        max(horizons) + max(ALIGNMENT_SHIFT_OFFSETS) + 1,
    )
    returns_end = min(
        contract.holdout_start - timedelta(days=1),
        end + timedelta(days=max(horizons) + max(abs(value) for value in relative_offsets) + 5),
    )
    returns_with_labels = (
        pl.scan_parquet(returns_path)
        .filter(
            pl.col("trade_date").is_between(
                start - timedelta(days=60),
                returns_end,
                closed="both",
            )
        )
        .select("trade_date", "asset_id", "total_return")
        .sort(["asset_id", "trade_date"])
        .with_columns(
            pl.when(pl.col("total_return").is_finite() & (pl.col("total_return") > -1.0))
            .then(pl.col("total_return").log1p())
            .alias("valid_log_return")
        )
        .with_columns(
            pl.col("valid_log_return")
            .rolling_sum(window_size=20, min_samples=15)
            .over("asset_id")
            .alias("reversal_log_return_20d")
        )
        .with_columns(
            (1.0 - pl.col("reversal_log_return_20d").exp()).alias("reversal_20d")
        )
        .with_columns(
            [
                pl.col("trade_date").shift(-offset).over("asset_id").alias(
                    f"relative_date_{_shift_key(offset)}"
                )
                for offset in relative_offsets
            ]
            + [
                pl.col("total_return").shift(-offset).over("asset_id").alias(
                    f"relative_return_{_shift_key(offset)}"
                )
                for offset in relative_offsets
            ]
            + [
                pl.col("reversal_20d").shift(-offset).over("asset_id").alias(
                    f"raw_signal_offset_{_shift_key(offset)}"
                )
                for offset in ALIGNMENT_SHIFT_OFFSETS
            ]
            + [
                pl.col("trade_date").shift(-offset).over("asset_id").alias(
                    f"raw_signal_date_offset_{_shift_key(offset)}"
                )
                for offset in ALIGNMENT_SHIFT_OFFSETS
            ]
        )
        .with_columns(
            [
                pl.sum_horizontal(
                    [
                        pl.col(f"relative_return_{_shift_key(offset)}")
                        for offset in range(1, horizon + 1)
                    ]
                ).alias(f"forward_return_{horizon}")
                for horizon in horizons
            ]
            + [
                pl.col("relative_date_p1").alias(f"label_start_date_{horizon}")
                for horizon in horizons
            ]
            + [
                pl.col(f"relative_date_{_shift_key(horizon)}").alias(
                    f"label_date_{horizon}"
                )
                for horizon in horizons
            ]
            + [
                pl.sum_horizontal([
                    pl.col(f"relative_return_{_shift_key(relative_offset)}")
                    for relative_offset in range(1 + shift, horizon + 1 + shift)
                ]).alias(f"label_window_offset_{_shift_key(shift)}_{horizon}")
                for horizon in (contract.primary_horizon,)
                for shift in ALIGNMENT_SHIFT_OFFSETS
            ]
            + [
                pl.col(f"relative_date_{_shift_key(1 + shift)}").alias(
                    f"label_window_start_offset_{_shift_key(shift)}_{horizon}"
                )
                for horizon in (contract.primary_horizon,)
                for shift in ALIGNMENT_SHIFT_OFFSETS
            ]
            + [
                pl.col(f"relative_date_{_shift_key(horizon + shift)}").alias(
                    f"label_window_end_offset_{_shift_key(shift)}_{horizon}"
                )
                for horizon in (contract.primary_horizon,)
                for shift in ALIGNMENT_SHIFT_OFFSETS
            ]
        )
        .filter(pl.col("trade_date").is_between(start, end, closed="both"))
        .collect()
    )

    # Permute whole dated cross-sections across the time axis.  Each source
    # day's return vector stays intact; only its date assignment changes.
    rng = np.random.default_rng(contract.placebo_seed + 1)
    dates = returns_with_labels["trade_date"].unique().sort().to_list()
    permuted_dates = list(np.asarray(dates, dtype=object)[rng.permutation(len(dates))])
    date_map = pl.DataFrame({"source_date": dates, "trade_date": permuted_dates})
    time_labels = (
        returns_with_labels
        .rename({"trade_date": "source_date"})
        .join(date_map, on="source_date", how="inner")
        .select(
            "trade_date",
            "asset_id",
            *[
                pl.col(f"forward_return_{horizon}").alias(f"time_shuffle_{horizon}")
                for horizon in (contract.primary_horizon,)
            ],
        )
    )
    # A second semantics independently permutes each asset's label history.
    # It destroys contemporaneous cross-sectional covariance, unlike the
    # whole-date permutation above.  A demeaned arm additionally removes each
    # asset's persistent unconditional return level.
    target_positions = (
        returns_with_labels
        .select("trade_date", "asset_id")
        .sort(["asset_id", "trade_date"])
        .with_columns(pl.int_range(0, pl.len()).over("asset_id").alias("asset_position"))
    )
    shuffled_sources = (
        returns_with_labels
        .select(
            "asset_id", "trade_date",
            *[f"forward_return_{horizon}" for horizon in horizons],
        )
        .with_columns(
            pl.struct("asset_id", "trade_date")
            .hash(seed=contract.placebo_seed + 2)
            .alias("shuffle_key")
        )
        .sort(["asset_id", "shuffle_key"])
        .with_columns(pl.int_range(0, pl.len()).over("asset_id").alias("asset_position"))
        .with_columns([
            (
                pl.col(f"forward_return_{horizon}")
                - pl.col(f"forward_return_{horizon}").mean().over("asset_id")
            ).alias(f"demeaned_source_{horizon}")
            for horizon in horizons
        ])
        .select(
            "asset_id", "asset_position",
            *[
                pl.col(f"forward_return_{horizon}").alias(
                    f"independent_time_shuffle_{horizon}"
                )
                for horizon in (contract.primary_horizon,)
            ],
            *[
                pl.col(f"demeaned_source_{horizon}").alias(
                    f"independent_demeaned_time_shuffle_{horizon}"
                )
                for horizon in (contract.primary_horizon,)
            ],
        )
    )
    independent_time_labels = target_positions.join(
        shuffled_sources, on=["asset_id", "asset_position"], how="left", validate="1:1"
    ).drop("asset_position")
    returns = (
        returns_with_labels
        .select(
            "trade_date", "asset_id", "reversal_20d",
            *[
                f"raw_signal_offset_{_shift_key(offset)}"
                for offset in ALIGNMENT_SHIFT_OFFSETS
            ],
            *[
                f"raw_signal_date_offset_{_shift_key(offset)}"
                for offset in ALIGNMENT_SHIFT_OFFSETS
            ],
            *[f"label_start_date_{horizon}" for horizon in horizons],
            *[f"label_date_{horizon}" for horizon in horizons],
            *[f"forward_return_{horizon}" for horizon in horizons],
            *[
                f"label_window_offset_{_shift_key(offset)}_{horizon}"
                for offset in ALIGNMENT_SHIFT_OFFSETS
                for horizon in (contract.primary_horizon,)
            ],
            *[
                f"label_window_start_offset_{_shift_key(offset)}_{horizon}"
                for offset in ALIGNMENT_SHIFT_OFFSETS
                for horizon in (contract.primary_horizon,)
            ],
            *[
                f"label_window_end_offset_{_shift_key(offset)}_{horizon}"
                for offset in ALIGNMENT_SHIFT_OFFSETS
                for horizon in (contract.primary_horizon,)
            ],
        )
        .join(time_labels, on=["trade_date", "asset_id"], how="left")
        .join(independent_time_labels, on=["trade_date", "asset_id"], how="left")
    )

    exposure_schema = pl.read_parquet_schema(exposure_path)
    industry_columns = sorted(
        column for column in exposure_schema
        if column.startswith("risk_industry_")
    )
    board_columns = sorted(
        column for column in exposure_schema if column.startswith("risk_board_")
    )
    risk_columns = ["risk_country", *industry_columns, *board_columns, *RISK_TAIL]
    exposure_rows = (
        pl.scan_parquet(exposure_path)
        .filter(pl.col("trade_date").is_between(
            start - timedelta(days=21), end + timedelta(days=21), closed="both"
        ))
        .select(["trade_date", "asset_id", *risk_columns])
        .collect()
    )
    exposure_dates = exposure_rows["trade_date"].unique().sort().to_list()
    exposure: pl.DataFrame | None = None
    for offset in (0, *[value for value in EXPOSURE_SHIFT_OFFSETS if value != 0]):
        key = _shift_key(offset)
        mapping = _global_exposure_date_mapping(
            exposure_dates, start=start, end=end, offset=offset
        )
        shifted = (
            mapping.join(
                exposure_rows.rename({"trade_date": f"exposure_date_{key}"}),
                on=f"exposure_date_{key}", how="inner", validate="1:m",
            )
            .select(
                "trade_date", "asset_id", f"exposure_date_{key}",
                *[
                    pl.col(column).alias(f"exposure_{key}_{column}")
                    for column in risk_columns
                ],
            )
        )
        if offset == 0:
            exposure = shifted.with_columns([
                pl.col(f"exposure_{key}_{column}").alias(column)
                for column in risk_columns
            ])
        else:
            if exposure is None:
                raise RuntimeError("EVALUATION_BASELINE_EXPOSURE_MAPPING_MISSING")
            exposure = exposure.join(
                shifted, on=["trade_date", "asset_id"], how="left", validate="1:1"
            )
    if exposure is None:
        raise RuntimeError("EVALUATION_EXPOSURE_MAPPING_EMPTY")
    exposure = exposure.with_columns([
        pl.col(f"exposure_m1_{column}").alias(f"shifted_{column}")
        for column in risk_columns
    ])

    panel = prices.join(returns, on=["trade_date", "asset_id"], how="inner").join(
        exposure, on=["trade_date", "asset_id"], how="inner"
    )
    if panel.is_empty():
        raise ValueError("E2E_SMOKE_EMPTY_PANEL")
    for horizon in horizons:
        if panel.filter(pl.col(f"label_date_{horizon}") <= pl.col("trade_date")).height:
            raise AssertionError(
                f"E2E_SMOKE_FORWARD_{horizon}_LABEL_NOT_STRICTLY_AFTER_FEATURE"
            )
        if panel.filter(pl.col(f"label_date_{horizon}") >= contract.holdout_start).height:
            raise AssertionError(f"EVALUATION_HOLDOUT_LABEL_LEAK_HORIZON_{horizon}")
    return panel, risk_columns, safe_end


def attach_next_day_execution_eligibility(
    panel: pl.DataFrame,
    data_root: Path,
    *,
    primary_horizon: int,
    liquidity_floor_quantile: float = 0.20,
) -> pl.DataFrame:
    """Attach T3 facts using the next market day, never the signal day."""
    if not 0.0 < liquidity_floor_quantile < 1.0:
        raise ValueError("EVALUATION_EXECUTION_LIQUIDITY_QUANTILE_INVALID")
    universe_path = data_root / TRADABLE_UNIVERSE_PATH
    prices_path = data_root / "silver" / "prices_daily" / "data.parquet"
    if not universe_path.exists() or not prices_path.exists():
        raise FileNotFoundError("EVALUATION_EXECUTION_FACTS_MISSING")
    execution_date_column = f"label_start_date_{primary_horizon}"
    execution_dates = panel[execution_date_column].drop_nulls()
    start = execution_dates.min()
    end = execution_dates.max()
    if start is None or end is None:
        raise ValueError("EVALUATION_EXECUTION_DATE_RANGE_EMPTY")

    execution = (
        pl.scan_parquet(universe_path)
        .filter(pl.col("trade_date").is_between(start, end, closed="both"))
        .select(
            pl.col("trade_date").alias("execution_trade_date"),
            "asset_id",
            "listed_as_of",
            "is_st",
            "is_suspended",
            "board_id",
            "days_since_exchange_list",
            "base_tradable_without_listing_age",
        )
        .join(
            pl.scan_parquet(prices_path)
            .filter(pl.col("trade_date").is_between(start, end, closed="both"))
            .select(
                pl.col("trade_date").alias("execution_trade_date"),
                "asset_id", "limit_up", "limit_down", "raw_open", "raw_high",
                "raw_low", "raw_close",
            ),
            on=["execution_trade_date", "asset_id"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("board_id") == "STAR").then(pl.lit(35))
            .when(pl.col("board_id") == "BSE").then(pl.lit(25))
            .when(
                (pl.col("board_id") == "CHINEXT")
                & (pl.col("execution_trade_date") < date(2020, 8, 24))
            ).then(pl.lit(75))
            .when(pl.col("board_id") == "CHINEXT").then(pl.lit(43))
            .when(
                (pl.col("board_id") == "MAIN")
                & (pl.col("execution_trade_date") < date(2023, 4, 10))
            ).then(pl.lit(75))
            .when(pl.col("board_id") == "MAIN").then(pl.lit(46))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("execution_d_star")
        )
        .with_columns(
            (
                pl.col("limit_up").is_not_null()
                & (pl.col("raw_open") >= pl.col("limit_up"))
                & (pl.col("raw_high") >= pl.col("limit_up"))
                & (pl.col("raw_low") >= pl.col("limit_up"))
                & (pl.col("raw_close") >= pl.col("limit_up"))
            ).alias("execution_one_word_limit_up"),
            (
                pl.col("limit_down").is_not_null()
                & (pl.col("raw_open") <= pl.col("limit_down"))
                & (pl.col("raw_high") <= pl.col("limit_down"))
                & (pl.col("raw_low") <= pl.col("limit_down"))
                & (pl.col("raw_close") <= pl.col("limit_down"))
            ).alias("execution_one_word_limit_down"),
        )
        .with_columns(
            (
                pl.col("base_tradable_without_listing_age")
                & pl.col("execution_d_star").is_not_null()
                & (pl.col("days_since_exchange_list") >= pl.col("execution_d_star"))
            ).alias("execution_core_eligible")
        )
        .with_columns(
            (
                pl.col("execution_core_eligible")
                & ~pl.col("execution_one_word_limit_up")
            ).alias("execution_can_buy"),
            (
                pl.col("execution_core_eligible")
                & ~pl.col("execution_one_word_limit_down")
            ).alias("execution_can_sell"),
        )
        .collect()
    )
    return (
        panel
        .with_columns(
            pl.when(pl.col("amount").is_finite() & (pl.col("amount") > 0))
            .then(
                pl.col("amount").rank("average").over("trade_date")
                / pl.col("amount").count().over("trade_date")
            )
            .otherwise(None)
            .alias("signal_date_amount_percentile")
        )
        .join(
            execution,
            left_on=[execution_date_column, "asset_id"],
            right_on=["execution_trade_date", "asset_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col(execution_date_column).alias("execution_trade_date")
        )
        .with_columns(
            (
                pl.col("execution_can_buy").fill_null(False)
                & pl.col("execution_can_sell").fill_null(False)
                & (
                    pl.col("signal_date_amount_percentile")
                    >= liquidity_floor_quantile
                ).fill_null(False)
            ).alias("execution_eligible")
        )
    )
