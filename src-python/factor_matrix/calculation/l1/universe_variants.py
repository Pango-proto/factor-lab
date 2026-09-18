"""Apply listing-age membership variants without changing exposure formulas."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import polars as pl


def _derived_threshold_expression(derivation: pl.DataFrame) -> pl.Expr:
    required = {
        "board_id", "regime_start", "d_star", "fallback_days", "stable",
    }
    missing = sorted(required - set(derivation.columns))
    if missing:
        raise ValueError(f"DERIVED_LISTING_WINDOW_COLUMNS_MISSING columns={','.join(missing)}")
    expression: pl.Expr = pl.lit(None, dtype=pl.Int64)
    rows = derivation.sort(["board_id", "regime_start"]).to_dicts()
    by_board: dict[str, list[dict]] = {}
    for row in rows:
        by_board.setdefault(row["board_id"], []).append(row)
    for board, regimes in by_board.items():
        for index, row in enumerate(regimes):
            start = row["regime_start"]
            if isinstance(start, str):
                start = date.fromisoformat(start)
            # The earliest measured regime is also the conservative fallback for
            # incumbent securities listed before the research sample begins.
            condition = pl.col("board_id") == board
            if index > 0:
                condition &= pl.col("exchange_list_date") >= pl.lit(start)
            if index + 1 < len(regimes):
                end = regimes[index + 1]["regime_start"]
                if isinstance(end, str):
                    end = date.fromisoformat(end)
                condition &= pl.col("exchange_list_date") < pl.lit(end)
            threshold = row["d_star"] if row["stable"] else row["fallback_days"]
            if threshold is None:
                raise ValueError(f"DERIVED_LISTING_WINDOW_HAS_NO_THRESHOLD regime={row}")
            expression = pl.when(condition).then(pl.lit(int(threshold))).otherwise(expression)
    return expression


def apply_universe_variant(
    universe: pl.DataFrame | pl.LazyFrame,
    *,
    universe_variant: str,
    derived_listing_window: pl.DataFrame | None = None,
    policy_path: Path | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    required = {
        "trade_date", "asset_id", "board_id", "exchange_list_date",
        "days_since_exchange_list", "base_tradable_without_listing_age",
    }
    columns = (
        universe.collect_schema().names()
        if isinstance(universe, pl.LazyFrame)
        else universe.columns
    )
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"UNIVERSE_VARIANT_COLUMNS_MISSING columns={','.join(missing)}")
    if universe_variant == "d120":
        threshold = pl.lit(120, dtype=pl.Int64)
    elif universe_variant == "derived":
        if derived_listing_window is None:
            raise ValueError("DERIVED_UNIVERSE_VARIANT_REQUIRES_WINDOW_ARTIFACT")
        threshold = _derived_threshold_expression(derived_listing_window)
    elif universe_variant == "frozen_d0":
        if policy_path is None:
            raise ValueError("FROZEN_D0_UNIVERSE_VARIANT_REQUIRES_POLICY")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        d0 = policy["d0_model_exclusion"]
        if d0.get("status") != "frozen" or not isinstance(d0.get("value"), int):
            raise RuntimeError("FROZEN_D0_UNIVERSE_VARIANT_REQUIRES_FROZEN_VALUE")
        threshold = pl.lit(int(d0["value"]), dtype=pl.Int64)
    else:
        raise ValueError(f"UNIVERSE_VARIANT_UNKNOWN variant={universe_variant}")
    return universe.with_columns(
        pl.lit(universe_variant).alias("universe_variant"),
        threshold.alias("listing_age_threshold"),
    ).with_columns(
        (
            pl.col("base_tradable_without_listing_age")
            & pl.col("listing_age_threshold").is_not_null()
            & (pl.col("days_since_exchange_list") >= pl.col("listing_age_threshold"))
        ).alias("is_variant_tradable")
    )


def summarize_variant_sensitivity(
    universe: pl.DataFrame | pl.LazyFrame,
    *,
    derived_listing_window: pl.DataFrame,
) -> pl.DataFrame:
    """Compare membership only; exposure calculations are deliberately absent."""
    lazy = universe.lazy() if isinstance(universe, pl.DataFrame) else universe
    base = lazy.with_columns(
        pl.lit(120, dtype=pl.Int64).alias("d120_threshold"),
        _derived_threshold_expression(derived_listing_window).alias("derived_threshold"),
    ).with_columns(
        (
            pl.col("base_tradable_without_listing_age")
            & (pl.col("days_since_exchange_list") >= pl.col("d120_threshold"))
        ).alias("eligible_d120"),
        (
            pl.col("base_tradable_without_listing_age")
            & pl.col("derived_threshold").is_not_null()
            & (pl.col("days_since_exchange_list") >= pl.col("derived_threshold"))
        ).alias("eligible_derived"),
    )

    def metrics() -> list[pl.Expr]:
        return [
            pl.len().alias("rows"),
            pl.col("base_tradable_without_listing_age").sum().alias("base_eligible_rows"),
            pl.col("eligible_d120").sum().alias("d120_eligible_rows"),
            pl.col("eligible_derived").sum().alias("derived_eligible_rows"),
            (pl.col("eligible_derived") & ~pl.col("eligible_d120")).sum().alias(
                "rows_added_by_derived"
            ),
            (pl.col("eligible_d120") & ~pl.col("eligible_derived")).sum().alias(
                "rows_removed_by_derived"
            ),
            pl.col("asset_id").filter(pl.col("eligible_d120")).n_unique().alias(
                "d120_assets"
            ),
            pl.col("asset_id").filter(pl.col("eligible_derived")).n_unique().alias(
                "derived_assets"
            ),
        ]

    by_board = base.group_by("board_id").agg(metrics())
    overall = base.select(metrics()).with_columns(pl.lit("ALL").alias("board_id"))
    return pl.concat([overall, by_board], how="diagonal_relaxed").collect().sort("board_id")
