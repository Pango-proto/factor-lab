"""Raw, label-free risk descriptors. No cross-sectional transform lives here."""

from __future__ import annotations

from datetime import date

import polars as pl


def _trailing_dates(frame: pl.DataFrame, as_of: date, count: int) -> list[date]:
    return (
        frame.filter(pl.col("trade_date") <= as_of)
        .get_column("trade_date")
        .drop_nulls()
        .unique()
        .sort()
        .tail(count)
        .to_list()
    )


def market_sensitivity_descriptors(
    *,
    as_of: date,
    assets: pl.DataFrame,
    universe_history: pl.DataFrame,
    returns: pl.DataFrame,
    board_benchmarks: pl.DataFrame,
    lookback_days: int,
    minimum_observations: int,
) -> pl.DataFrame:
    dates = _trailing_dates(board_benchmarks, as_of, lookback_days)
    if not dates:
        raise ValueError("L1_BETA_BENCHMARK_HISTORY_EMPTY")
    observations = (
        universe_history.filter(pl.col("trade_date").is_in(dates))
        .select("trade_date", "asset_id", "board_id")
        .join(assets.select("asset_id"), on="asset_id", how="inner")
        .join(
            returns.filter(
                pl.col("trade_date").is_in(dates)
                & pl.col("total_return").is_not_null()
                & (pl.col("return_source") != "resumption")
            ).select("trade_date", "asset_id", "total_return"),
            on=["trade_date", "asset_id"], how="inner", validate="1:1",
        )
        .join(
            board_benchmarks.filter(pl.col("trade_date").is_in(dates)).select(
                "trade_date", "board_id", "board_return"
            ),
            on=["trade_date", "board_id"], how="inner", validate="m:1",
        )
        .filter(pl.col("board_return").is_not_null())
    )
    aggregates = observations.group_by("asset_id").agg(
        pl.len().alias("n_beta_obs"),
        pl.col("board_return").sum().alias("sx"),
        pl.col("total_return").sum().alias("sy"),
        (pl.col("board_return") ** 2).sum().alias("sxx"),
        (pl.col("board_return") * pl.col("total_return")).sum().alias("sxy"),
        (pl.col("total_return") ** 2).sum().alias("syy"),
    ).with_columns(
        (
            (
                pl.col("n_beta_obs") * pl.col("sxy")
                - pl.col("sx") * pl.col("sy")
            )
            / (
                pl.col("n_beta_obs") * pl.col("sxx")
                - pl.col("sx") ** 2
            )
        ).alias("beta_raw")
    ).with_columns(
        (
            (pl.col("sy") - pl.col("beta_raw") * pl.col("sx"))
            / pl.col("n_beta_obs")
        ).alias("alpha_intercept")
    ).with_columns(
        (
            pl.col("syy")
            + pl.col("n_beta_obs") * pl.col("alpha_intercept") ** 2
            + pl.col("beta_raw") ** 2 * pl.col("sxx")
            - 2 * pl.col("alpha_intercept") * pl.col("sy")
            - 2 * pl.col("beta_raw") * pl.col("sxy")
            + 2 * pl.col("alpha_intercept") * pl.col("beta_raw") * pl.col("sx")
        ).clip(lower_bound=0).alias("residual_sse")
    ).with_columns(
        (
            pl.col("residual_sse") / (pl.col("n_beta_obs") - 2)
        ).sqrt().alias("residual_volatility_raw")
    ).with_columns(
        pl.when(pl.col("n_beta_obs") >= minimum_observations)
        .then(pl.col("beta_raw")).otherwise(None).alias("beta_raw"),
        pl.when(pl.col("n_beta_obs") >= max(minimum_observations, 3))
        .then(pl.col("residual_volatility_raw")).otherwise(None)
        .alias("residual_volatility_raw"),
    )
    return assets.select("asset_id").join(
        aggregates.select(
            "asset_id", "n_beta_obs", "beta_raw", "residual_volatility_raw"
        ),
        on="asset_id", how="left",
    )


def liquidity_descriptor(
    *,
    as_of: date,
    assets: pl.DataFrame,
    valuation: pl.DataFrame,
    lookback_days: int,
    minimum_observations: int,
) -> pl.DataFrame:
    dates = _trailing_dates(valuation, as_of, lookback_days)
    aggregate = (
        valuation.filter(
            pl.col("trade_date").is_in(dates)
            & pl.col("turnover_rate").is_not_null()
            & (pl.col("turnover_rate") > 0)
        )
        .join(assets.select("asset_id"), on="asset_id", how="inner")
        .group_by("asset_id")
        .agg(
            pl.len().alias("n_liquidity_obs"),
            pl.col("turnover_rate").mean().alias("mean_turnover_rate"),
        )
        .with_columns(
            pl.when(pl.col("n_liquidity_obs") >= minimum_observations)
            .then(pl.col("mean_turnover_rate").log())
            .otherwise(None)
            .alias("liquidity_raw")
        )
    )
    return assets.select("asset_id").join(
        aggregate.select("asset_id", "n_liquidity_obs", "liquidity_raw"),
        on="asset_id", how="left",
    )


def index_membership_descriptors(
    *, as_of: date, assets: pl.DataFrame, membership: pl.DataFrame,
) -> pl.DataFrame:
    active = membership.filter(
        (pl.col("effective_from") <= as_of)
        & (pl.col("effective_to").is_null() | (pl.col("effective_to") > as_of))
        & pl.col("benchmark_id").is_in(["CSI300", "CSI500"])
    ).select("asset_id", "benchmark_id").unique()
    flags = active.group_by("asset_id").agg(
        (pl.col("benchmark_id") == "CSI300").any().alias("index_CSI300"),
        (pl.col("benchmark_id") == "CSI500").any().alias("index_CSI500"),
    )
    return assets.select("asset_id").join(flags, on="asset_id", how="left").with_columns(
        pl.col("index_CSI300").fill_null(False),
        pl.col("index_CSI500").fill_null(False),
    )
