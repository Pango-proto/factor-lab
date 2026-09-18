from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from .source import TushareResponse


def response_frame(response: TushareResponse) -> pl.DataFrame:
    if not response.items:
        return pl.DataFrame(schema={field: pl.Null for field in response.fields})
    return pl.DataFrame(response.items, schema=response.fields, orient="row", infer_schema_length=None)


def canonicalize_bse_codes(
    frame: pl.DataFrame, mapping: pl.DataFrame, column: str = "ts_code"
) -> pl.DataFrame:
    if frame.is_empty() or column not in frame.columns or mapping.is_empty():
        return frame
    lookup = mapping.select(
        pl.col("o_code").alias(column), pl.col("n_code").alias("__canonical_asset_id")
    )
    return (
        frame.join(lookup, on=column, how="left")
        .with_columns(
            pl.coalesce("__canonical_asset_id", column).alias(column)
        )
        .drop("__canonical_asset_id")
    )


def normalize_bse_mapping(frame: pl.DataFrame, ingested_at: datetime) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("o_code").alias("old_asset_id"),
        pl.col("n_code").alias("asset_id"),
        parse_yyyymmdd("list_date"),
        pl.lit("tushare").alias("source_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ).drop("o_code", "n_code")


def parse_yyyymmdd(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=False)


def normalize_security(frames: list[pl.DataFrame], ingested_at: datetime) -> pl.DataFrame:
    frame = pl.concat(frames, how="diagonal_relaxed")
    return (
        frame.with_columns(
            pl.col("ts_code").cast(pl.String).alias("asset_id"),
            parse_yyyymmdd("list_date"),
            parse_yyyymmdd("delist_date"),
            pl.lit(ingested_at).alias("ingested_at"),
            pl.lit("tushare").alias("source_id"),
        )
        .drop("ts_code")
        .select(
            "asset_id",
            "symbol",
            "name",
            "area",
            "industry",
            "market",
            "exchange",
            "curr_type",
            "list_status",
            "list_date",
            "delist_date",
            "source_id",
            "ingested_at",
        )
    )


def normalize_calendar(frame: pl.DataFrame, ingested_at: datetime) -> pl.DataFrame:
    return frame.with_columns(
        parse_yyyymmdd("cal_date"),
        parse_yyyymmdd("pretrade_date"),
        pl.col("is_open").cast(pl.Int8),
        pl.lit("tushare").alias("source_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    )


def normalize_market(
    daily: pl.DataFrame,
    adj: pl.DataFrame,
    valuation: pl.DataFrame,
    ingested_at: datetime,
    anchor_date: str,
    limits: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    prices = daily.join(adj, on=["ts_code", "trade_date"], how="left")
    if limits is not None and not limits.is_empty():
        prices = prices.join(
            limits.select(
                "ts_code",
                "trade_date",
                pl.col("up_limit").alias("limit_up"),
                pl.col("down_limit").alias("limit_down"),
            ).unique(["ts_code", "trade_date"], keep="last"),
            on=["ts_code", "trade_date"],
            how="left",
        )
    else:
        prices = prices.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("limit_up"),
            pl.lit(None, dtype=pl.Float64).alias("limit_down"),
        )
    anchor = (
        adj.sort("trade_date")
        .group_by("ts_code")
        .agg(pl.col("adj_factor").last().alias("anchor_adj_factor"))
    )
    prices = (
        prices.join(anchor, on="ts_code", how="left")
        .with_columns(
            parse_yyyymmdd("trade_date"),
            pl.col("open").alias("raw_open"),
            pl.col("high").alias("raw_high"),
            pl.col("low").alias("raw_low"),
            pl.col("close").alias("raw_close"),
            (pl.col("open") * pl.col("adj_factor")).alias("gross_open_index"),
            (pl.col("close") * pl.col("adj_factor")).alias("gross_close_index"),
            (pl.col("close") * pl.col("adj_factor") / pl.col("anchor_adj_factor")).alias(
                "qfq_close"
            ),
        )
        .sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("gross_close_index") / pl.col("gross_close_index").shift(1).over("ts_code") - 1).alias(
                "total_return"
            ),
            pl.col("ts_code").alias("asset_id"),
            pl.col("pre_close").alias("prev_close"),
            pl.col("vol").alias("volume"),
            pl.lit(anchor_date).str.strptime(pl.Date, "%Y%m%d").alias("qfq_anchor_date"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        )
        .drop("ts_code", "pre_close", "vol", "anchor_adj_factor")
    )

    unadjusted_close = daily.select(
        "ts_code", "trade_date", pl.col("close").alias("__unadjusted_close")
    ).unique(["ts_code", "trade_date"], keep="last")
    valuations = valuation.join(
        unadjusted_close, on=["ts_code", "trade_date"], how="left"
    ).with_columns(
        parse_yyyymmdd("trade_date"),
        pl.col("ts_code").alias("asset_id"),
        (pl.col("total_share") * 10_000).alias("shares_total"),
        (pl.col("float_share") * 10_000).alias("shares_float"),
        (pl.col("free_share") * 10_000).alias("shares_free_float"),
        (pl.col("total_share") * 10_000 * pl.col("__unadjusted_close")).alias("total_mkt_cap"),
        (pl.col("float_share") * 10_000 * pl.col("__unadjusted_close")).alias("float_mkt_cap"),
        pl.lit("tushare").alias("source_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ).drop(
        "ts_code", "total_share", "float_share", "free_share", "total_mv", "circ_mv",
        "__unadjusted_close",
    )
    return prices, valuations


def normalize_price_limits(frame: pl.DataFrame, ingested_at: datetime) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "asset_id": pl.String,
                "limit_up": pl.Float64,
                "limit_down": pl.Float64,
                "source_id": pl.String,
                "ingested_at": pl.Datetime(time_zone="UTC"),
            }
        )
    return (
        frame.with_columns(
            parse_yyyymmdd("trade_date"),
            pl.col("ts_code").alias("asset_id"),
            pl.col("up_limit").cast(pl.Float64).alias("limit_up"),
            pl.col("down_limit").cast(pl.Float64).alias("limit_down"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        )
        .select(
            "trade_date",
            "asset_id",
            "limit_up",
            "limit_down",
            "source_id",
            "ingested_at",
        )
        .unique(["trade_date", "asset_id"], keep="last")
    )


def recompute_price_series(frame: pl.DataFrame, open_dates: pl.Series | None = None) -> pl.DataFrame:
    """Recompute returns and QFQ prices across the complete loaded history.

    This prevents missing returns at incremental batch boundaries and prevents
    multiple QFQ anchor dates from coexisting in the canonical table.
    """
    anchor_date = frame.get_column("trade_date").max()
    anchors = (
        frame.sort("trade_date")
        .group_by("asset_id")
        .agg(pl.col("adj_factor").drop_nulls().last().alias("anchor_adj_factor"))
    )
    # Older canonical/test snapshots may only contain close and adj_factor.
    # Preserve their readability during the one-time schema migration; newly
    # ingested market rows always carry the complete raw OHLC contract.
    raw_open = pl.col("open") if "open" in frame.columns else pl.col("close")
    raw_high = pl.col("high") if "high" in frame.columns else pl.col("close")
    raw_low = pl.col("low") if "low" in frame.columns else pl.col("close")
    result = (
        frame.drop(
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "gross_open_index",
            "gross_close_index",
            "qfq_close",
            "qfq_anchor_date",
            "total_return",
            strict=False,
        )
        .join(anchors, on="asset_id", how="left")
        .with_columns(
            raw_open.alias("raw_open"),
            raw_high.alias("raw_high"),
            raw_low.alias("raw_low"),
            pl.col("close").alias("raw_close"),
            (raw_open * pl.col("adj_factor")).alias("gross_open_index"),
            (pl.col("close") * pl.col("adj_factor")).alias("gross_close_index"),
            (pl.col("close") * pl.col("adj_factor") / pl.col("anchor_adj_factor")).alias(
                "qfq_close"
            ),
            pl.lit(anchor_date).cast(pl.Date).alias("qfq_anchor_date"),
        )
        .sort(["asset_id", "trade_date"])
        .with_columns(
            pl.col("trade_date").shift(1).over("asset_id").alias("previous_observation_date"),
            (
                pl.col("gross_close_index")
                / pl.col("gross_close_index").shift(1).over("asset_id")
                - 1
            ).alias("candidate_total_return"),
        )
        .drop("anchor_adj_factor")
    )
    if open_dates is None:
        return result.rename({"candidate_total_return": "total_return"}).drop(
            "previous_observation_date"
        )

    calendar = (
        pl.DataFrame({"trade_date": open_dates})
        .unique()
        .sort("trade_date")
        .with_columns(pl.col("trade_date").shift(1).alias("expected_previous_trade_date"))
    )
    return (
        result.join(calendar, on="trade_date", how="left")
        .with_columns(
            pl.when(pl.col("previous_observation_date") == pl.col("expected_previous_trade_date"))
            .then(pl.col("candidate_total_return"))
            .otherwise(None)
            .alias("total_return")
        )
        .drop(
            "candidate_total_return",
            "previous_observation_date",
            "expected_previous_trade_date",
        )
    )


def build_returns_daily(
    prices: pl.DataFrame, suspensions: pl.DataFrame, ingested_at: datetime
) -> pl.DataFrame:
    """Build an investable daily return series with explicit suspension policy.

    Pure suspension days receive zero return only when a prior traded price is
    known. On resumption, the full price change since the last traded day is
    recognized on the resumption date.
    """
    gross_close = (
        pl.col("gross_close_index")
        if "gross_close_index" in prices.columns
        else pl.col("close") * pl.col("adj_factor")
    )
    suspension_flags = (
        suspensions.select("trade_date", "asset_id", "suspend_type")
        .unique(["trade_date", "asset_id"], keep="last")
        if not suspensions.is_empty()
        else pl.DataFrame(
            schema={"trade_date": pl.Date, "asset_id": pl.String, "suspend_type": pl.String}
        )
    )
    price_events = (
        prices.select(
            "trade_date",
            "asset_id",
            gross_close.alias("gross_price_index"),
        )
        .join(suspension_flags, on=["trade_date", "asset_id"], how="left")
        .with_columns(
            pl.lit(False).alias("is_suspended"),
            pl.when(pl.col("suspend_type") == "R")
            .then(pl.lit("resumption"))
            .otherwise(pl.lit("price"))
            .alias("return_source"),
        )
    )
    suspension_events = (
        suspension_flags.join(
            prices.select("trade_date", "asset_id"),
            on=["trade_date", "asset_id"],
            how="anti",
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("gross_price_index"),
            pl.lit(True).alias("is_suspended"),
            pl.lit("suspension").alias("return_source"),
        )
    )
    events = (
        pl.concat([price_events, suspension_events], how="diagonal_relaxed")
        .sort(["asset_id", "trade_date"])
        .with_columns(
            pl.col("gross_price_index")
            .forward_fill()
            .over("asset_id")
            .alias("last_known_gross_price")
        )
        .with_columns(
            pl.col("last_known_gross_price")
            .shift(1)
            .over("asset_id")
            .alias("previous_known_gross_price")
        )
        .with_columns(
            pl.when(pl.col("gross_price_index").is_not_null())
            .then(
                pl.col("gross_price_index") / pl.col("previous_known_gross_price") - 1
            )
            .when(pl.col("previous_known_gross_price").is_not_null())
            .then(pl.lit(0.0))
            .otherwise(None)
            .alias("total_return"),
            pl.lit("zero_during_suspension_recognize_on_resumption").alias(
                "missing_return_policy"
            ),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        )
        .drop(
            "gross_price_index",
            "last_known_gross_price",
            "previous_known_gross_price",
            "suspend_type",
        )
    )
    return events
