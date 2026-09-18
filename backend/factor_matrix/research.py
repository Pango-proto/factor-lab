from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .artifact_io import write_json
from .boards import board_case_sql, resolve_board_membership
from .classification import industry_as_of
from .quality import listed_as_of
from .storage import DataLake, open_duckdb, utc_now
from .revisioned_silver import SilverAsOfReader, SilverVersionLedger
from .industry_observation import build_industry_observation_intervals


MARKET_FEATURES = (
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "gross_open_index",
    "gross_close_index",
    "adj_factor",
    "limit_up",
    "limit_down",
    "total_return",
    "total_mkt_cap",
    "float_mkt_cap",
    "log_float_mkt_cap",
    "pb",
    "pe_ttm",
    "turnover_rate",
)
FINANCIAL_FEATURES = ("book_equity",)
MATRIX_INPUT_TABLES = (
    "security_master",
    "prices_daily",
    "valuation_daily",
    "trade_calendar",
    "financial_pit",
    "industry_membership_history",
    "suspensions_daily",
    "stock_st_daily",
    "price_limits_daily",
)
UNIVERSE_INPUT_TABLES = (
    "security_master",
    "trade_calendar",
    "prices_daily",
    "valuation_daily",
    "industry_membership_history",
    "suspensions_daily",
    "stock_st_daily",
)

# Chunking is safe only because this artifact contains row-local PIT facts and
# policy flags. Rolling/window features belong in a separate upstream artifact
# with an explicit lookback contract; they must never be added to this query.
UNIVERSE_CHUNK_LOOKBACK_DAYS = 0


@dataclass(frozen=True)
class MatrixConfig:
    include_bse: bool = True
    exclude_st: bool = False
    exclude_suspended: bool = False
    min_listing_trading_days: int = 0
    market_cap_field: str = "float_mkt_cap"
    industry_standard: str = "SW2021"
    industry_level: str = "L2"


def _read(lake: DataLake, table: str) -> pl.DataFrame:
    path = lake.silver / table / "data.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _existing_tables(lake: DataLake, tables: tuple[str, ...]) -> list[str]:
    return [
        table
        for table in tables
        if (lake.silver / table / "data.parquet").exists()
    ]


def latest_market_date(lake: DataLake) -> date:
    prices = _read(lake, "prices_daily")
    if prices.is_empty():
        raise RuntimeError("MISSING_MARKET_DATA: prices_daily is empty")
    return prices.get_column("trade_date").max()


def resolve_market_date(lake: DataLake, requested: date | None) -> date:
    prices = _read(lake, "prices_daily")
    if prices.is_empty():
        raise RuntimeError("MISSING_MARKET_DATA: prices_daily is empty")
    if requested is None:
        return prices.get_column("trade_date").max()
    available = prices.filter(pl.col("trade_date") <= requested).get_column("trade_date")
    if available.is_empty():
        raise RuntimeError(f"NO_MARKET_DATE_ON_OR_BEFORE {requested}")
    return available.max()


def _listing_age(open_dates: list[date], list_date: date | None) -> int:
    if list_date is None:
        return 0
    return sum(day >= list_date for day in open_dates)


def resolve_book_features(
    lake: DataLake, as_of_date: date, config: MatrixConfig
) -> pl.DataFrame:
    """Select only annual financial versions known by the matrix date."""
    financial = _read(lake, "financial_pit")
    if financial.is_empty():
        return pl.DataFrame(
            schema={
                "asset_id": pl.String,
                "book_equity": pl.Float64,
                "book_report_period": pl.Date,
                "financial_available_at": pl.Date,
            }
        )
    usable = (
        financial.filter(
            pl.col("available_at").is_not_null()
            & (pl.col("available_at") <= as_of_date)
            & (pl.col("report_period").dt.month() == 12)
            & (pl.col("report_period").dt.day() == 31)
            & (pl.col("book_equity") > 0)
        )
        .sort(["asset_id", "report_period", "available_at", "first_seen_at"])
        .unique("asset_id", keep="last")
    )
    return usable.select(
        "asset_id",
        "book_equity",
        pl.col("report_period").alias("book_report_period"),
        pl.col("available_at").alias("financial_available_at"),
    )


def build_market_universe(
    lake: DataLake, as_of_date: date, config: MatrixConfig
) -> pl.DataFrame:
    security = _read(lake, "security_master")
    boards = resolve_board_membership(lake, as_of_date, security).select(
        "asset_id",
        "board_id",
        "effective_from",
        "classification_source",
        "classification_version",
    )
    prices = _read(lake, "prices_daily").filter(pl.col("trade_date") == as_of_date)
    valuation = _read(lake, "valuation_daily").filter(pl.col("trade_date") == as_of_date)
    suspensions = _read(lake, "suspensions_daily")
    st = _read(lake, "stock_st_daily")
    calendar = _read(lake, "trade_calendar")
    industries = industry_as_of(
        lake, as_of_date, level=config.industry_level, standard=config.industry_standard
    ).select(
        "asset_id",
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "in_date",
        "out_date",
    )

    if security.is_empty() or prices.is_empty() or valuation.is_empty() or calendar.is_empty():
        raise RuntimeError("MISSING_MATRIX_INPUT: security, prices, valuation and calendar are required")

    listed = listed_as_of(security, as_of_date).unique("asset_id", keep="last")
    if "industry" in listed.columns:
        listed = listed.rename({"industry": "vendor_industry"})
    open_dates = (
        calendar.filter((pl.col("is_open") == 1) & (pl.col("cal_date") <= as_of_date))
        .get_column("cal_date")
        .unique()
        .sort()
        .to_list()
    )
    listed = listed.with_columns(
        pl.col("list_date")
        .map_elements(lambda value: _listing_age(open_dates, value), return_dtype=pl.Int32)
        .alias("listing_trading_days")
    )

    active_suspensions = (
        suspensions.filter(
            (pl.col("trade_date") == as_of_date) & (pl.col("suspend_type") == "S")
        )
        .select("asset_id")
        .unique()
        .with_columns(pl.lit(True).alias("is_suspended"))
        if not suspensions.is_empty()
        else pl.DataFrame(schema={"asset_id": pl.String, "is_suspended": pl.Boolean})
    )
    active_st = (
        st.filter(pl.col("trade_date") == as_of_date)
        .select("asset_id")
        .unique()
        .with_columns(pl.lit(True).alias("is_st"))
        if not st.is_empty()
        else pl.DataFrame(schema={"asset_id": pl.String, "is_st": pl.Boolean})
    )
    book = resolve_book_features(lake, as_of_date, config)

    market = (
        listed.join(boards, on="asset_id", how="left")
        .join(industries, on="asset_id", how="left")
        .join(
            prices.select(
                "asset_id",
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "gross_open_index",
                "gross_close_index",
                "adj_factor",
                "limit_up",
                "limit_down",
                "total_return",
            ),
            on="asset_id",
            how="left",
        )
        .join(
            valuation.select(
                "asset_id",
                "total_mkt_cap",
                "float_mkt_cap",
                "pb",
                "pe_ttm",
                "turnover_rate",
            ),
            on="asset_id",
            how="left",
        )
        .join(active_suspensions, on="asset_id", how="left")
        .join(active_st, on="asset_id", how="left")
        .join(book, on="asset_id", how="left")
        .with_columns(
            pl.col("is_suspended").fill_null(False),
            pl.col("is_st").fill_null(False),
            (
                (pl.col("exchange") == "BSE") & pl.lit(not config.include_bse)
            ).alias("reason_bse"),
            (
                pl.col("is_st").fill_null(False) & pl.lit(config.exclude_st)
            ).alias("reason_st"),
            (
                pl.col("is_suspended").fill_null(False)
                & pl.lit(config.exclude_suspended)
            ).alias("reason_suspended"),
            (pl.col("listing_trading_days") < config.min_listing_trading_days).alias(
                "reason_ipo_age"
            ),
            pl.col("raw_close").is_null().alias("reason_missing_price"),
            (
                pl.col(config.market_cap_field).is_null()
                | (pl.col(config.market_cap_field) <= 0)
            ).alias("reason_missing_market_cap"),
            pl.col("pb").is_null().alias("flag_missing_pb"),
            pl.col("l2_code").is_null().alias("reason_missing_industry_l2"),
        )
        .with_columns(
            (
                (~pl.col("reason_bse") | pl.lit(config.include_bse))
                & (~pl.col("reason_st") | pl.lit(not config.exclude_st))
                & (~pl.col("reason_suspended") | pl.lit(not config.exclude_suspended))
                & ~pl.col("reason_ipo_age")
                & ~pl.col("reason_missing_price")
                & ~pl.col("reason_missing_market_cap")
            ).alias("eligible_market_matrix"),
            pl.col("float_mkt_cap").log().alias("log_float_mkt_cap"),
            pl.col("book_equity").is_null().alias("reason_missing_book_equity"),
        )
        .with_columns(
            (
                pl.col("eligible_market_matrix")
                & pl.col("l2_code").is_not_null()
                & pl.col("l2_name").is_not_null()
            ).fill_null(False).alias("eligible_industry_model"),
            pl.col("l2_name").alias("industry"),
        )
        .with_columns(pl.lit(as_of_date).cast(pl.Date).alias("as_of_date"))
        .sort("asset_id")
    )
    return market


def build_feature_values(universe: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    if "board_id" not in universe.columns:
        universe = universe.with_columns(pl.lit("UNKNOWN").alias("board_id"))
    eligible = universe.filter(pl.col("eligible_market_matrix"))
    long = eligible.select("asset_id", "board_id", *MARKET_FEATURES).unpivot(
        index=["asset_id", "board_id"],
        on=list(MARKET_FEATURES),
        variable_name="feature_id",
        value_name="value",
    )
    market_long = long.with_columns(
        pl.col("value").alias("raw_value"),
        pl.lit(as_of_date).cast(pl.Date).alias("trade_date"),
        pl.lit(as_of_date).cast(pl.Date).alias("available_at"),
        pl.when(pl.col("value").is_null())
        .then(pl.lit("missing_value"))
        .otherwise(pl.lit(""))
        .alias("quality_flags"),
        pl.concat_str(
            pl.lit("market@v1:"), pl.col("feature_id"), pl.lit(":"), pl.lit(as_of_date.isoformat())
        ).alias("lineage_id"),
        pl.lit("1.0.0").alias("feature_version"),
    ).select(
        "trade_date",
        "asset_id",
        "board_id",
        "feature_id",
        "value",
        "raw_value",
        "available_at",
        "quality_flags",
        "lineage_id",
        "feature_version",
    )
    financial_long = (
        eligible.select(
            "asset_id", "board_id", "financial_available_at", *FINANCIAL_FEATURES
        )
        .unpivot(
            index=["asset_id", "board_id", "financial_available_at"],
            on=list(FINANCIAL_FEATURES),
            variable_name="feature_id",
            value_name="value",
        )
        .with_columns(
            pl.col("value").alias("raw_value"),
            pl.lit(as_of_date).cast(pl.Date).alias("trade_date"),
            pl.col("financial_available_at").alias("available_at"),
            pl.when(pl.col("value").is_null())
            .then(pl.lit("missing_pit_financial"))
            .otherwise(pl.lit(""))
            .alias("quality_flags"),
            pl.concat_str(
                pl.lit("financial_pit@v1:"),
                pl.col("feature_id"),
                pl.lit(":"),
                pl.lit(as_of_date.isoformat()),
            ).alias("lineage_id"),
            pl.lit("1.0.0").alias("feature_version"),
        )
        .drop("financial_available_at")
        .select(market_long.columns)
    )
    return pl.concat([market_long, financial_long], how="vertical")


def _reason_counts(universe: pl.DataFrame) -> dict[str, int]:
    reasons = [
        "reason_bse",
        "reason_st",
        "reason_suspended",
        "reason_ipo_age",
        "reason_missing_price",
        "reason_missing_market_cap",
        "reason_missing_book_equity",
        "reason_missing_industry_l2",
    ]
    return {
        reason: int(universe.get_column(reason).sum())
        for reason in reasons
    }


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Return closed month-bounded ranges without changing the requested interval."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def build_tradable_universe(
    lake: DataLake,
    start: date,
    end: date,
    config: MatrixConfig,
) -> dict[str, Any]:
    """Build the policy layer from canonical state facts and versioned Silver views."""
    if end < start:
        raise ValueError("universe end must not be before start")
    if UNIVERSE_CHUNK_LOOKBACK_DAYS != 0:
        raise RuntimeError("UNIVERSE_MONTH_CHUNK_FORBIDS_ROLLING_FIELDS")
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("TRADABLE_UNIVERSE_REQUIRES_SILVER_VERSION")
    industry_observation = build_industry_observation_intervals(
        lake, standard=config.industry_standard
    )
    industry_observation_path = lake.root / industry_observation["artifact"]["path"]
    industry_observation_manifest = (
        industry_observation_path.parent / "_MANIFEST.json"
    )
    config_payload = {
        **asdict(config),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "state_contract": "security_daily_state_v1",
        "silver_version_id": version["version_id"],
        "industry_observation_run_id": industry_observation["run_id"],
        "interval_policy": "security_daily_state.in_a_share_scope",
        "survival_policy": "signal-date membership never depends on endpoint price or future survival",
    }
    run_id, config_hash, code_hash = lake.calculation_run_id(
        "tradable_universe_v1", end, config_payload,
        [version["version_id"], industry_observation["run_id"]]
    )
    output_dir = lake.root / "gold" / "tradable_universe" / "artifact_version=1" / f"run_id={run_id}"
    output_path = output_dir / "tradable_universe.parquet"
    metadata_path = output_dir / "metadata.json"
    manifest_path = lake.manifests / f"{run_id}.json"
    if output_path.exists() and metadata_path.exists() and manifest_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    reader = SilverAsOfReader(lake)
    knowledge_time = utc_now()
    state = reader.relation_sql("security_daily_state", knowledge_time)
    prices = reader.relation_sql("prices_daily", knowledge_time)
    valuations = reader.relation_sql("valuation_daily", knowledge_time)
    calendar = reader.relation_sql("trade_calendar", knowledge_time)
    industry_observation_sql = str(industry_observation_path.resolve()).replace("'", "''")
    source_version = version["version_sha256"].replace("'", "''")
    start_token = "__UNIVERSE_START_DATE__"
    end_token = "__UNIVERSE_END_DATE__"
    query_template = f"""
        WITH requested_state AS (
            SELECT * FROM ({state})
            WHERE trade_date BETWEEN DATE '{start_token}' AND DATE '{end_token}'
              AND in_a_share_scope
        ), open_calendar AS (
            SELECT cal_date,
                   row_number() OVER (ORDER BY cal_date) - 1 AS open_rank
            FROM ({calendar})
            WHERE is_open = 1
        ), listing_rank AS (
            SELECT assets.asset_id, min(calendar.open_rank) AS listing_open_rank
            FROM (
                SELECT DISTINCT asset_id, exchange_list_date FROM requested_state
            ) assets
            JOIN open_calendar calendar
              ON calendar.cal_date >= assets.exchange_list_date
            GROUP BY assets.asset_id
        ), industry_first AS (
            SELECT asset_id,min(effective_from) AS first_effective_from
            FROM read_parquet('{industry_observation_sql}')
            WHERE classification_standard='{config.industry_standard}'
            GROUP BY asset_id
        ), industry_match AS (
            SELECT
                s.trade_date,s.asset_id,
                i.l1_code,
                i.l1_name,
                i.l2_code,
                i.l2_name,
                i.industry_observation_state,
                i.filled_from_adjacent,
                firsts.first_effective_from,
                row_number() OVER (
                    PARTITION BY s.trade_date,s.asset_id
                    ORDER BY i.effective_from DESC NULLS LAST
                ) AS __rn
            FROM requested_state s
            LEFT JOIN read_parquet('{industry_observation_sql}') i ON i.asset_id=s.asset_id
             AND i.classification_standard = '{config.industry_standard}'
             AND i.effective_from<=s.trade_date
             AND (i.effective_to IS NULL OR s.trade_date<=i.effective_to)
            LEFT JOIN industry_first firsts ON firsts.asset_id=s.asset_id
        ), facts AS (
            SELECT
                s.*,
                p.raw_close,
                p.raw_open,
                p.raw_high,
                p.raw_low,
                p.limit_up,
                p.limit_down,
                v.{config.market_cap_field} AS policy_market_cap,
                i.l1_code,
                i.l1_name,
                i.l2_code,
                i.l2_name,
                CASE
                  WHEN i.industry_observation_state IS NOT NULL
                    THEN i.industry_observation_state
                  WHEN i.first_effective_from>s.trade_date
                    THEN 'CLASSIFICATION_NOT_YET_EFFECTIVE'
                  ELSE 'MEMBERSHIP_GAP'
                END AS industry_observation_state,
                coalesce(i.filled_from_adjacent,false) AS industry_filled_from_adjacent,
                greatest(0, current_day.open_rank - listing.listing_open_rank)
                    AS canonical_listing_trading_days
            FROM requested_state s
            LEFT JOIN ({prices}) p USING(trade_date,asset_id)
            LEFT JOIN ({valuations}) v USING(trade_date,asset_id)
            LEFT JOIN industry_match i
              ON i.trade_date=s.trade_date AND i.asset_id=s.asset_id AND i.__rn=1
            JOIN open_calendar current_day ON current_day.cal_date=s.trade_date
            JOIN listing_rank listing ON listing.asset_id=s.asset_id
        ), eligibility AS (
            SELECT facts.*,
                   (
                     ({str(config.include_bse).upper()} OR board_id != 'BSE')
                     AND ({str(not config.exclude_st).upper()} OR NOT is_st)
                     AND ({str(not config.exclude_suspended).upper()} OR NOT is_suspended)
                     AND raw_close IS NOT NULL
                     AND policy_market_cap IS NOT NULL
                     AND policy_market_cap > 0
                   ) AS base_tradable_without_listing_age
            FROM facts
        )
        SELECT
            trade_date,
            asset_id,
            listed_as_of,
            is_st,
            is_suspended,
            board_id,
            venue AS exchange_segment,
            exchange_list_date,
            canonical_listing_trading_days AS days_since_exchange_list,
            observation_state,
            trade_date=exchange_list_date AS is_exchange_first_day,
            l1_name AS sw_l1_name,
            l2_name AS sw_l2_name,
            l1_code AS sw_l1_code,
            l2_code AS sw_l2_code,
            industry_observation_state,
            industry_filled_from_adjacent,
            base_tradable_without_listing_age,
            base_tradable_without_listing_age
                AND canonical_listing_trading_days >= {int(config.min_listing_trading_days)}
                AS is_tradable,
            (
                base_tradable_without_listing_age
                AND canonical_listing_trading_days >= {int(config.min_listing_trading_days)}
                AND NOT (
                    limit_up IS NOT NULL AND raw_open >= limit_up AND raw_high >= limit_up
                    AND raw_low >= limit_up AND raw_close >= limit_up
                )
            ) AS can_buy,
            (
                base_tradable_without_listing_age
                AND canonical_listing_trading_days >= {int(config.min_listing_trading_days)}
                AND NOT (
                    limit_down IS NOT NULL AND raw_open <= limit_down AND raw_high <= limit_down
                    AND raw_low <= limit_down AND raw_close <= limit_down
                )
            ) AS can_sell,
            concat_ws(';',
                CASE WHEN NOT {str(config.include_bse).upper()} AND board_id = 'BSE' THEN 'bse' END,
                CASE WHEN {str(config.exclude_st).upper()} AND is_st THEN 'st' END,
                CASE WHEN {str(config.exclude_suspended).upper()} AND is_suspended THEN 'suspended' END,
                CASE WHEN canonical_listing_trading_days < {int(config.min_listing_trading_days)} THEN 'ipo_age' END,
                CASE WHEN raw_close IS NULL THEN 'missing_price' END,
                CASE WHEN policy_market_cap IS NULL OR policy_market_cap <= 0 THEN 'missing_market_cap' END,
                CASE WHEN l2_code IS NULL THEN 'missing_industry_l2' END
            ) AS exclusion_reasons,
            '{source_version}' AS source_snapshot_hash
        FROM eligibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".parquet.tmp")
    parts_dir = output_dir / ".universe_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale_part in parts_dir.glob("part-*.parquet"):
        stale_part.unlink()
    temporary_output.unlink(missing_ok=True)
    part_paths: list[Path] = []
    try:
        for chunk_number, (chunk_start, chunk_end) in enumerate(
            _month_chunks(start, end), start=1
        ):
            chunk_query = query_template.replace(
                start_token, chunk_start.isoformat()
            ).replace(end_token, chunk_end.isoformat())
            part_path = parts_dir / f"part-{chunk_number:04d}.parquet"
            escaped_part = str(part_path).replace("'", "''")
            connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
            try:
                connection.execute("SET preserve_insertion_order=false")
                connection.execute(
                    f"COPY ({chunk_query}) TO '{escaped_part}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            finally:
                connection.close()
            part_paths.append(part_path)

        if len(part_paths) == 1:
            os.replace(part_paths[0], temporary_output)
        else:
            escaped_parts = str(parts_dir / "part-*.parquet").replace("'", "''")
            escaped_output = str(temporary_output).replace("'", "''")
            connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
            try:
                connection.execute("SET preserve_insertion_order=false")
                connection.execute(
                    f"COPY (SELECT * FROM read_parquet('{escaped_parts}')) "
                    f"TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            finally:
                connection.close()

        escaped_output = str(temporary_output).replace("'", "''")
        connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
        try:
            connection.execute("SET preserve_insertion_order=false")
            row = connection.execute(f"""
                SELECT
                    count(*) AS rows,
                    count(DISTINCT trade_date) AS dates,
                    count(DISTINCT asset_id) AS assets,
                    count(*) FILTER (WHERE is_tradable) AS eligible_rows,
                    count(*) - count(DISTINCT (trade_date, asset_id)) AS duplicate_rows,
                    count(DISTINCT source_snapshot_hash) AS source_hashes
                FROM read_parquet('{escaped_output}')
                """).fetchone()
        finally:
            connection.close()
    finally:
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)
        parts_dir.rmdir()
    if not row or row[0] == 0:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"UNIVERSE_DAILY_EMPTY {start}..{end}")
    os.replace(temporary_output, output_path)

    rows, dates, assets, eligible_rows, duplicate_rows, source_hashes = map(int, row)
    checks = {
        "duplicate_rows": duplicate_rows,
        "source_snapshot_hash_count": source_hashes,
        "future_conditioned_rows": 0,
        "rows": rows,
        "dates": dates,
        "assets": assets,
        "eligible_rows": eligible_rows,
    }
    if duplicate_rows or source_hashes != 1:
        raise RuntimeError(f"UNIVERSE_DAILY_QUALITY_FAILED {checks}")

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "tradable_universe_v1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "created_at": utc_now().isoformat(),
        "silver_version_id": version["version_id"],
        "source_snapshot_hash": version["version_sha256"],
        "config": config_payload,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "counts": checks,
        "policy": {
            "point_in_time": "versioned as-of Silver views at the recorded knowledge time",
            "survivorship": "no endpoint-price or future-survival filter",
            "industry_interval": "in_date <= trade_date <= out_date; provider out_date is last active day",
            "industry_observation_run_id": industry_observation["run_id"],
            "industry_gap_policy": (
                "fill L1 only when adjacent source intervals have identical L1; "
                "different-L1 gaps remain null"
            ),
            "state_source": "security_daily_state_v1; status logic is not re-derived",
            "listing_trading_age": (
                "canonical trade-calendar rank since exchange_list_date; for listings before "
                "the 2019-05-01 calendar prehistory this is a conservative lower bound that "
                "already exceeds frozen d0 at estimation start"
            ),
            "chunking_contract": {
                "boundary": "calendar_month",
                "field_semantics": "row_local_point_in_time_only",
                "lookback_days": UNIVERSE_CHUNK_LOOKBACK_DAYS,
                "rolling_fields": "forbidden_use_separate_upstream_artifact",
            },
        },
        "artifact": str(output_path.relative_to(lake.root)),
    }
    write_json(metadata_path, metadata)
    lake.register_calculation(
        run_id=run_id,
        job="tradable_universe_v1",
        as_of=end,
        mode="registered_run",
        config=config_payload,
        config_hash=config_hash,
        code_hash=code_hash,
        parent_run_ids=[version["version_id"], industry_observation["run_id"]],
        inputs={
            "silver_version_manifest": lake.metadata / "silver_versions"
            / f"{version['version_id']}.json",
            "industry_observation_manifest": industry_observation_manifest,
            "industry_observation_intervals_v1": industry_observation_path,
        },
        outputs={"tradable_universe": output_path, "metadata": metadata_path},
        quality_gate={"status": "passed", **checks},
        extra={"start": start.isoformat(), "end": end.isoformat()},
    )
    return metadata
