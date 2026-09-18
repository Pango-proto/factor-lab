"""Append-only daily L0 market ingestion for the frozen Silver base."""

from __future__ import annotations

import json
import uuid
from bisect import bisect_left, bisect_right
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .boards import board_id_expr
from .market_sources import PRICE_LIMIT_FIELDS, REQUIRED_SOURCE_FIELDS, SECURITY_FIELDS
from .normalize import (
    canonicalize_bse_codes, normalize_calendar, normalize_market,
    normalize_price_limits, normalize_security, response_frame,
)
from .revisioned_silver import SilverAsOfReader, publish_revisioned_batch
from .revisioned_silver import read_as_of_frame
from .source import TushareClient, TushareResponse
from .storage import DataLake, json_hash, open_duckdb, utc_now


STABLE_PRICE_COLUMNS = (
    "trade_date", "open", "high", "low", "close", "change", "pct_chg", "amount",
    "adj_factor", "asset_id", "prev_close", "volume", "source_id", "ingested_at",
    "limit_up", "limit_down", "raw_open", "raw_high", "raw_low", "raw_close",
    "gross_open_index", "gross_close_index",
)


def _latest_passed(lake: DataLake, trade_date: date) -> Path | None:
    pattern = f"incremental_market_{trade_date:%Y%m%d}_*.json"
    for path in sorted(lake.manifests.glob(pattern), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") == "passed" and payload.get("silver_version_id"):
            return path
    return None


def incremental_market_passed(lake: DataLake, trade_date: date) -> bool:
    return _latest_passed(lake, trade_date) is not None


def _base_mapping(lake: DataLake) -> pl.DataFrame:
    path = lake.silver / "bse_code_mapping" / "data.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"o_code": pl.String, "n_code": pl.String})
    return pl.read_parquet(path).select(
        pl.col("old_asset_id").alias("o_code"), pl.col("asset_id").alias("n_code")
    ).unique(["o_code", "n_code"])


def _normalize_status(
    frame: pl.DataFrame, mapping: pl.DataFrame, observed_at: datetime, *, st: bool,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    normalized = canonicalize_bse_codes(frame, mapping).with_columns(
        pl.col("ts_code").alias("asset_id"),
        pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
        pl.lit("tushare").alias("source_id"),
        pl.lit(observed_at).alias("ingested_at"),
    ).drop("ts_code")
    return normalized.unique(["trade_date", "asset_id"], keep="last")


def _exchange_list_dates(lake: DataLake, security: pl.DataFrame) -> pl.DataFrame:
    base = json.loads(
        (lake.metadata / "base_manifests" / "legacy_base_v1.json").read_text(encoding="utf-8")
    )
    transfer_path = (
        lake.silver / "base" / f"snapshot_id={base['snapshot_id']}"
        / "bse_transfer_batch" / "data.parquet"
    )
    transfer = (
        pl.read_parquet(transfer_path).select("asset_id", "exchange_list_date")
        if transfer_path.exists()
        else pl.DataFrame(schema={"asset_id": pl.String, "exchange_list_date": pl.Date})
    )
    return security.join(transfer, on="asset_id", how="left").with_columns(
        pl.coalesce("exchange_list_date", "list_date").alias("exchange_list_date")
    )


def build_daily_security_state(
    lake: DataLake,
    trade_date: date,
    *,
    security: pl.DataFrame,
    prices: pl.DataFrame,
    valuations: pl.DataFrame,
    limits: pl.DataFrame,
    suspensions: pl.DataFrame,
    st_rows: pl.DataFrame,
    source_snapshot_id: str,
    calendar_update: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build policy-free state facts from one internally consistent source batch."""
    security = _exchange_list_dates(lake, security).with_columns(board_id_expr())
    facts_assets: set[str] = set()
    for frame in (prices, valuations, limits, suspensions, st_rows):
        if not frame.is_empty() and "asset_id" in frame.columns:
            facts_assets.update(frame.get_column("asset_id").drop_nulls().to_list())
    scoped = security.filter(
        ((pl.col("exchange_list_date") <= trade_date)
         & (pl.col("delist_date").is_null() | (pl.col("delist_date") > trade_date)))
        | ((pl.col("exchange") == "BSE") & pl.col("asset_id").is_in(facts_assets))
    ).unique("asset_id", keep="last")

    calendar_base = read_as_of_frame(lake, "trade_calendar")
    if calendar_update is not None and not calendar_update.is_empty():
        calendar_columns = [
            column for column in ("exchange", "cal_date", "is_open", "pretrade_date")
            if column in calendar_base.columns and column in calendar_update.columns
        ]
        calendar_base = pl.concat(
            [calendar_base.select(calendar_columns), calendar_update.select(calendar_columns)],
            how="diagonal_relaxed",
        ).unique(
            ["exchange", "cal_date"], keep="last"
        )
    calendar = calendar_base.filter(
        (pl.col("is_open") == 1) & (pl.col("cal_date") <= trade_date)
    ).get_column("cal_date").unique().sort().to_list()
    price_assets = set(prices.get_column("asset_id").to_list())
    value_assets = set(valuations.get_column("asset_id").to_list())
    limit_assets = set(limits.get_column("asset_id").to_list())
    st_assets = set(st_rows.get_column("asset_id").to_list()) if not st_rows.is_empty() else set()
    suspended_assets = (
        set(suspensions.filter(pl.col("suspend_type") == "S").get_column("asset_id").to_list())
        if not suspensions.is_empty() else set()
    )
    price_map = {
        row["asset_id"]: row for row in prices.select(
            "asset_id", "raw_open", "raw_high", "raw_low", "raw_close"
        ).to_dicts()
    }
    limit_map = {
        row["asset_id"]: row for row in limits.select(
            "asset_id", "limit_up", "limit_down"
        ).to_dicts()
    }
    rows: list[dict[str, Any]] = []
    for item in scoped.to_dicts():
        asset = item["asset_id"]
        exchange_date = item["exchange_list_date"]
        in_scope = exchange_date <= trade_date and (
            item["delist_date"] is None or trade_date < item["delist_date"]
        )
        days_since = None
        if in_scope:
            days_since = max(
                0, bisect_right(calendar, trade_date) - bisect_left(calendar, exchange_date) - 1
            )
        has_price, has_value, has_limit = (
            asset in price_assets, asset in value_assets, asset in limit_assets
        )
        is_suspended = asset in suspended_assets
        price, limit = price_map.get(asset, {}), limit_map.get(asset, {})
        is_limit_up = bool(
            has_price and has_limit and price.get("raw_close") is not None
            and limit.get("limit_up") is not None
            and price["raw_close"] >= limit["limit_up"]
        )
        is_limit_down = bool(
            has_price and has_limit and price.get("raw_close") is not None
            and limit.get("limit_down") is not None
            and price["raw_close"] <= limit["limit_down"]
        )
        if not in_scope:
            observation = "PRE_EXCHANGE_NEEQ"
        elif trade_date == exchange_date:
            observation = "LISTING_DAY"
        elif has_price and has_value:
            observation = "TRADED"
        elif is_suspended:
            observation = "SUSPENDED_CONFIRMED"
        elif not has_price and not has_value:
            observation = "NONTRADING_INFERRED"
        else:
            observation = "SOURCE_INCOMPLETE"
        rows.append({
            "trade_date": trade_date,
            "asset_id": asset,
            "source_asset_id": asset,
            "source_identity_state": "CANONICALIZED_INCREMENTAL",
            "source_list_date": item["list_date"],
            "neeq_list_date": item["list_date"] if item["exchange"] == "BSE" else None,
            "exchange_list_date": exchange_date,
            "delist_date": item["delist_date"],
            "list_status": item["list_status"],
            "venue": "NEEQ" if item["exchange"] == "BSE" and not in_scope else item["exchange"],
            "board_id": item["board_id"],
            "in_a_share_scope": in_scope,
            "listed_as_of": in_scope,
            "days_since_exchange_list": days_since,
            "has_price": has_price,
            "has_valuation": has_value,
            "has_limit": has_limit,
            "is_st": asset in st_assets,
            "is_suspended": is_suspended,
            "is_limit_up": is_limit_up,
            "is_limit_down": is_limit_down,
            "observation_state": observation,
            "source_day_complete": True,
            "source_snapshot_id": source_snapshot_id,
        })
    return pl.DataFrame(rows).sort("asset_id")


def _prior_gross_prices(lake: DataLake, trade_date: date, observed_at: datetime) -> dict[str, float]:
    reader = SilverAsOfReader(lake)
    relation = reader.relation_sql("prices_daily", observed_at)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        rows = connection.execute(f"""
            SELECT asset_id, arg_max(gross_close_index, trade_date)
            FROM ({relation}) WHERE trade_date < DATE '{trade_date.isoformat()}'
              AND gross_close_index IS NOT NULL GROUP BY asset_id
        """).fetchall()
    finally:
        connection.close()
    return {asset: float(value) for asset, value in rows}


def build_incremental_returns(
    lake: DataLake,
    trade_date: date,
    prices: pl.DataFrame,
    suspensions: pl.DataFrame,
    observed_at: datetime,
) -> pl.DataFrame:
    prior = _prior_gross_prices(lake, trade_date, observed_at)
    suspension_type = (
        {row["asset_id"]: row["suspend_type"] for row in suspensions.to_dicts()}
        if not suspensions.is_empty() else {}
    )
    rows: list[dict[str, Any]] = []
    price_assets = set()
    for row in prices.select("asset_id", "gross_close_index").to_dicts():
        asset, gross = row["asset_id"], row["gross_close_index"]
        price_assets.add(asset)
        previous = prior.get(asset)
        rows.append({
            "trade_date": trade_date,
            "asset_id": asset,
            "is_suspended": False,
            "return_source": "resumption" if suspension_type.get(asset) == "R" else "price",
            "total_return": gross / previous - 1 if previous and gross is not None else None,
            "missing_return_policy": "zero_during_suspension_recognize_on_resumption",
            "source_id": "tushare",
            "ingested_at": observed_at,
        })
    if not suspensions.is_empty():
        for row in suspensions.filter(pl.col("suspend_type") == "S").to_dicts():
            asset = row["asset_id"]
            if asset in price_assets:
                continue
            rows.append({
                "trade_date": trade_date, "asset_id": asset, "is_suspended": True,
                "return_source": "suspension", "total_return": 0.0 if asset in prior else None,
                "missing_return_policy": "zero_during_suspension_recognize_on_resumption",
                "source_id": "tushare", "ingested_at": observed_at,
            })
    return pl.DataFrame(rows).sort("asset_id")


def validate_incremental_market(
    trade_date: date,
    *,
    prices: pl.DataFrame,
    valuations: pl.DataFrame,
    limits: pl.DataFrame,
    state: pl.DataFrame,
    returns: pl.DataFrame,
) -> dict[str, Any]:
    duplicate = lambda frame: frame.height - frame.unique(["trade_date", "asset_id"]).height
    incomplete = state.filter(
        pl.col("in_a_share_scope") & (pl.col("observation_state") == "SOURCE_INCOMPLETE")
    ).height
    checks = {
        "required_sources_nonempty": prices.height > 0 and valuations.height > 0 and limits.height > 0,
        "price_primary_key": duplicate(prices) == 0,
        "valuation_primary_key": duplicate(valuations) == 0,
        "limit_primary_key": duplicate(limits) == 0,
        "state_primary_key": duplicate(state) == 0,
        "return_primary_key": duplicate(returns) == 0,
        "state_source_complete": incomplete == 0,
        "adjustment_factor_valid": prices.filter(pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)).is_empty(),
        "ohlc_valid": prices.filter(
            (pl.col("raw_low") > pl.min_horizontal("raw_open", "raw_close"))
            | (pl.col("raw_high") < pl.max_horizontal("raw_open", "raw_close"))
        ).is_empty(),
        "market_cap_valid": valuations.filter(
            pl.col("float_mkt_cap").is_null() | (pl.col("float_mkt_cap") <= 0)
        ).is_empty(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "trade_date": trade_date.isoformat(),
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "prices": prices.height, "valuations": valuations.height, "limits": limits.height,
            "state": state.height, "returns": returns.height,
            "source_incomplete_state": incomplete,
        },
    }


class IncrementalMarketPipeline:
    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint
        self.sources: list[dict[str, str]] = []

    def _fetch(self, api_name: str, params: dict[str, str], fields=None) -> TushareResponse:
        response = self.client.query(api_name, params, fields)
        self.sources.append(self.lake.save_bronze(response, utc_now()))
        missing = REQUIRED_SOURCE_FIELDS.get(api_name, set()) - set(response.fields)
        if missing:
            raise RuntimeError(f"SOURCE_SCHEMA_DRIFT {api_name}: missing fields {sorted(missing)}")
        return response

    def sync(self, trade_date: date, *, force: bool = False) -> Path:
        existing = _latest_passed(self.lake, trade_date)
        if existing is not None and not force:
            return existing
        observed_at = datetime.now(UTC)
        run_id = f"incremental_market_{trade_date:%Y%m%d}_{uuid.uuid4().hex[:8]}"
        day = trade_date.strftime("%Y%m%d")
        calendar_response = self._fetch(
            "trade_cal", {"exchange": "SSE", "start_date": day, "end_date": day}
        )
        calendar = normalize_calendar(response_frame(calendar_response), observed_at)
        if calendar.is_empty() or not bool(calendar.get_column("is_open").max()):
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "incremental_market_sync",
                "status": "closed", "trade_date": trade_date.isoformat(),
                "observed_at": observed_at.isoformat(), "bronze_objects": self.sources,
            })

        security_frames = [
            response_frame(self._fetch("stock_basic", {"list_status": status}, SECURITY_FIELDS))
            for status in ("L", "D", "P")
        ]
        mapping = _base_mapping(self.lake)
        security = normalize_security(security_frames, observed_at)
        security = canonicalize_bse_codes(security, mapping, column="asset_id").unique(
            "asset_id", keep="last"
        )
        daily = canonicalize_bse_codes(
            response_frame(self._fetch("daily", {"trade_date": day})), mapping
        )
        adjustment = canonicalize_bse_codes(
            response_frame(self._fetch("adj_factor", {"trade_date": day})), mapping
        )
        valuation_raw = canonicalize_bse_codes(
            response_frame(self._fetch("daily_basic", {"trade_date": day})), mapping
        )
        limit_raw = canonicalize_bse_codes(
            response_frame(self._fetch("stk_limit", {"trade_date": day}, PRICE_LIMIT_FIELDS)), mapping
        )
        if any(frame.is_empty() for frame in (daily, adjustment, valuation_raw, limit_raw)):
            raise RuntimeError(f"SOURCE_EMPTY_REQUIRED_DATA trade_date={day}")
        prices, valuations = normalize_market(
            daily, adjustment, valuation_raw, observed_at, day, limit_raw
        )
        prices = prices.select(*STABLE_PRICE_COLUMNS)
        limits = normalize_price_limits(limit_raw, observed_at)
        suspensions = _normalize_status(
            response_frame(self._fetch("suspend_d", {"trade_date": day})),
            mapping, observed_at, st=False,
        )
        st_rows = _normalize_status(
            response_frame(self._fetch("stock_st", {"trade_date": day})),
            mapping, observed_at, st=True,
        )
        security_assets = security.get_column("asset_id").unique().to_list()
        prices = prices.filter(pl.col("asset_id").is_in(security_assets))
        valuations = valuations.filter(pl.col("asset_id").is_in(security_assets))
        limits = limits.filter(pl.col("asset_id").is_in(security_assets))
        if not suspensions.is_empty():
            suspensions = suspensions.filter(pl.col("asset_id").is_in(security_assets))
        if not st_rows.is_empty():
            st_rows = st_rows.filter(pl.col("asset_id").is_in(security_assets))
        state = build_daily_security_state(
            self.lake, trade_date, security=security, prices=prices,
            valuations=valuations, limits=limits, suspensions=suspensions,
            st_rows=st_rows, source_snapshot_id=run_id, calendar_update=calendar,
        )
        returns = build_incremental_returns(
            self.lake, trade_date, prices, suspensions, observed_at
        )
        quality = validate_incremental_market(
            trade_date, prices=prices, valuations=valuations, limits=limits,
            state=state, returns=returns,
        )
        if quality["status"] != "passed":
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "incremental_market_sync",
                "status": "failed", "trade_date": trade_date.isoformat(),
                "observed_at": observed_at.isoformat(), "bronze_objects": self.sources,
                "quality_gate": quality,
            })

        correction = force
        frames = {
            "trade_calendar": calendar, "security_master": security,
            "prices_daily": prices, "valuation_daily": valuations,
            "price_limits_daily": limits, "security_daily_state": state,
            "returns_daily": returns,
        }
        if not suspensions.is_empty():
            frames["suspensions_daily"] = suspensions
        if not st_rows.is_empty():
            frames["stock_st_daily"] = st_rows
        changes, version = publish_revisioned_batch(
            self.lake, frames, effective_date=trade_date,
            observed_at=observed_at, quality_gate=quality, correction=correction,
        )
        return self.lake.write_manifest(run_id, {
            "schema_version": 1, "run_id": run_id, "job": "incremental_market_sync",
            "status": "passed", "trade_date": trade_date.isoformat(),
            "observed_at": observed_at.isoformat(), "source_id": "tushare",
            "source_credential_fingerprint": self.token_fingerprint,
            "bronze_objects": self.sources, "quality_gate": quality,
            "silver_version_id": version["version_id"], "changes": changes,
            "content_hash": json_hash(changes),
        })
