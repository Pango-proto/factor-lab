"""Canonical point-in-time listing-board classification and quality checks."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import polars as pl

from .revisioned_silver import publish_revisioned_batch, read_as_of_frame
from .storage import DataLake, json_hash, utc_now


BOARD_IDS = ("MAIN", "CHINEXT", "STAR", "BSE")
BOARD_LABELS = {
    "MAIN": "沪深主板",
    "CHINEXT": "创业板",
    "STAR": "科创板",
    "BSE": "北交所",
}
BOARD_CLASSIFICATION_VERSION = "tushare_market_pit_v1"


def board_id_expr() -> pl.Expr:
    """Return the single canonical Polars expression used to classify securities."""
    market = pl.col("market").cast(pl.String).fill_null("")
    exchange = pl.col("exchange").cast(pl.String).fill_null("")
    asset_id = pl.col("asset_id").cast(pl.String).fill_null("")
    symbol = pl.col("symbol").cast(pl.String).fill_null("")
    return (
        pl.when((market == "北交所") | (exchange == "BSE") | asset_id.str.ends_with(".BJ"))
        .then(pl.lit("BSE"))
        .when(market == "科创板")
        .then(pl.lit("STAR"))
        .when(market == "创业板")
        .then(pl.lit("CHINEXT"))
        .when(market == "主板")
        .then(pl.lit("MAIN"))
        .when(symbol.str.starts_with("688") | symbol.str.starts_with("689"))
        .then(pl.lit("STAR"))
        .when(symbol.str.starts_with("300") | symbol.str.starts_with("301"))
        .then(pl.lit("CHINEXT"))
        .when(exchange.is_in(["SSE", "SZSE", "SH", "SZ"]))
        .then(pl.lit("MAIN"))
        .otherwise(pl.lit("UNKNOWN"))
        .alias("board_id")
    )


def board_source_expr() -> pl.Expr:
    market = pl.col("market").cast(pl.String).fill_null("")
    return (
        pl.when(market.is_in(["主板", "创业板", "科创板", "北交所"]))
        .then(pl.lit("tushare_stock_basic_market"))
        .otherwise(pl.lit("exchange_symbol_fallback"))
        .alias("classification_source")
    )


def board_case_sql(*, market: str = "market", exchange: str = "exchange", asset: str = "asset_id", symbol: str = "symbol") -> str:
    """DuckDB equivalent of :func:`board_id_expr`; keep SQL call sites declarative."""
    return f"""CASE
        WHEN {market} = '北交所' OR {exchange} = 'BSE' OR {asset} LIKE '%.BJ' THEN 'BSE'
        WHEN {market} = '科创板' THEN 'STAR'
        WHEN {market} = '创业板' THEN 'CHINEXT'
        WHEN {market} = '主板' THEN 'MAIN'
        WHEN {symbol} LIKE '688%' OR {symbol} LIKE '689%' THEN 'STAR'
        WHEN {symbol} LIKE '300%' OR {symbol} LIKE '301%' THEN 'CHINEXT'
        WHEN {exchange} IN ('SSE', 'SZSE', 'SH', 'SZ') THEN 'MAIN'
        ELSE 'UNKNOWN'
    END"""


def normalize_board_membership(security: pl.DataFrame) -> pl.DataFrame:
    required = {"asset_id", "exchange", "list_date", "delist_date"}
    missing = required - set(security.columns)
    if missing:
        raise RuntimeError(f"BOARD_SECURITY_SCHEMA_MISSING {sorted(missing)}")
    frame = security
    for column in ("market", "symbol"):
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(column))
    if "ingested_at" in frame.columns:
        frame = frame.sort("ingested_at").unique("asset_id", keep="last")
    else:
        frame = frame.unique("asset_id", keep="last")
    ingested_at = utc_now()
    return (
        frame.with_columns(board_id_expr(), board_source_expr())
        .with_columns(
            pl.col("list_date").alias("effective_from"),
            pl.col("delist_date").alias("effective_to"),
            pl.lit(BOARD_CLASSIFICATION_VERSION).alias("classification_version"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("board_ingested_at"),
        )
        .select(
            "asset_id",
            "board_id",
            "effective_from",
            "effective_to",
            "classification_source",
            "classification_version",
            "source_id",
            pl.col("board_ingested_at").alias("ingested_at"),
        )
        .sort(["asset_id", "effective_from"])
    )


def board_as_of(lake: DataLake, as_of: date) -> pl.DataFrame:
    try:
        history = read_as_of_frame(lake, "board_membership_history")
    except FileNotFoundError:
        raise RuntimeError("PIT_BOARD_NOT_LOADED")
    active = history.filter(
        (pl.col("effective_from") <= as_of)
        & (pl.col("effective_to").is_null() | (pl.col("effective_to") > as_of))
    )
    duplicates = active.group_by("asset_id").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise RuntimeError(f"PIT_BOARD_AMBIGUOUS as_of={as_of} assets={duplicates.height}")
    return active.unique("asset_id", keep="last")


def resolve_board_membership(
    lake: DataLake, as_of: date, security: pl.DataFrame
) -> pl.DataFrame:
    """Read the registered PIT table, with one deterministic legacy fallback.

    The fallback keeps isolated unit fixtures and pre-migration commands usable;
    the daily production pipeline always materializes and quality-gates the table.
    """
    path = lake.silver / "board_membership_history" / "data.parquet"
    if path.exists():
        return board_as_of(lake, as_of)
    return normalize_board_membership(security).filter(
        (pl.col("effective_from") <= as_of)
        & (pl.col("effective_to").is_null() | (pl.col("effective_to") > as_of))
    )


def board_quality(lake: DataLake, as_of: date) -> dict[str, Any]:
    try:
        state = read_as_of_frame(lake, "security_daily_state").filter(
            (pl.col("trade_date") == as_of) & pl.col("in_a_share_scope")
        )
    except FileNotFoundError:
        state = pl.DataFrame()
    if not state.is_empty():
        listed = state.select("asset_id")
        joined = state.select("asset_id", "board_id").with_columns(
            pl.lit("security_daily_state").alias("classification_source")
        )
        duplicates = state.height - state.unique("asset_id").height
    else:
        memberships = board_as_of(lake, as_of)
        security = read_as_of_frame(lake, "security_master")
        if "ingested_at" in security.columns:
            security = security.sort("ingested_at").unique("asset_id", keep="last")
        listed = security.filter(
            (pl.col("list_date") <= as_of)
            & (pl.col("delist_date").is_null() | (pl.col("delist_date") > as_of))
        ).select("asset_id")
        joined = listed.join(
            memberships.select("asset_id", "board_id", "classification_source"),
            on="asset_id", how="left",
        )
        duplicates = memberships.group_by("asset_id").len().filter(pl.col("len") > 1).height
    missing = joined.filter(pl.col("board_id").is_null() | (pl.col("board_id") == "UNKNOWN"))
    counts = {
        board_id: joined.filter(pl.col("board_id") == board_id).height
        for board_id in BOARD_IDS
    }
    status = "passed" if missing.is_empty() and duplicates == 0 and sum(counts.values()) == listed.height else "failed"
    return {
        "schema_version": 1,
        "job": "board_quality",
        "status": status,
        "as_of_date": as_of.isoformat(),
        "classification_version": BOARD_CLASSIFICATION_VERSION,
        "listed_assets": listed.height,
        "covered_assets": joined.filter(pl.col("board_id").is_in(BOARD_IDS)).height,
        "coverage": (joined.filter(pl.col("board_id").is_in(BOARD_IDS)).height / listed.height if listed.height else 0.0),
        "duplicate_active_assets": duplicates,
        "fallback_assets": joined.filter(pl.col("classification_source") == "exchange_symbol_fallback").height,
        "unknown_assets": missing.height,
        "counts": counts,
        "unknown_samples": missing.get_column("asset_id").head(20).to_list(),
    }


def sync_board_membership(lake: DataLake, as_of: date) -> dict[str, Any]:
    observed_at = utc_now()
    try:
        state = read_as_of_frame(lake, "security_daily_state", observed_at).filter(
            (pl.col("trade_date") == as_of) & pl.col("in_a_share_scope")
        )
    except FileNotFoundError:
        state = pl.DataFrame()
    if not state.is_empty():
        memberships = state.select(
            "asset_id", "board_id",
            pl.col("exchange_list_date").alias("effective_from"),
            pl.col("delist_date").alias("effective_to"),
        ).with_columns(
            pl.lit("security_daily_state").alias("classification_source"),
            pl.lit(BOARD_CLASSIFICATION_VERSION).alias("classification_version"),
            pl.lit("canonical_l0").alias("source_id"),
            pl.lit(observed_at).alias("ingested_at"),
        )
        source_snapshot_ids = sorted(
            state.get_column("source_snapshot_id").drop_nulls().unique().to_list()
        ) if "source_snapshot_id" in state.columns else [as_of.isoformat()]
    else:
        try:
            security = read_as_of_frame(lake, "security_master", observed_at)
        except FileNotFoundError as exc:
            raise RuntimeError("BOARD_SECURITY_MASTER_MISSING") from exc
        memberships = normalize_board_membership(security)
        source_snapshot_ids = ["legacy_security_master"]
    config = {
        "as_of_date": as_of.isoformat(),
        "classification_version": BOARD_CLASSIFICATION_VERSION,
        "source_contract": "security_daily_state_v1",
        "source_snapshot_ids": source_snapshot_ids,
        "storage_contract": "revisioned_silver_v1",
    }
    run_id = f"board_membership_{as_of:%Y%m%d}_{json_hash(config)[:12]}"
    manifest_path = lake.manifests / f"{run_id}.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("quality_gate", {}).get("status") == "passed":
            return payload
    try:
        existing = read_as_of_frame(lake, "board_membership_history", observed_at)
    except FileNotFoundError:
        existing = pl.DataFrame(schema=memberships.schema)
    comparison = ["board_id", "effective_to", "classification_source", "classification_version"]
    prior = existing.select(
        "asset_id", "effective_from", *[pl.col(c).alias(f"__old_{c}") for c in comparison]
    )
    changed = memberships.join(prior, on=["asset_id", "effective_from"], how="left").filter(
        pl.col("__old_board_id").is_null()
        | pl.any_horizontal(
            *[
                pl.col(column).cast(pl.String).fill_null("__NULL__")
                != pl.col(f"__old_{column}").cast(pl.String).fill_null("__NULL__")
                for column in comparison
            ]
        )
    ).select(memberships.columns)
    quality = {
        "status": "passed", "candidate_rows": memberships.height,
        "changed_rows": changed.height,
        "duplicate_rows": memberships.height - memberships.unique(
            ["asset_id", "effective_from"]
        ).height,
    }
    if quality["duplicate_rows"]:
        raise RuntimeError(f"BOARD_MEMBERSHIP_QUALITY_FAILED {quality}")
    changes, version = publish_revisioned_batch(
        lake, {"board_membership_history": changed}, effective_date=as_of,
        observed_at=observed_at, quality_gate=quality,
    )
    lake.refresh_catalog()
    report = board_quality(lake, as_of)
    report_dir = lake.metadata / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"board_{as_of:%Y%m%d}.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "board_membership_sync",
        "status": report["status"],
        "created_at": observed_at.isoformat(),
        "config": config,
        "config_hash": json_hash(config),
        "changes": changes,
        "silver_version_id": version["version_id"],
        "outputs": {"quality_report": lake.artifact_record(report_path)},
        "quality_gate": report,
    }
    lake.write_immutable_json(manifest_path, manifest)
    if report["status"] != "passed":
        raise RuntimeError(f"BOARD_QUALITY_FAILED {report}")
    return manifest
