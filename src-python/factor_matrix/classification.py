"""Point-in-time Shenwan industry classifications with explicit standard."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .normalize import parse_yyyymmdd, response_frame
from .revisioned_silver import publish_revisioned_batch, read_as_of_frame
from .source import TushareClient
from .storage import DataLake, utc_now


SUPPORTED_CLASSIFICATION_STANDARDS = ("SW2014", "SW2021")


def _validate_standard(standard: str) -> str:
    normalized = standard.upper()
    if normalized not in SUPPORTED_CLASSIFICATION_STANDARDS:
        raise ValueError(f"Unsupported industry classification standard: {standard}")
    return normalized


def normalize_industry_classification(
    frame: pl.DataFrame, ingested_at: datetime, standard: str
) -> pl.DataFrame:
    standard = _validate_standard(standard)
    return (
        frame.with_columns(
            pl.col("is_pub").cast(pl.String),
            pl.lit(standard).alias("classification_standard"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
            pl.lit(ingested_at).alias("available_at"),
        )
        .rename({"index_code": "industry_index_code"})
        .unique(subset=["industry_index_code"], keep="last", maintain_order=True)
    )


def normalize_industry_membership(
    frame: pl.DataFrame, ingested_at: datetime, standard: str
) -> pl.DataFrame:
    standard = _validate_standard(standard)
    return (
        frame.with_columns(
            pl.col("ts_code").alias("asset_id"),
            parse_yyyymmdd("in_date"),
            parse_yyyymmdd("out_date"),
            (pl.col("is_new") == "Y").alias("is_current"),
            pl.lit(standard).alias("classification_standard"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
            pl.lit(ingested_at).alias("available_at"),
        )
        .drop("ts_code")
        .filter(pl.col("asset_id").is_not_null() & pl.col("in_date").is_not_null())
        .unique(
            subset=["asset_id", "l1_code", "l2_code", "l3_code", "in_date"],
            keep="last",
        )
    )


SUPPORTED_INDUSTRY_LEVELS = ("L1", "L2", "L3")


def _level_columns(level: str) -> tuple[str, str]:
    normalized = level.upper()
    if normalized not in SUPPORTED_INDUSTRY_LEVELS:
        raise ValueError(f"Unsupported SW2021 industry level: {level}")
    prefix = normalized.lower()
    return f"{prefix}_code", f"{prefix}_name"


def industry_as_of(
    lake: DataLake, as_of: date, *, level: str = "L2", standard: str = "SW2021"
) -> pl.DataFrame:
    standard = _validate_standard(standard)
    level_code, level_name = _level_columns(level)
    try:
        history = read_as_of_frame(lake, "industry_membership_history")
    except FileNotFoundError:
        raise RuntimeError("PIT_INDUSTRY_NOT_LOADED")
    memberships = history.filter(
        (pl.col("classification_standard") == standard)
        & (pl.col("in_date") <= as_of)
        & (pl.col("out_date").is_null() | (pl.col("out_date") >= as_of))
        & pl.col(level_code).is_not_null()
        & pl.col(level_name).is_not_null()
    )
    if memberships.is_empty():
        return memberships
    duplicate_assets = memberships.group_by("asset_id").len().filter(pl.col("len") > 1)
    if not duplicate_assets.is_empty():
        raise RuntimeError(
            f"PIT_INDUSTRY_AMBIGUOUS level={level.upper()} as_of={as_of} "
            f"assets={duplicate_assets.height}"
        )
    return memberships.sort("in_date").unique(subset=["asset_id"], keep="last")


def industry_quality(
    lake: DataLake, as_of: date, *, level: str = "L2", standard: str = "SW2021"
) -> dict[str, Any]:
    standard = _validate_standard(standard)
    level_code, level_name = _level_columns(level)
    normalized_level = level.upper()
    history = read_as_of_frame(lake, "industry_membership_history")
    active_raw = history.filter(
        (pl.col("classification_standard") == standard)
        & (pl.col("in_date") <= as_of)
        & (pl.col("out_date").is_null() | (pl.col("out_date") >= as_of))
        & pl.col(level_code).is_not_null()
        & pl.col(level_name).is_not_null()
    )
    duplicate_assets = (
        active_raw.group_by("asset_id").len().filter(pl.col("len") > 1).height
        if not active_raw.is_empty()
        else 0
    )
    memberships = active_raw.sort("in_date").unique(subset=["asset_id"], keep="last")
    state = read_as_of_frame(lake, "security_daily_state").filter(
        (pl.col("trade_date") == as_of)
        & pl.col("in_a_share_scope")
        & (pl.col("board_id") != "BSE")
    )
    eligible = state.select("asset_id").unique()
    covered = eligible.join(memberships.select("asset_id"), on="asset_id", how="inner").height
    total = eligible.height
    return {
        "as_of_date": as_of.isoformat(),
        "classification_standard": standard,
        "level": normalized_level,
        "eligible_assets": total,
        "covered_assets": covered,
        "coverage": covered / total if total else 0.0,
        "active_memberships": memberships.height,
        "duplicate_active_assets": duplicate_assets,
        "future_memberships": memberships.filter(pl.col("in_date") > as_of).height,
        "status": "passed" if total and covered / total >= 0.97 and duplicate_assets == 0 else "failed",
    }


class IndustryClassificationPipeline:
    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint

    def sync(self, standard: str, request_delay: float = 0.1) -> Path:
        standard = _validate_standard(standard)
        ingested_at = utc_now()
        config = {
            "standard": standard,
            "levels": list(SUPPORTED_INDUSTRY_LEVELS),
            "member_states": ["Y", "N"],
            "storage_contract": "revisioned_silver_v1",
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        run_id = f"industry_classification_{ingested_at:%Y%m%d}_{config_hash}"
        manifest_path = self.lake.manifests / f"{run_id}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("quality_gate", {}).get("status") == "passed" and existing.get(
                "silver_version_id"
            ):
                return manifest_path
        classification_frames: list[pl.DataFrame] = []
        artifacts: list[dict[str, str]] = []
        for level in SUPPORTED_INDUSTRY_LEVELS:
            classification_response = self.client.query(
                "index_classify", {"level": level, "src": standard}
            )
            artifacts.append(self.lake.save_bronze(classification_response, ingested_at))
            classification_frames.append(response_frame(classification_response))
            if request_delay:
                time.sleep(request_delay)
        classification = normalize_industry_classification(
            pl.concat(classification_frames, how="diagonal_relaxed"), ingested_at, standard
        )
        member_frames: list[pl.DataFrame] = []
        l1_codes = (
            classification.filter(pl.col("level") == "L1")
            .get_column("industry_index_code")
            .sort()
            .to_list()
        )
        for l1_code in l1_codes:
            for is_new in ("Y", "N"):
                response = self.client.query(
                    "index_member_all", {"l1_code": l1_code, "is_new": is_new}
                )
                artifacts.append(self.lake.save_bronze(response, ingested_at))
                member_frames.append(response_frame(response))
                if request_delay:
                    time.sleep(request_delay)
        raw_members = pl.concat(member_frames, how="diagonal_relaxed")
        memberships = normalize_industry_membership(raw_members, ingested_at, standard)
        classification_keys = ["classification_standard", "industry_index_code"]
        membership_keys = [
            "classification_standard", "asset_id", "l1_code", "l2_code", "l3_code", "in_date",
        ]
        quality = {
            "status": "passed",
            "classification_rows": classification.height,
            "membership_rows": memberships.height,
            "classification_duplicate_rows": classification.height - classification.unique(
                classification_keys
            ).height,
            "membership_duplicate_rows": memberships.height - memberships.unique(
                membership_keys
            ).height,
        }
        if (
            not classification.height or not memberships.height
            or quality["classification_duplicate_rows"] or quality["membership_duplicate_rows"]
        ):
            quality["status"] = "failed"
            raise RuntimeError(f"INDUSTRY_CLASSIFICATION_QUALITY_FAILED {quality}")
        changes, version = publish_revisioned_batch(
            self.lake,
            {
                "industry_classification": classification,
                "industry_membership_history": memberships,
            },
            effective_date=ingested_at.date(), observed_at=ingested_at,
            quality_gate=quality,
        )
        self.lake.refresh_catalog()
        return self.lake.write_manifest(
            run_id,
            {
                "run_id": run_id,
                "job": "industry_classification_sync",
                "status": "passed",
                "created_at": ingested_at.isoformat(),
                "config": config,
                "config_hash": config_hash,
                "classification_standard": standard,
                "interval_policy": "in_date <= as_of <= out_date; provider out_date is last active day",
                "counts": {
                    "classification_rows": classification.height,
                    "classification_rows_by_level": {
                        level: classification.filter(pl.col("level") == level).height
                        for level in SUPPORTED_INDUSTRY_LEVELS
                    },
                    "membership_rows": memberships.height,
                    "assets": memberships.get_column("asset_id").n_unique(),
                    "assets_with_l2": memberships.filter(
                        pl.col("l2_code").is_not_null() & pl.col("l2_name").is_not_null()
                    ).get_column("asset_id").n_unique(),
                },
                "bronze_artifacts": artifacts,
                "token_fingerprint_sha256": self.token_fingerprint,
                "changes": changes,
                "silver_version_id": version["version_id"],
                "quality_gate": quality,
            },
        )
