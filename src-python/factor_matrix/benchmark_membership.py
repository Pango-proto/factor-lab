"""Append-only point-in-time benchmark constituent history."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from .normalize import response_frame
from .revisioned_silver import publish_revisioned_batch
from .source import TushareClient
from .storage import DataLake, json_hash, utc_now


BENCHMARK_INDEXES = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI800": "000906.SH",
}


class BenchmarkMembershipPipeline:
    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint

    def sync(self, start: date, end: date) -> Path:
        if end < start:
            raise ValueError("benchmark end must not be before start")
        config = {
            "start": start.isoformat(), "end": end.isoformat(),
            "benchmarks": BENCHMARK_INDEXES,
            "storage_contract": "revisioned_silver_v1",
        }
        run_id = f"benchmark_history_{end:%Y%m%d}_{json_hash(config)[:12]}"
        manifest_path = self.lake.manifests / f"{run_id}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("quality_gate", {}).get("status") == "passed" and existing.get(
                "silver_version_id"
            ):
                return manifest_path
        observed_at = utc_now()
        frames: list[pl.DataFrame] = []
        bronze: list[dict[str, str]] = []
        cursor = date(start.year, start.month, 1)
        while cursor <= end:
            next_month = date(
                cursor.year + (cursor.month == 12),
                1 if cursor.month == 12 else cursor.month + 1, 1,
            )
            month_start, month_end = max(start, cursor), min(end, next_month - timedelta(days=1))
            for benchmark_id, index_code in BENCHMARK_INDEXES.items():
                response = self.client.query(
                    "index_weight",
                    {
                        "index_code": index_code,
                        "start_date": month_start.strftime("%Y%m%d"),
                        "end_date": month_end.strftime("%Y%m%d"),
                    },
                    ["index_code", "con_code", "trade_date", "weight"],
                )
                bronze.append(self.lake.save_bronze(response, observed_at))
                frame = response_frame(response)
                if not frame.is_empty():
                    frames.append(frame.with_columns(pl.lit(benchmark_id).alias("benchmark_id")))
            cursor = next_month
        if not frames:
            raise RuntimeError("BENCHMARK_HISTORY_EMPTY")
        raw = pl.concat(frames, how="diagonal_relaxed").with_columns(
            pl.col("con_code").alias("asset_id"),
            pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
            pl.col("weight").cast(pl.Float64) / 100.0,
        ).unique(["benchmark_id", "asset_id", "trade_date"], keep="last")
        snapshot_dates = raw.select("benchmark_id", "trade_date").unique().sort(
            ["benchmark_id", "trade_date"]
        ).with_columns(
            pl.col("trade_date").shift(-1).over("benchmark_id").alias("effective_to")
        )
        history = raw.join(
            snapshot_dates, on=["benchmark_id", "trade_date"], how="left"
        ).with_columns(
            pl.col("trade_date").alias("effective_from"),
            pl.col("trade_date").alias("announced_at"),
            pl.lit("effective snapshot date; no earlier announcement assumed").alias(
                "announcement_policy"
            ),
            pl.lit(json_hash(bronze)).alias("source_snapshot_hash"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(observed_at).alias("ingested_at"),
        ).select(
            "benchmark_id", "index_code", "asset_id", "effective_from",
            "effective_to", "weight", "announced_at", "announcement_policy",
            "source_snapshot_hash", "source_id", "ingested_at",
        )
        quality = {
            "status": "passed", "rows": history.height,
            "duplicate_rows": history.height - history.unique(
                ["benchmark_id", "asset_id", "effective_from"]
            ).height,
            "benchmarks": history.get_column("benchmark_id").n_unique(),
        }
        if quality["duplicate_rows"] or quality["benchmarks"] != len(BENCHMARK_INDEXES):
            quality["status"] = "failed"
            raise RuntimeError(f"BENCHMARK_MEMBERSHIP_QUALITY_FAILED {quality}")
        changes, version = publish_revisioned_batch(
            self.lake, {"benchmark_membership_history": history},
            effective_date=end, observed_at=observed_at, quality_gate=quality,
        )
        self.lake.refresh_catalog()
        return self.lake.write_immutable_json(manifest_path, {
            "schema_version": 1, "run_id": run_id, "job": "benchmark_history_sync",
            "status": "passed", "created_at": observed_at.isoformat(), "config": config,
            "config_hash": json_hash(config),
            "source_credential_fingerprint": self.token_fingerprint,
            "bronze_objects": bronze, "quality_gate": quality, "changes": changes,
            "silver_version_id": version["version_id"],
        })
