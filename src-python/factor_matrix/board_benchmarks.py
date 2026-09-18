"""Official index facts and derived full-sample listing-board benchmarks."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from .boards import BOARD_IDS, BOARD_LABELS
from .normalize import response_frame
from .revisioned_silver import SilverAsOfReader, SilverVersionLedger, publish_revisioned_batch
from .source import TushareClient
from .storage import DataLake, json_hash, open_duckdb, utc_now


BOARD_REFERENCE_INDEXES: dict[str, tuple[str, ...]] = {
    "MAIN": ("000001.SH", "399001.SZ"),
    "CHINEXT": ("399006.SZ",),
    "STAR": ("000688.SH",),
    "BSE": ("899050.BJ",),
}
INDEX_NAMES = {
    "000001.SH": "上证综指",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "899050.BJ": "北证50",
}
INDEX_DAILY_FIELDS = [
    "ts_code", "trade_date", "close", "open", "high", "low", "pre_close",
    "change", "pct_chg", "vol", "amount",
]


class IndexDailyPipeline:
    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint

    def sync(self, start: date, end: date) -> Path:
        if end < start:
            raise ValueError("index daily end must not be before start")
        config = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "indexes": sorted(INDEX_NAMES),
            "storage_contract": "revisioned_silver_v1",
        }
        run_id = f"index_daily_{end:%Y%m%d}_{json_hash(config)[:12]}"
        manifest_path = self.lake.manifests / f"{run_id}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "passed" and existing.get("silver_version_id"):
                return manifest_path
        ingested_at = utc_now()
        frames: list[pl.DataFrame] = []
        bronze: list[dict[str, str]] = []
        for index_code in sorted(INDEX_NAMES):
            response = self.client.query(
                "index_daily",
                {
                    "ts_code": index_code,
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
                INDEX_DAILY_FIELDS,
            )
            bronze.append(self.lake.save_bronze(response, ingested_at))
            frame = response_frame(response)
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            raise RuntimeError("BOARD_INDEX_DAILY_EMPTY")
        daily = (
            pl.concat(frames, how="diagonal_relaxed")
            .with_columns(
                pl.col("ts_code").alias("index_code"),
                pl.col("trade_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d"),
                (pl.col("pct_chg").cast(pl.Float64) / 100.0).alias("index_return"),
                pl.lit("tushare").alias("source_id"),
                pl.lit(ingested_at).alias("ingested_at"),
            )
            .drop("ts_code")
            .unique(["trade_date", "index_code"], keep="last")
        )
        target_rows = daily.filter(pl.col("trade_date") == end).get_column("index_code").n_unique()
        quality = {
            "status": "passed" if target_rows == len(INDEX_NAMES) else "failed",
            "target_date": end.isoformat(),
            "target_indexes": target_rows,
            "expected_indexes": len(INDEX_NAMES),
            "duplicate_rows": daily.height - daily.unique(["trade_date", "index_code"]).height,
        }
        if quality["status"] != "passed":
            raise RuntimeError(f"BOARD_INDEX_DAILY_QUALITY_FAILED {quality}")
        changes, version = publish_revisioned_batch(
            self.lake, {"index_prices_daily": daily}, effective_date=end,
            observed_at=ingested_at, quality_gate=quality,
        )
        self.lake.refresh_catalog()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "job": "index_daily_sync",
            "status": quality["status"],
            "created_at": ingested_at.isoformat(),
            "config": config,
            "config_hash": json_hash(config),
            "source_credential_fingerprint": self.token_fingerprint,
            "bronze_objects": bronze,
            "changes": changes,
            "silver_version_id": version["version_id"],
            "quality_gate": quality,
        }
        self.lake.write_immutable_json(manifest_path, manifest)
        return manifest_path


def _escaped(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_board_benchmarks(lake: DataLake, start: date, end: date) -> dict[str, Any]:
    """Build full-board total-return indexes with prior observable float-cap weights."""
    if end < start:
        raise ValueError("board benchmark end must not be before start")
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("BOARD_BENCHMARK_REQUIRES_SILVER_VERSION")
    config = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "boards": list(BOARD_IDS),
        "weight": "latest prior float_mkt_cap",
        "universe": "all listed board securities with an actionable return",
        "return": "returns_daily.total_return",
        "board_source": "security_daily_state_v1",
        "silver_version_id": version["version_id"],
    }
    run_id, config_hash, code_hash = lake.calculation_run_id(
        "board_benchmark", end, config, [version["version_id"]]
    )
    output_dir = lake.root / "gold" / "board_benchmarks" / f"run_id={run_id}"
    output_path = output_dir / "board_benchmark_daily.parquet"
    summary_path = output_dir / "summary.json"
    manifest_path = lake.manifests / f"{run_id}.json"
    if output_path.exists() and summary_path.exists() and manifest_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    reader = SilverAsOfReader(lake)
    knowledge_time = utc_now()
    returns = reader.relation_sql("returns_daily", knowledge_time)
    states = reader.relation_sql("security_daily_state", knowledge_time)
    valuations = reader.relation_sql("valuation_daily", knowledge_time)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        connection.execute(
            f"""
            COPY (
                WITH observations AS (
                    SELECT r.trade_date, r.asset_id, r.total_return, b.board_id,
                           v.float_mkt_cap,
                           last_value(v.float_mkt_cap IGNORE NULLS) OVER (
                               PARTITION BY r.asset_id ORDER BY r.trade_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                           ) AS prior_float_mkt_cap
                    FROM ({returns}) r
                    JOIN ({states}) b
                      ON b.asset_id = r.asset_id AND b.trade_date = r.trade_date
                     AND b.in_a_share_scope
                    LEFT JOIN ({valuations}) v
                      ON v.trade_date = r.trade_date AND v.asset_id = r.asset_id
                    WHERE r.trade_date <= DATE '{end.isoformat()}'
                ), daily AS (
                    SELECT trade_date, board_id,
                           sum(prior_float_mkt_cap * total_return)
                             / nullif(sum(prior_float_mkt_cap), 0) AS board_return,
                           count(*) FILTER (
                             WHERE total_return IS NOT NULL AND prior_float_mkt_cap > 0
                           ) AS constituent_count,
                           sum(prior_float_mkt_cap) FILTER (
                             WHERE total_return IS NOT NULL AND prior_float_mkt_cap > 0
                           ) AS effective_weight_base,
                           count(*) FILTER (
                             WHERE total_return IS NULL OR prior_float_mkt_cap IS NULL
                                OR prior_float_mkt_cap <= 0
                           ) AS excluded_observations
                    FROM observations
                    WHERE trade_date BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'
                    GROUP BY 1,2
                )
                SELECT *,
                       1000.0 * exp(sum(ln(1.0 + board_return)) OVER (
                           PARTITION BY board_id ORDER BY trade_date
                       )) AS total_return_index,
                       'CUSTOM_' || board_id || '_ALL' AS benchmark_id,
                       'prior_float_mkt_cap_total_return_v1' AS methodology_version
                FROM daily
                ORDER BY trade_date, board_id
            ) TO '{_escaped(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()
    temporary.replace(output_path)
    frame = pl.read_parquet(output_path)
    latest = frame.filter(pl.col("trade_date") == end)
    present = set(latest.get_column("board_id").to_list())
    missing = sorted(set(BOARD_IDS) - present)
    invalid = latest.filter(
        pl.col("board_return").is_null()
        | ~pl.col("board_return").is_finite()
        | (pl.col("constituent_count") <= 0)
    ).height
    quality = {
        "status": "passed" if not missing and invalid == 0 else "failed",
        "target_date": end.isoformat(),
        "missing_boards": missing,
        "invalid_latest_rows": invalid,
        "duplicate_rows": frame.height - frame.unique(["trade_date", "board_id"]).height,
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "board_benchmark",
        "status": quality["status"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "boards": [
            {
                "board_id": board,
                "label": BOARD_LABELS[board],
                "benchmark_id": f"CUSTOM_{board}_ALL",
                "official_references": list(BOARD_REFERENCE_INDEXES[board]),
                **(
                    latest.filter(pl.col("board_id") == board)
                    .select("board_return", "total_return_index", "constituent_count")
                    .to_dicts()[0]
                    if board in present else {}
                ),
            }
            for board in BOARD_IDS
        ],
        "quality_gate": quality,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lake.register_calculation(
        run_id=run_id,
        job="board_benchmark",
        as_of=end,
        mode="registered_run",
        config=config,
        config_hash=config_hash,
        code_hash=code_hash,
        parent_run_ids=[version["version_id"]],
        inputs={
            "silver_version_manifest": lake.metadata / "silver_versions"
            / f"{version['version_id']}.json"
        },
        outputs={"board_benchmark_daily": output_path, "summary": summary_path},
        quality_gate=quality,
    )
    if quality["status"] != "passed":
        raise RuntimeError(f"BOARD_BENCHMARK_QUALITY_FAILED {quality}")
    return summary
