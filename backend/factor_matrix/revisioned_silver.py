"""Append-only Silver partitions, version ledger, and reproducible as-of reads."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .storage import DataLake, json_hash


PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "trade_calendar": ("exchange", "cal_date"),
    "security_master": ("asset_id",),
    "prices_daily": ("trade_date", "asset_id"),
    "valuation_daily": ("trade_date", "asset_id"),
    "price_limits_daily": ("trade_date", "asset_id"),
    "suspensions_daily": ("trade_date", "asset_id"),
    "stock_st_daily": ("trade_date", "asset_id"),
    "security_daily_state": ("trade_date", "asset_id"),
    "returns_daily": ("trade_date", "asset_id"),
    "concept_membership_snapshot": ("snapshot_date", "concept_id", "asset_id"),
    "concept_membership_changes": ("observed_date", "concept_id", "asset_id", "change_type"),
    "index_prices_daily": ("trade_date", "index_code"),
    "risk_free_daily": ("trade_date", "rate_id"),
    "industry_classification": ("classification_standard", "industry_index_code"),
    "industry_membership_history": (
        "classification_standard", "asset_id", "l1_code", "l2_code", "l3_code", "in_date",
    ),
    "financial_pit": ("revision_id",),
    "income_pit": ("revision_id",),
    "cashflow_pit": ("revision_id",),
    "board_membership_history": ("asset_id", "effective_from"),
    "benchmark_membership_history": ("benchmark_id", "asset_id", "effective_from"),
}


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


class RevisionedSilverStore:
    def __init__(self, lake: DataLake) -> None:
        self.lake = lake

    def append(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        effective_date: date,
        first_seen_at: datetime,
        correction: bool = False,
    ) -> dict[str, Any]:
        if table not in PRIMARY_KEYS:
            raise ValueError(f"REVISIONED_TABLE_NOT_REGISTERED {table}")
        if frame.is_empty():
            raise ValueError(f"REVISIONED_EMPTY_PARTITION {table}")
        forbidden = {"qfq_close", "qfq_anchor_date"} & set(frame.columns)
        if table == "prices_daily" and forbidden:
            raise ValueError(f"DEPRECATED_PRICE_COLUMNS {sorted(forbidden)}")
        missing = set(PRIMARY_KEYS[table]) - set(frame.columns)
        if missing:
            raise ValueError(f"REVISIONED_PRIMARY_KEY_MISSING {table} {sorted(missing)}")
        revision = first_seen_at.astimezone(UTC)
        normalized = frame.with_columns(pl.lit(revision).alias("revision_at"))
        if "first_seen_at" not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(revision).alias("first_seen_at"))
        keys = list(PRIMARY_KEYS[table])
        if normalized.height != normalized.unique(keys).height:
            raise ValueError(f"REVISIONED_DUPLICATE_KEYS {table}")
        layer = "corrections" if correction else "delta"
        partition = (
            self.lake.silver / layer / table / f"dt={effective_date:%Y-%m}"
            / f"effective_date={effective_date.isoformat()}"
            / f"revision_at={_stamp(revision)}"
        )
        partition.mkdir(parents=True, exist_ok=False)
        output = partition / "data.parquet"
        normalized.write_parquet(output, compression="zstd", statistics=True)
        return {
            "table": table,
            "layer": layer,
            "effective_date": effective_date.isoformat(),
            "revision_at": revision.isoformat(),
            **self.lake.artifact_record(output),
        }


def publish_revisioned_batch(
    lake: DataLake,
    frames: dict[str, pl.DataFrame],
    *,
    effective_date: date,
    observed_at: datetime,
    quality_gate: dict[str, Any],
    correction: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append a validated batch and atomically make all partitions visible."""
    ledger = SilverVersionLedger(lake)
    current = ledger.current()
    if current is None:
        raise RuntimeError("SILVER_VERSION_LEDGER_REQUIRED")
    store = RevisionedSilverStore(lake)
    changes = [
        store.append(
            table, frame, effective_date=effective_date,
            first_seen_at=observed_at, correction=correction,
        )
        for table, frame in frames.items()
        if not frame.is_empty()
    ]
    if not changes:
        return [], current
    version = ledger.publish(
        base_id=current["base_id"],
        base_snapshot_sha256=current["base_snapshot_sha256"],
        observed_at=observed_at,
        changes=[*current["changes"], *changes],
        quality_gate=quality_gate,
    )
    return changes, version


class SilverVersionLedger:
    def __init__(self, lake: DataLake) -> None:
        self.lake = lake
        self.root = lake.metadata / "silver_versions"
        self.pointer = self.root / "_CURRENT"

    def current(self) -> dict[str, Any] | None:
        if not self.pointer.exists():
            return None
        name = self.pointer.read_text(encoding="utf-8").strip()
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def publish(
        self,
        *,
        base_id: str,
        base_snapshot_sha256: str,
        observed_at: datetime,
        changes: Iterable[dict[str, Any]],
        quality_gate: dict[str, Any],
    ) -> dict[str, Any]:
        parent = self.current()
        ordered_changes = sorted(
            list(changes), key=lambda row: (row.get("table", ""), row.get("path", ""))
        )
        identity = {
            "base_id": base_id,
            "base_snapshot_sha256": base_snapshot_sha256,
            "parent_version_id": parent["version_id"] if parent else None,
            "changes": ordered_changes,
            "quality_gate": quality_gate,
        }
        digest = json_hash(identity)
        payload = {
            "schema_version": 1,
            "version_id": f"silver_version_{digest[:16]}",
            "observed_at": observed_at.astimezone(UTC).isoformat(),
            "revision_at_semantics": "first_seen_at",
            "as_of_filter_required": True,
            **identity,
            "version_sha256": digest,
        }
        path = self.root / f"{payload['version_id']}.json"
        self.lake.write_immutable_json(path, payload)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.pointer.with_suffix(".tmp")
        temporary.write_text(path.name + "\n", encoding="utf-8")
        os.replace(temporary, self.pointer)
        return payload


class SilverAsOfReader:
    """Build DuckDB relations; callers must supply a knowledge timestamp."""

    def __init__(self, lake: DataLake, base_id: str = "legacy_base_v1", *,
                 version_id: str | None = None) -> None:
        self.lake = lake
        self.pinned_version = None
        if version_id is not None:
            if not version_id.startswith('silver_version_') or not all(
                c in '0123456789abcdef' for c in version_id.removeprefix('silver_version_')
            ):
                raise ValueError('SILVER_PINNED_VERSION_ID_INVALID')
            version_path = lake.metadata/'silver_versions'/f'{version_id}.json'
            pinned = json.loads(version_path.read_text(encoding='utf-8'))
            identity = {key: pinned[key] for key in (
                'base_id', 'base_snapshot_sha256', 'parent_version_id', 'changes', 'quality_gate')}
            digest = json_hash(identity)
            if pinned['version_id'] != version_id or pinned['version_sha256'] != digest or version_id != f'silver_version_{digest[:16]}':
                raise ValueError('SILVER_PINNED_VERSION_HASH_MISMATCH')
            self.pinned_version = pinned
            base_id = pinned['base_id']
        manifest = lake.metadata / "base_manifests" / f"{base_id}.json"
        self.base = json.loads(manifest.read_text(encoding="utf-8"))
        self.base_revision_at = self.base["frozen_at"]

    def _base_path(self, table: str) -> Path:
        if table in self.base.get("artifacts", {}):
            return self.lake.root / self.base["artifacts"][table]["path"]
        if table == "security_daily_state":
            return (
                self.lake.silver / "base"
                / f"snapshot_id={self.base['snapshot_id']}"
                / table / "data.parquet"
            )
        raise KeyError(table)

    def relation_sql(self, table: str, as_of_timestamp: datetime) -> str:
        if table not in PRIMARY_KEYS:
            raise ValueError(f"AS_OF_TABLE_NOT_REGISTERED {table}")
        timestamp = as_of_timestamp.astimezone(UTC).replace(tzinfo=None).isoformat()
        if timestamp < datetime.fromisoformat(self.base_revision_at).replace(tzinfo=None).isoformat():
            raise ValueError("AS_OF_PRECEDES_BASE_FIRST_SEEN")
        sources: list[str] = []
        try:
            base_source = self._base_path(table)
        except KeyError:
            base_source = None
        if base_source is not None and base_source.exists():
            base_path = str(base_source).replace("'", "''")
            deprecated = (
                {"qfq_close", "qfq_anchor_date", "total_return"}
                if table == "prices_daily" else set()
            )
            base_columns = [
                name for name in pl.scan_parquet(base_source).collect_schema().names()
                if name not in {"revision_at", *deprecated}
            ]
            base_projection = ",".join(f'"{name}"' for name in base_columns)
            # The base's publication timestamp is NOT the first observation of
            # a financial revision. Preserve source provenance when present;
            # replacing Aug-06/10 with the Aug-15 freeze fabricated PIT failures.
            first_seen_projection = (
                "" if "first_seen_at" in base_columns
                else f", TIMESTAMPTZ '{self.base_revision_at}' AS first_seen_at"
            )
            sources.append(
                f"SELECT {base_projection}, TIMESTAMPTZ '{self.base_revision_at}' AS revision_at"
                f"{first_seen_projection} "
                f"FROM read_parquet('{base_path}', hive_partitioning=false)"
            )
        current = self.pinned_version if self.pinned_version is not None else SilverVersionLedger(self.lake).current()
        published_paths = None
        if current is not None:
            published_paths = {
                str(item["path"]): item for item in current.get("changes", [])
                if item.get("table") == table and item.get("layer") in {"delta", "corrections"}
            }
            if self.pinned_version is not None:
                missing = [path for path in published_paths if not (self.lake.root/path).is_file()]
                if missing:
                    raise FileNotFoundError('SILVER_PINNED_PARTITION_MISSING:'+missing[0])
        for layer in ("delta", "corrections"):
            if published_paths is None:
                files = sorted((self.lake.silver / layer / table).glob("**/data.parquet"))
            else:
                files = sorted(
                    self.lake.root / path for path, item in published_paths.items()
                    if item.get("layer") == layer and (self.lake.root / path).exists()
                )
            if files:
                paths = ",".join(
                    "'" + str(path).replace("'", "''") + "'" for path in files
                )
                sources.append(
                    f"SELECT * FROM read_parquet([ {paths} ], union_by_name=true, "
                    "hive_partitioning=false)"
                )
        if not sources:
            raise FileNotFoundError(f"AS_OF_TABLE_HAS_NO_MATERIALIZATION {table}")
        keys = ",".join(PRIMARY_KEYS[table])
        union = " UNION ALL BY NAME ".join(sources)
        selected = "*"
        return f"""
            SELECT {selected} FROM ({union})
            WHERE revision_at <= TIMESTAMPTZ '{timestamp}+00:00'
            QUALIFY row_number() OVER (
              PARTITION BY {keys} ORDER BY revision_at DESC
            )=1
        """


def read_as_of_frame(
    lake: DataLake, table: str, as_of_timestamp: datetime | None = None
) -> pl.DataFrame:
    """Materialize one canonical Silver table without bypassing its version ledger."""
    from .storage import open_duckdb, utc_now

    if not (lake.metadata / "base_manifests" / "legacy_base_v1.json").exists():
        frames: list[pl.DataFrame] = []
        direct = lake.silver / table / "data.parquet"
        if direct.exists():
            frame = pl.read_parquet(direct)
            if "revision_at" not in frame.columns:
                frame = frame.with_columns(
                    pl.lit(datetime(1970, 1, 1, tzinfo=UTC)).alias("revision_at")
                )
            frames.append(frame)
        current = SilverVersionLedger(lake).current()
        if current is not None:
            for item in current.get("changes", []):
                if item.get("table") == table and (lake.root / item["path"]).exists():
                    frames.append(pl.read_parquet(lake.root / item["path"]))
        if not frames:
            raise FileNotFoundError(f"AS_OF_TABLE_HAS_NO_MATERIALIZATION {table}")
        combined = pl.concat(frames, how="diagonal_relaxed")
        keys = list(PRIMARY_KEYS[table])
        return combined.sort("revision_at").unique(keys, keep="last")
    relation = SilverAsOfReader(lake).relation_sql(table, as_of_timestamp or utc_now())
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        sql = f"SELECT * FROM ({relation})"
        try:
            return connection.execute(sql).pl()
        except ModuleNotFoundError as exc:
            if exc.name != "pyarrow":
                raise
            export_root = lake.root / "tmp" / "as_of_exports"
            export_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=export_root, suffix=".parquet", delete=False
            ) as temporary:
                export_path = Path(temporary.name)
            escaped = str(export_path).replace("'", "''")
            try:
                connection.execute(f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET)")
                return pl.read_parquet(export_path)
            finally:
                export_path.unlink(missing_ok=True)
    finally:
        connection.close()


def latest_silver_date(lake: DataLake, table: str, column: str) -> date:
    """Read a date bound from the canonical versioned relation."""
    from .storage import open_duckdb, utc_now

    relation = SilverAsOfReader(lake).relation_sql(table, utc_now())
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        value = connection.execute(f'SELECT max("{column}") FROM ({relation})').fetchone()[0]
    finally:
        connection.close()
    if value is None:
        raise RuntimeError(f"SILVER_TABLE_EMPTY {table}")
    return value
