from __future__ import annotations

import gzip
import hashlib
import json
import os
import fcntl
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from .source import TushareResponse


DUCKDB_MEMORY_LIMIT = "4GB"


def open_duckdb(
    database: str | Path = ":memory:",
    *,
    read_only: bool = False,
    temp_directory: Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with bounded memory and optional disk spill."""
    connection = duckdb.connect(str(database), read_only=read_only)
    # Timestamp-to-date and Arrow timezone conversion must not depend on the
    # host's locale. Financial first-seen policies are normalized in UTC.
    connection.execute("SET TimeZone='UTC'")
    connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    connection.execute("SET threads=4")
    if temp_directory is not None:
        temp_directory.mkdir(parents=True, exist_ok=True)
        escaped = str(temp_directory.resolve()).replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped}'")
    return connection


def utc_now() -> datetime:
    return datetime.now(UTC)


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(project_root: Path | None = None) -> str:
    """Hash production calculation code when a Git commit is unavailable.

    Frontend layout and test-only edits must not invalidate multi-million-row
    research artifacts. Their own build/test results are tracked separately.
    """
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    digest = hashlib.sha256()
    allowed_suffixes = {".py", ".ts", ".tsx", ".css", ".json", ".toml"}
    files: list[Path] = []
    for directory in (root / "src-python", root / "scripts"):
        if directory.exists():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix in allowed_suffixes
                and "__pycache__" not in path.parts
            )
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class DataLake:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.bronze = self.root / "bronze" / "tushare"
        self.silver = self.root / "silver"
        self.metadata = self.root / "metadata"
        self.manifests = self.metadata / "manifests"
        for directory in (self.bronze, self.silver, self.manifests):
            directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def pipeline_lock(self):
        """Prevent concurrent writers from touching the same local data lake."""
        path = self.metadata / "pipeline.lock"
        with path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"Another factor-matrix writer is active for {self.root}"
                ) from exc
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def immutable_base_guard(self):
        """Fail a calculation if Bronze or Silver changes while it runs.

        Ingestion commands intentionally do not use this guard. Any future Gold
        calculation must use it so data ingestion and derived artifacts cannot
        silently bleed into one another.
        """
        def fingerprint() -> dict[str, tuple[int, int]]:
            result: dict[str, tuple[int, int]] = {}
            for root in (self.bronze, self.silver):
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.is_file():
                        stat = path.stat()
                        result[str(path.relative_to(self.root))] = (
                            stat.st_size,
                            stat.st_mtime_ns,
                        )
            return result

        before = fingerprint()
        try:
            yield
        finally:
            after = fingerprint()
            if before != after:
                changed = sorted(set(before) ^ set(after))
                changed.extend(
                    path for path in set(before) & set(after) if before[path] != after[path]
                )
                raise RuntimeError(
                    f"BASE_DATA_MUTATED_DURING_CALCULATION {sorted(set(changed))[:10]}"
                )

    def save_bronze(self, response: TushareResponse, ingested_at: datetime) -> dict[str, str]:
        safe_raw = {
            "schema_version": 1,
            "api_name": response.api_name,
            "params": response.params,
            "ingested_at": ingested_at.isoformat(),
            "response": response.raw,
        }
        raw_bytes = json.dumps(
            safe_raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        query_hash = hashlib.sha256(
            json.dumps(response.params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        directory = self.bronze / response.api_name / f"ingest_date={ingested_at.date().isoformat()}"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = ingested_at.strftime("%H%M%S%f")
        path = directory / f"{stamp}_{query_hash}_{content_hash[:12]}.json.gz"
        with gzip.open(path, "wb") as file:
            file.write(raw_bytes)
        return {
            "path": str(path.relative_to(self.root)),
            "sha256": content_hash,
            "api_name": response.api_name,
            "params_hash": query_hash,
            "ingested_at": ingested_at.isoformat(),
        }

    def load_bronze(self, path: Path) -> TushareResponse:
        """Load a Bronze response for deterministic Silver replay.

        Records written before the replay envelope are still readable, but have
        no recoverable request parameters and are therefore marked with an empty
        parameter mapping.
        """
        resolved = path if path.is_absolute() else self.root / path
        with gzip.open(resolved, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if "response" in payload:
            raw = payload["response"]
            api_name = str(payload["api_name"])
            params = dict(payload["params"])
        else:
            raw = payload
            api_name = resolved.parent.parent.name
            params = {}
        data = raw.get("data") or {}
        return TushareResponse(
            api_name=api_name,
            params=params,
            fields=data.get("fields") or [],
            items=data.get("items") or [],
            raw=raw,
        )

    def bronze_objects(self, api_name: str) -> tuple[Path, ...]:
        return tuple(sorted((self.bronze / api_name).glob("ingest_date=*/*.json.gz")))

    def _assert_base_table_mutable(self, table: str) -> None:
        """Prevent legacy writers from mutating a content-addressed frozen base."""
        manifest = self.metadata / "base_manifests" / "legacy_base_v1.json"
        if not manifest.exists():
            return
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if table in payload.get("artifacts", {}):
            raise RuntimeError(
                f"FROZEN_BASE_MUTATION_FORBIDDEN table={table}; "
                "write an append-only Silver delta/correction partition"
            )

    def upsert(
        self,
        table: str,
        frame: pl.DataFrame,
        *,
        primary_key: list[str],
    ) -> Path:
        self._assert_base_table_mutable(table)
        if frame.is_empty():
            raise ValueError(f"Refusing to write empty Silver table: {table}")
        directory = self.silver / table
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "data.parquet"
        if destination.exists():
            pending = pl.concat(
                [pl.scan_parquet(destination), frame.lazy()], how="diagonal_relaxed"
            )
        else:
            pending = frame.lazy()
        pending = pending.unique(
            subset=primary_key, keep="last", maintain_order=True
        )
        temporary = destination.with_suffix(".parquet.tmp")
        pending.sink_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, destination)
        return destination

    def replace(self, table: str, frame: pl.DataFrame) -> Path:
        self._assert_base_table_mutable(table)
        if frame.is_empty():
            raise ValueError(f"Refusing to replace with empty Silver table: {table}")
        directory = self.silver / table
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "data.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        os.replace(temporary, destination)
        return destination

    def write_manifest(self, run_id: str, payload: dict[str, Any]) -> Path:
        path = self.manifests / f"{run_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def write_immutable_json(self, path: Path, payload: dict[str, Any]) -> Path:
        """Create an immutable registry record; never silently replace it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
        )
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != encoded:
                raise RuntimeError(f"IMMUTABLE_RECORD_CONFLICT {path}")
            return path
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def artifact_record(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        try:
            stored_path = str(resolved.relative_to(self.root))
        except ValueError:
            stored_path = str(resolved)
        return {
            "path": stored_path,
            "sha256": file_sha256(resolved),
            "bytes": resolved.stat().st_size,
        }

    def register_data_snapshot(self, as_of: date, tables: list[str]) -> dict[str, Any]:
        artifacts: dict[str, dict[str, Any]] = {}
        for table in sorted(set(tables)):
            path = self.silver / table / "data.parquet"
            artifacts[table] = self.artifact_record(path)
        snapshot_hash = json_hash(
            {"as_of": as_of.isoformat(), "artifacts": artifacts}
        )
        snapshot_id = f"data_snapshot_{as_of:%Y%m%d}_{snapshot_hash[:12]}"
        payload = {
            "schema_version": 1,
            "run_id": snapshot_id,
            "job": "data_snapshot",
            "status": "registered",
            "as_of_date": as_of.isoformat(),
            "snapshot_hash": snapshot_hash,
            "artifacts": artifacts,
        }
        self.write_immutable_json(self.manifests / f"{snapshot_id}.json", payload)
        return payload

    def calculation_run_id(
        self,
        job: str,
        as_of: date,
        config: dict[str, Any],
        parent_run_ids: list[str],
        *,
        code_hash: str | None = None,
    ) -> tuple[str, str, str]:
        config_hash = json_hash(config)
        resolved_code_hash = code_hash or source_tree_hash()
        identity_hash = json_hash(
            {
                "job": job,
                "as_of_date": as_of.isoformat(),
                "config_hash": config_hash,
                "code_hash": resolved_code_hash,
                "parent_run_ids": sorted(parent_run_ids),
            }
        )
        return f"{job}_{as_of:%Y%m%d}_{identity_hash[:12]}", config_hash, resolved_code_hash

    def register_calculation(
        self,
        *,
        run_id: str,
        job: str,
        as_of: date,
        mode: str,
        config: dict[str, Any],
        config_hash: str,
        code_hash: str,
        parent_run_ids: list[str],
        inputs: dict[str, Path],
        outputs: dict[str, Path],
        quality_gate: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        input_artifacts = {
            name: self.artifact_record(path) for name, path in sorted(inputs.items())
        }
        output_artifacts = {
            name: self.artifact_record(path) for name, path in sorted(outputs.items())
        }
        result_hash = json_hash(output_artifacts)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "job": job,
            "mode": mode,
            "status": quality_gate.get("status", "unknown"),
            "as_of_date": as_of.isoformat(),
            "config": config,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "parent_run_ids": sorted(parent_run_ids),
            "inputs": input_artifacts,
            "outputs": output_artifacts,
            "result_hash": result_hash,
            "quality_gate": quality_gate,
        }
        if extra:
            payload.update(extra)
        return self.write_immutable_json(self.manifests / f"{run_id}.json", payload)

    def refresh_catalog(self) -> Path:
        from .revisioned_silver import PRIMARY_KEYS, SilverAsOfReader

        catalog_path = self.metadata / "catalog.duckdb"
        connection = open_duckdb(
            catalog_path, temp_directory=self.root / "tmp" / "duckdb"
        )
        try:
            versioned = (
                self.metadata / "base_manifests" / "legacy_base_v1.json"
            ).exists()
            for parquet in sorted(self.silver.glob("*/data.parquet")):
                table = parquet.parent.name
                if versioned and table in PRIMARY_KEYS:
                    continue
                escaped_path = str(parquet).replace("'", "''")
                connection.execute(
                    f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM '
                    f"read_parquet('{escaped_path}', hive_partitioning=false)"
                )
            if versioned:
                reader = SilverAsOfReader(self)
                for table in sorted(PRIMARY_KEYS):
                    try:
                        relation = reader.relation_sql(table, utc_now())
                    except FileNotFoundError:
                        continue
                    connection.execute(f'CREATE OR REPLACE VIEW "{table}" AS {relation}')
        finally:
            connection.close()
        return catalog_path
