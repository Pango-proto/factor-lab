"""Atomic daily THS concept-membership snapshots.

The snapshot is published only when the concept master and every constituent
response pass completeness checks. A failed run leaves Bronze evidence and a
SOURCE_INCOMPLETE manifest, but never fabricates or carries forward Silver data.
"""

from __future__ import annotations

import threading
import time
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

import polars as pl

from .normalize import parse_yyyymmdd, response_frame
from .revisioned_silver import RevisionedSilverStore, SilverVersionLedger
from .source import TushareClient, TushareResponse
from .storage import DataLake, json_hash


MASTER_FIELDS = ["ts_code", "name", "count", "exchange", "list_date", "type"]
MEMBER_FIELDS = [
    "ts_code", "con_code", "con_name", "weight", "in_date", "out_date", "is_new",
]


def concept_snapshot_passed(lake: DataLake, snapshot_date: date) -> bool:
    pattern = f"concept_snapshot_{snapshot_date:%Y%m%d}_*.json"
    for path in sorted(lake.manifests.glob(pattern), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        output = payload.get("output") or {}
        if (
            payload.get("status") == "passed"
            and payload.get("source_id") == "tushare.ths_index+ths_member"
            and output.get("path")
            and (lake.root / output["path"]).exists()
        ):
            return True
    return False


def concept_scan_passed(lake: DataLake, snapshot_date: date) -> bool:
    pattern = f"concept_change_scan_{snapshot_date:%Y%m%d}_*.json"
    for path in lake.manifests.glob(pattern):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "passed":
                return True
        except (OSError, ValueError):
            continue
    return False


def concept_observation_manifest(lake: DataLake, snapshot_date: date) -> Path | None:
    """Return one successful daily scan or full reconciliation manifest."""
    patterns = (
        f"concept_change_scan_{snapshot_date:%Y%m%d}_*.json",
        f"concept_snapshot_{snapshot_date:%Y%m%d}_*.json",
    )
    for pattern in patterns:
        for path in sorted(lake.manifests.glob(pattern), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if payload.get("status") == "passed":
                return path
    return None


def _latest_full_snapshot(lake: DataLake, before_or_on: date) -> tuple[dict, pl.DataFrame] | None:
    candidates = []
    for path in lake.manifests.glob("concept_snapshot_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            snapshot_date = date.fromisoformat(payload["snapshot_date"])
        except (OSError, ValueError, KeyError):
            continue
        output = payload.get("output", {}).get("path")
        if payload.get("status") == "passed" and snapshot_date <= before_or_on and output:
            candidates.append((snapshot_date, payload, lake.root / output))
    if not candidates:
        return None
    _snapshot_date, payload, output = max(candidates, key=lambda item: item[0])
    return payload, pl.read_parquet(output)


def _latest_scan_master(lake: DataLake, before: date) -> pl.DataFrame | None:
    """Return the master observed by the most recent successful earlier scan."""
    candidates: list[tuple[date, dict]] = []
    for path in lake.manifests.glob("concept_change_scan_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scan_date = date.fromisoformat(payload["snapshot_date"])
        except (OSError, ValueError, KeyError):
            continue
        if payload.get("status") == "passed" and scan_date < before:
            candidates.append((scan_date, payload))
    if not candidates:
        return None
    _scan_date, payload = max(candidates, key=lambda item: item[0])
    master_object = next(
        (item for item in payload.get("bronze_objects", []) if item.get("api_name") == "ths_index"),
        None,
    )
    if master_object is None:
        return None
    return response_frame(lake.load_bronze(lake.root / master_object["path"])).unique(
        "ts_code", keep="last"
    )


def _apply_published_changes(
    lake: DataLake, baseline: dict[str, set[str]], before: date
) -> dict[str, set[str]]:
    """Materialize current membership from a full baseline plus published events."""
    state = {concept_id: set(members) for concept_id, members in baseline.items()}
    current = SilverVersionLedger(lake).current()
    if current is None:
        return state
    frames: list[pl.DataFrame] = []
    for item in current.get("changes", []):
        if item.get("table") != "concept_membership_changes":
            continue
        path = lake.root / item["path"]
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return state
    changes = pl.concat(frames, how="diagonal_relaxed").filter(
        pl.col("observed_date") < before
    ).sort(["observed_date", "revision_at", "concept_id", "asset_id"])
    for row in changes.to_dicts():
        members = state.setdefault(str(row["concept_id"]), set())
        if row["change_type"] == "ADD":
            members.add(str(row["asset_id"]))
        elif row["change_type"] == "REMOVE":
            members.discard(str(row["asset_id"]))
    return state


class ConceptChangeScanner:
    """Cheap daily scan; member endpoints are called only for master-level candidates."""

    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint

    @staticmethod
    def _master_map(frame: pl.DataFrame) -> dict[str, tuple[str, int | None]]:
        result = {}
        for row in frame.to_dicts():
            result[str(row["ts_code"])] = (
                str(row.get("name") or ""), ConceptSnapshotPipeline._expected_count(row.get("count"))
            )
        return result

    def sync(self, snapshot_date: date) -> Path:
        pattern = f"concept_change_scan_{snapshot_date:%Y%m%d}_*.json"
        for path in sorted(self.lake.manifests.glob(pattern), reverse=True):
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "passed":
                return path
        observed_at = datetime.now(UTC)
        run_id = f"concept_change_scan_{snapshot_date:%Y%m%d}_{uuid.uuid4().hex[:8]}"
        baseline = _latest_full_snapshot(self.lake, snapshot_date)
        if baseline is None:
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "concept_change_scan",
                "status": "full_reconciliation_required", "snapshot_date": snapshot_date.isoformat(),
                "reason": "NO_FULL_BASELINE", "observed_at": observed_at.isoformat(),
            })
        baseline_payload, baseline_frame = baseline
        try:
            master_response = self.client.query(
                "ths_index", {"exchange": "A", "type": "N"}, MASTER_FIELDS
            )
        except Exception as exc:
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "concept_change_scan",
                "status": "source_incomplete", "snapshot_date": snapshot_date.isoformat(),
                "observed_at": observed_at.isoformat(),
                "source_id": "tushare.ths_index+conditional_ths_member",
                "quality_gate": {
                    "status": "failed", "reason": "SOURCE_INCOMPLETE", "detail": str(exc),
                },
            })
        master_bronze = self.lake.save_bronze(master_response, observed_at)
        current_master = response_frame(master_response)
        if current_master.is_empty() or "ts_code" not in current_master.columns:
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "concept_change_scan",
                "status": "source_incomplete", "snapshot_date": snapshot_date.isoformat(),
                "observed_at": observed_at.isoformat(),
                "source_id": "tushare.ths_index+conditional_ths_member",
                "bronze_objects": [master_bronze],
                "quality_gate": {"status": "failed", "reason": "CONCEPT_MASTER_EMPTY"},
            })
        current_master = current_master.unique("ts_code", keep="last")
        previous_master = _latest_scan_master(self.lake, snapshot_date)
        if previous_master is None:
            previous_master_object = next(
                item for item in baseline_payload["bronze_objects"] if item["api_name"] == "ths_index"
            )
            previous_master = response_frame(
                self.lake.load_bronze(self.lake.root / previous_master_object["path"])
            ).unique("ts_code", keep="last")
        current_map, previous_map = self._master_map(current_master), self._master_map(previous_master)
        candidate_ids = sorted(
            concept_id for concept_id in set(current_map) | set(previous_map)
            if current_map.get(concept_id) != previous_map.get(concept_id)
        )
        baseline_members: dict[str, set[str]] = {}
        for key, group in baseline_frame.partition_by("concept_id", as_dict=True).items():
            concept_id = key[0] if isinstance(key, tuple) else key
            baseline_members[str(concept_id)] = set(group.get_column("asset_id").to_list())
        current_members = _apply_published_changes(
            self.lake, baseline_members, snapshot_date
        )
        bronze_objects = [master_bronze]
        events: list[dict[str, object]] = []
        count_shortfalls: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        for concept_id in candidate_ids:
            old_members = current_members.get(concept_id, set())
            if concept_id in current_map:
                try:
                    response = self.client.query(
                        "ths_member", {"ts_code": concept_id}, MEMBER_FIELDS
                    )
                except Exception as exc:
                    failures.append({"concept_id": concept_id, "detail": str(exc)})
                    continue
                bronze = self.lake.save_bronze(response, observed_at)
                bronze_objects.append(bronze)
                frame = response_frame(response)
                if "is_new" in frame.columns:
                    frame = frame.filter(pl.col("is_new") == "Y")
                new_members = set(frame.get_column("con_code").drop_nulls().to_list())
                expected = current_map[concept_id][1]
                if not new_members or (expected is not None and len(new_members) < expected):
                    count_shortfalls.append({
                        "concept_id": concept_id, "expected": expected,
                        "observed": len(new_members),
                    })
                    continue
            else:
                new_members = set()
            concept_name = current_map.get(concept_id, previous_map.get(concept_id, ("", None)))[0]
            for asset_id in sorted(new_members - old_members):
                events.append({
                    "observed_date": snapshot_date, "concept_id": concept_id,
                    "concept_name": concept_name, "asset_id": asset_id, "change_type": "ADD",
                    "source_id": "tushare.ths_index+ths_member", "baseline_snapshot_date": baseline_payload["snapshot_date"],
                })
            for asset_id in sorted(old_members - new_members):
                events.append({
                    "observed_date": snapshot_date, "concept_id": concept_id,
                    "concept_name": concept_name, "asset_id": asset_id, "change_type": "REMOVE",
                    "source_id": "tushare.ths_index+ths_member", "baseline_snapshot_date": baseline_payload["snapshot_date"],
                })
        if failures or count_shortfalls:
            return self.lake.write_manifest(run_id, {
                "schema_version": 1, "run_id": run_id, "job": "concept_change_scan",
                "status": "source_incomplete", "snapshot_date": snapshot_date.isoformat(),
                "observed_at": observed_at.isoformat(), "source_id": "tushare.ths_index+conditional_ths_member",
                "bronze_objects": bronze_objects, "candidate_concepts": candidate_ids,
                "member_requests": len(candidate_ids), "event_rows": 0,
                "quality_gate": {
                    "status": "failed", "reason": "SOURCE_INCOMPLETE",
                    "failures": failures, "count_shortfalls": count_shortfalls,
                },
            })
        changes = []
        ledger = SilverVersionLedger(self.lake)
        current_version = ledger.current()
        silver_version_id = current_version["version_id"] if current_version else None
        if events:
            output = RevisionedSilverStore(self.lake).append(
                "concept_membership_changes", pl.DataFrame(events), effective_date=snapshot_date,
                first_seen_at=observed_at,
            )
            current = ledger.current()
            if current is None:
                raise RuntimeError("CONCEPT_CHANGE_SCAN_REQUIRES_SILVER_BASE_VERSION")
            version = ledger.publish(
                base_id=current["base_id"], base_snapshot_sha256=current["base_snapshot_sha256"],
                observed_at=observed_at, changes=[*current["changes"], output],
                quality_gate={"status": "passed", "event_rows": len(events)},
            )
            silver_version_id = version["version_id"]
            changes.append(output)
        return self.lake.write_manifest(run_id, {
            "schema_version": 1, "run_id": run_id, "job": "concept_change_scan",
            "status": "passed", "snapshot_date": snapshot_date.isoformat(),
            "observed_at": observed_at.isoformat(), "baseline_snapshot_date": baseline_payload["snapshot_date"],
            "source_id": "tushare.ths_index+conditional_ths_member",
            "source_credential_fingerprint": self.token_fingerprint,
            "bronze_objects": bronze_objects, "candidate_concepts": candidate_ids,
            "member_requests": len(candidate_ids), "event_rows": len(events),
            "changes": changes, "silver_version_id": silver_version_id,
            "detection_scope": {
                "daily": "master_add_remove_name_or_count_change",
                "equal_count_substitution": "periodic_full_reconciliation_only",
            },
            "quality_gate": {"status": "passed", "master_concepts": len(current_map)},
        })


class _StartRateLimiter:
    """Serialize request starts while allowing responses to overlap."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval = max(0.0, interval_seconds)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.interval
        if delay:
            time.sleep(delay)


class ConceptSnapshotPipeline:
    def __init__(self, client: TushareClient, lake: DataLake, token_fingerprint: str) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint

    def _manifest(
        self,
        snapshot_date: date,
        *,
        status: str,
        observed_at: datetime,
        extra: dict[str, object],
    ):
        run_id = f"concept_snapshot_{snapshot_date:%Y%m%d}_{uuid.uuid4().hex[:8]}"
        return self.lake.write_manifest(
            run_id,
            {
                "schema_version": 2,
                "run_id": run_id,
                "job": "concept_membership_snapshot",
                "status": status,
                "snapshot_date": snapshot_date.isoformat(),
                "observed_at": observed_at.isoformat(),
                "source_id": "tushare.ths_index+ths_member",
                "source_credential_fingerprint": self.token_fingerprint,
                "carry_forward_allowed": False,
                **extra,
            },
        )

    def _existing_passed(self, snapshot_date: date):
        pattern = f"concept_snapshot_{snapshot_date:%Y%m%d}_*.json"
        for path in sorted(self.lake.manifests.glob(pattern), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                payload.get("status") == "passed"
                and payload.get("source_id") == "tushare.ths_index+ths_member"
                and payload.get("output", {}).get("path")
                and (self.lake.root / payload["output"]["path"]).exists()
            ):
                return path
        return None

    def _replayable_run(self, snapshot_date: date):
        """Reuse a complete THS Bronze fetch after a stricter gate rejected it."""
        pattern = f"concept_snapshot_{snapshot_date:%Y%m%d}_*.json"
        for path in sorted(self.lake.manifests.glob(pattern), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            objects = payload.get("bronze_objects") or []
            if (
                payload.get("source_id") == "tushare.ths_index+ths_member"
                and sum(item.get("api_name") == "ths_index" for item in objects) == 1
                and sum(item.get("api_name") == "ths_member" for item in objects) > 0
                and all((self.lake.root / item["path"]).exists() for item in objects)
            ):
                return [
                    (item, self.lake.load_bronze(self.lake.root / item["path"]))
                    for item in objects
                ]
        return None

    @staticmethod
    def _expected_count(value: object) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def sync(
        self,
        snapshot_date: date,
        *,
        force: bool = False,
        request_delay: float = 0.36,
        workers: int = 8,
        progress: Callable[[str], None] | None = None,
    ):
        existing = self._existing_passed(snapshot_date)
        if existing is not None and not force:
            return existing

        observed_at = datetime.now(UTC)
        bronze_objects: list[dict[str, object]] = []
        replay = self._replayable_run(snapshot_date) if not force else None
        replay_members: dict[str, tuple[TushareResponse, dict[str, object]]] = {}
        if replay is not None:
            bronze_objects = [item for item, _response in replay]
            master_response = next(
                response for _item, response in replay if response.api_name == "ths_index"
            )
            for bronze, response in replay:
                if response.api_name != "ths_member":
                    continue
                replay_members[str(response.params["ts_code"])] = (response, bronze)
            if progress:
                progress(f"replaying {len(replay_members)} concept responses from Bronze")
        else:
            try:
                master_response = self.client.query(
                    "ths_index", {"exchange": "A", "type": "N"}, MASTER_FIELDS
                )
            except Exception as exc:  # the client normalizes transport/vendor errors
                return self._manifest(
                    snapshot_date,
                    status="source_incomplete",
                    observed_at=observed_at,
                    extra={
                        "quality_gate": {
                            "status": "failed", "reason": "SOURCE_INCOMPLETE", "detail": str(exc)
                        }
                    },
                )
            bronze_objects.append(self.lake.save_bronze(master_response, observed_at))
        master = response_frame(master_response)
        if master.is_empty() or "ts_code" not in master.columns:
            return self._manifest(
                snapshot_date,
                status="source_incomplete",
                observed_at=observed_at,
                extra={
                    "bronze_objects": bronze_objects,
                    "quality_gate": {"status": "failed", "reason": "CONCEPT_MASTER_EMPTY"},
                },
            )
        master = master.unique("ts_code", keep="last").sort("ts_code")
        concept_rows = master.to_dicts()
        limiter = _StartRateLimiter(request_delay)

        def fetch(row: dict[str, object]) -> tuple[dict[str, object], TushareResponse]:
            limiter.wait()
            return row, self.client.query(
                "ths_member", {"ts_code": str(row["ts_code"])}, MEMBER_FIELDS
            )

        successes: list[tuple[dict[str, object], TushareResponse]] = []
        failures: list[dict[str, str]] = []
        completed = 0
        if replay_members:
            for row in concept_rows:
                concept_id = str(row["ts_code"])
                if concept_id in replay_members:
                    successes.append((row, replay_members[concept_id][0]))
                else:
                    failures.append({"concept_id": concept_id, "detail": "BRONZE_RESPONSE_MISSING"})
        else:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(fetch, row): row for row in concept_rows}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        successes.append(future.result())
                    except Exception as exc:
                        failures.append({"concept_id": str(row["ts_code"]), "detail": str(exc)})
                    completed += 1
                    if progress and (completed % 25 == 0 or completed == len(concept_rows)):
                        progress(
                            f"concept members {completed}/{len(concept_rows)}; failures={len(failures)}"
                        )

        published_at = datetime.now(UTC)
        member_frames: list[pl.DataFrame] = []
        count_shortfalls: list[dict[str, object]] = []
        count_advisories: list[dict[str, object]] = []
        concept_names = {str(row["ts_code"]): str(row.get("name") or "") for row in concept_rows}
        for row, response in sorted(successes, key=lambda pair: str(pair[0]["ts_code"])):
            concept_id = str(row["ts_code"])
            if concept_id in replay_members:
                bronze = replay_members[concept_id][1]
            else:
                bronze = self.lake.save_bronze(response, observed_at)
                bronze_objects.append(bronze)
            frame = response_frame(response)
            if "is_new" in frame.columns:
                frame = frame.filter(pl.col("is_new") == "Y")
            observed = frame.get_column("con_code").n_unique() if not frame.is_empty() else 0
            expected = self._expected_count(row.get("count"))
            if observed == 0 or (expected is not None and observed < expected):
                count_shortfalls.append(
                    {"concept_id": concept_id, "expected": expected, "observed": observed}
                )
                continue
            if expected is not None and observed > expected:
                count_advisories.append(
                    {"concept_id": concept_id, "master_count": expected, "member_count": observed}
                )
            member_frames.append(
                frame.with_columns(
                    pl.lit(snapshot_date).alias("snapshot_date"),
                    pl.lit(published_at).alias("available_at"),
                    pl.col("ts_code").alias("concept_id"),
                    pl.lit(concept_names[concept_id]).alias("concept_name"),
                    pl.col("con_code").alias("asset_id"),
                    pl.col("con_code").alias("source_asset_id"),
                    pl.col("con_name").alias("asset_name"),
                    parse_yyyymmdd("in_date").alias("in_date"),
                    parse_yyyymmdd("out_date").alias("out_date"),
                    pl.lit("tushare.ths_member").alias("source_id"),
                    pl.lit(bronze["sha256"]).alias("bronze_object_hash"),
                ).select(
                    "snapshot_date", "available_at", "concept_id", "concept_name", "asset_id",
                    "source_asset_id", "asset_name", "weight", "in_date", "out_date",
                    "is_new", "source_id", "bronze_object_hash",
                )
            )

        if failures or count_shortfalls or len(member_frames) != len(concept_rows):
            return self._manifest(
                snapshot_date,
                status="source_incomplete",
                observed_at=observed_at,
                extra={
                    "bronze_objects": bronze_objects,
                    "quality_gate": {
                        "status": "failed",
                        "reason": "SOURCE_INCOMPLETE",
                        "concepts_expected": len(concept_rows),
                        "concepts_received": len(successes),
                        "failures": failures,
                        "count_shortfalls": count_shortfalls,
                    },
                },
            )

        normalized = pl.concat(member_frames, how="vertical_relaxed").sort(
            ["concept_id", "asset_id"]
        )
        key_columns = ["snapshot_date", "concept_id", "asset_id"]
        if normalized.height != normalized.unique(key_columns).height:
            return self._manifest(
                snapshot_date,
                status="source_incomplete",
                observed_at=observed_at,
                extra={
                    "bronze_objects": bronze_objects,
                    "quality_gate": {"status": "failed", "reason": "DUPLICATE_MEMBERSHIP_KEYS"},
                },
            )

        output = RevisionedSilverStore(self.lake).append(
            "concept_membership_snapshot",
            normalized,
            effective_date=snapshot_date,
            first_seen_at=published_at,
        )
        quality_gate = {
            "status": "passed",
            "rows": normalized.height,
            "concepts": normalized.get_column("concept_id").n_unique(),
            "assets": normalized.get_column("asset_id").n_unique(),
            "master_concepts": len(concept_rows),
            "no_member_count_shortfalls": True,
            "membership_counts_exactly_match_master": not count_advisories,
            "master_count_lag_advisories": len(count_advisories),
            "master_count_lag_examples": count_advisories[:20],
        }
        ledger = SilverVersionLedger(self.lake)
        current = ledger.current()
        version = None
        if current is not None:
            version = ledger.publish(
                base_id=current["base_id"],
                base_snapshot_sha256=current["base_snapshot_sha256"],
                observed_at=published_at,
                changes=[*current["changes"], output],
                quality_gate=quality_gate,
            )
        return self._manifest(
            snapshot_date,
            status="passed",
            observed_at=observed_at,
            extra={
                "bronze_objects": bronze_objects,
                "output": output,
                "silver_version_id": version["version_id"] if version else None,
                "content_hash": json_hash(
                    normalized.select(*key_columns, "bronze_object_hash").to_dicts()
                ),
                "quality_gate": quality_gate,
            },
        )
