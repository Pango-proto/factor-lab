from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Mapping

from ...factor_engine import FactorRegistryStore
from .contracts import ResearchEvent


def configuration_sha(configuration: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(configuration), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResearchEventStore:
    """Append-only audit boundary required before any L2a result is returned."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self, event: ResearchEvent, *, attempt_id: str | None = None,
    ) -> None:
        FactorRegistryStore(self.path).initialize()
        with sqlite3.connect(self.path) as connection:
            try:
                if attempt_id is not None and not connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM research_attempt WHERE attempt_id=?)",
                    (attempt_id,),
                ).fetchone()[0]:
                    raise ValueError("L2A_ATTEMPT_NOT_ISSUED")
                connection.execute(
                    """
                    INSERT INTO research_events (
                      event_id, attempt_id, run_at, factor_ids_json, config_sha, data_end_date,
                      touched_holdout, headline_ic, decision, operator, payload_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        event.event_id, attempt_id, event.run_at.isoformat(), json.dumps(event.factor_ids),
                        event.config_sha, event.data_end_date.isoformat(),
                        int(event.touched_holdout), event.headline_ic,
                        event.decision, event.operator,
                        json.dumps(dict(event.payload), ensure_ascii=False, sort_keys=True),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("RESEARCH_EVENT_APPEND_ONLY_CONFLICT") from exc

    def issue_attempt(self, **kwargs: object) -> str:
        """Issue an append-only research attempt through the L2 audit boundary."""
        return FactorRegistryStore(self.path).issue_research_attempt(**kwargs)

    def validate_attempt(
        self, attempt_id: str, *, feature_id: str, feature_version: int,
        risk_set_version: int, horizon_days: int,
    ) -> None:
        FactorRegistryStore(self.path).validate_research_attempt(
            attempt_id, feature_id=feature_id, feature_version=feature_version,
            risk_set_version=risk_set_version, horizon_days=horizon_days,
        )

    def resolve_attempt(self, attempt_id: str, *, submitted: bool, decision: str) -> None:
        FactorRegistryStore(self.path).resolve_research_attempt(
            attempt_id, submitted=submitted, decision=decision,
        )

    def attempt_count(self, family_root_id: str | None = None) -> int:
        return FactorRegistryStore(self.path).research_attempt_count(family_root_id)

    def holdout_has_ever_been_touched(self) -> bool:
        FactorRegistryStore(self.path).initialize()
        with sqlite3.connect(self.path) as connection:
            return bool(connection.execute(
                "SELECT EXISTS(SELECT 1 FROM research_events WHERE touched_holdout=1)"
            ).fetchone()[0])

    def count(self) -> int:
        FactorRegistryStore(self.path).initialize()
        with sqlite3.connect(self.path) as connection:
            return int(connection.execute("SELECT count(*) FROM research_events").fetchone()[0])

    def horizons_for_factor(self, factor_id: str) -> frozenset[int]:
        FactorRegistryStore(self.path).initialize()
        values: set[int] = set()
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT factor_ids_json, payload_json FROM research_events"
            ).fetchall()
        for factor_ids_json, payload_json in rows:
            if factor_id not in json.loads(factor_ids_json):
                continue
            horizon = json.loads(payload_json).get("evaluation_horizon_days")
            if isinstance(horizon, int):
                values.add(horizon)
        return frozenset(values)


    def evaluation_status(self) -> str:
        return (
            "degraded_holdout_contaminated"
            if self.holdout_has_ever_been_touched()
            else "eligible_for_out_of_sample_claim"
        )


class SemanticProbeLedger:
    """Append-only accounting for label-free, basis-scoped semantic decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS semantic_probe_ledger (
                  event_id TEXT PRIMARY KEY,
                  risk_basis_id TEXT NOT NULL,
                  descriptor_manifest_sha TEXT NOT NULL,
                  protocol_sha TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_semantic_probe_basis_manifest
                ON semantic_probe_ledger(risk_basis_id, descriptor_manifest_sha)
            """)

    def record(
        self, *, event_id: str, risk_basis_id: str, descriptor_manifest_sha: str,
        protocol_sha: str, created_at: str,
    ) -> None:
        if any(len(value) != 64 for value in (descriptor_manifest_sha, protocol_sha)):
            raise ValueError("SEMANTIC_PROBE_LEDGER_SHA_INVALID")
        self.initialize()
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "INSERT INTO semantic_probe_ledger VALUES (?,?,?,?,?)",
                    (event_id, risk_basis_id, descriptor_manifest_sha, protocol_sha, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("SEMANTIC_PROBE_LEDGER_APPEND_ONLY_CONFLICT") from exc

    def count(self) -> int:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            return int(connection.execute(
                "SELECT count(*) FROM semantic_probe_ledger"
            ).fetchone()[0])

    def count_for_basis(self, risk_basis_id: str) -> int:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            return int(connection.execute(
                "SELECT count(*) FROM semantic_probe_ledger WHERE risk_basis_id=?",
                (risk_basis_id,),
            ).fetchone()[0])
