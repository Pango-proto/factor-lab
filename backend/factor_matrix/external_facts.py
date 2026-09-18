"""Auditable time-varying external facts; never use a timeless market-rule constant."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalFact:
    fact_key: str
    scope: str | None
    effective_from: date
    effective_to: date | None
    value: Any
    source_url: str


class ExternalFacts:
    def __init__(self, facts: tuple[ExternalFact, ...]) -> None:
        self.facts = facts
        self._validate()

    @classmethod
    def load(cls, path: Path) -> "ExternalFacts":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(ExternalFact(
            fact_key=row["fact_key"], scope=row.get("scope"),
            effective_from=date.fromisoformat(row["effective_from"]),
            effective_to=date.fromisoformat(row["effective_to"]) if row.get("effective_to") else None,
            value=row["value"], source_url=row["source_url"],
        ) for row in payload["facts"]))

    def _validate(self) -> None:
        grouped: dict[tuple[str, str | None], list[ExternalFact]] = {}
        for fact in self.facts:
            if not fact.fact_key or not fact.source_url.startswith("https://"):
                raise ValueError("EXTERNAL_FACT_SOURCE_REQUIRED")
            if fact.effective_to is not None and fact.effective_to < fact.effective_from:
                raise ValueError("EXTERNAL_FACT_INTERVAL_INVALID")
            grouped.setdefault((fact.fact_key, fact.scope), []).append(fact)
        for facts in grouped.values():
            ordered = sorted(facts, key=lambda item: item.effective_from)
            for previous, current in zip(ordered, ordered[1:]):
                if previous.effective_to is None or previous.effective_to >= current.effective_from:
                    raise ValueError("EXTERNAL_FACT_INTERVAL_OVERLAP")

    def lookup(self, fact_key: str, as_of: date, *, scope: str | None = None) -> ExternalFact:
        matches = [fact for fact in self.facts if fact.fact_key == fact_key
                   and fact.scope == scope and fact.effective_from <= as_of
                   and (fact.effective_to is None or as_of <= fact.effective_to)]
        if len(matches) != 1:
            raise KeyError(f"EXTERNAL_FACT_NOT_UNIQUE key={fact_key} scope={scope} date={as_of}")
        return matches[0]

    def regime_break_dates(self) -> tuple[date, ...]:
        return tuple(sorted(fact.effective_from for fact in self.facts
                            if fact.fact_key == "regime_break"))

    def scope_active(self, fact_key: str, scope: str, as_of: date) -> bool:
        return any(
            fact.fact_key == fact_key and fact.scope == scope
            and fact.effective_from <= as_of
            for fact in self.facts
        )

    def sync_to_sqlite(self, path: Path) -> None:
        """Materialize the reviewed JSON source into an immutable query table."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS external_facts (
                  fact_key TEXT NOT NULL, scope TEXT,
                  effective_from TEXT NOT NULL, effective_to TEXT,
                  value_json TEXT NOT NULL, source_url TEXT NOT NULL,
                  PRIMARY KEY (fact_key, scope, effective_from)
                );
                CREATE TRIGGER IF NOT EXISTS external_facts_update_forbidden
                BEFORE UPDATE ON external_facts BEGIN
                  SELECT RAISE(ABORT, 'EXTERNAL_FACT_UPDATE_FORBIDDEN_APPEND_NEW_INTERVAL');
                END;
                CREATE TRIGGER IF NOT EXISTS external_facts_delete_forbidden
                BEFORE DELETE ON external_facts BEGIN
                  SELECT RAISE(ABORT, 'EXTERNAL_FACT_DELETE_FORBIDDEN');
                END;
            """)
            for fact in self.facts:
                row = (
                    fact.fact_key, fact.scope, fact.effective_from.isoformat(),
                    fact.effective_to.isoformat() if fact.effective_to else None,
                    json.dumps(fact.value, ensure_ascii=False, sort_keys=True), fact.source_url,
                )
                existing = connection.execute(
                    """SELECT fact_key, scope, effective_from, effective_to, value_json, source_url
                       FROM external_facts WHERE fact_key=? AND scope IS ? AND effective_from=?""",
                    row[:3],
                ).fetchone()
                if existing is not None and existing != row:
                    raise RuntimeError("EXTERNAL_FACT_IMMUTABLE_CONFLICT")
                if existing is not None:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO external_facts VALUES (?,?,?,?,?,?)", row,
                )
