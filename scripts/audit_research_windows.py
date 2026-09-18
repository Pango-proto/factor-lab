#!/usr/bin/env python3
"""Read-only coverage audit that determines whether the configured backfill is complete."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import polars as pl

from factor_matrix.research_protocol import ProtocolSealStore, ResearchProtocol
from factor_matrix.external_facts import ExternalFacts


TABLES = (
    ("prices_daily", "trade_date"), ("returns_daily", "trade_date"),
    ("valuation_daily", "trade_date"), ("trade_calendar", "cal_date"),
    ("price_limits_daily", "trade_date"), ("suspensions_daily", "trade_date"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--protocol", type=Path, default=Path("config/research_protocol_v1.json"))
    args = parser.parse_args()
    protocol = ResearchProtocol.load(args.protocol)
    ProtocolSealStore(args.data_root / "metadata" / "research_protocol.sqlite").seal(protocol)
    ExternalFacts.load(Path(protocol.external_facts_config)).sync_to_sqlite(
        args.data_root / "metadata" / "system_definitions.sqlite"
    )
    requested_start = protocol.frozen_choices["estimation_start"]
    coverage = {}
    missing = []
    for table, date_column in TABLES:
        path = args.data_root / "silver" / table / "data.parquet"
        if not path.exists():
            coverage[table] = {"status": "missing"}
            missing.append(table)
            continue
        row = pl.scan_parquet(path).select(
            pl.col(date_column).min().alias("minimum"),
            pl.col(date_column).max().alias("maximum"),
            pl.col(date_column).n_unique().alias("date_count"),
        ).collect().row(0, named=True)
        minimum = row["minimum"]
        complete = minimum is not None and minimum.isoformat() <= requested_start
        coverage[table] = {
            "status": "complete" if complete else "backfill_required",
            "minimum": minimum.isoformat() if minimum else None,
            "maximum": row["maximum"].isoformat() if row["maximum"] else None,
            "date_count": row["date_count"],
        }
        if not complete:
            missing.append(table)
    report = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "requested_estimation_start": requested_start,
        "holdout_start": protocol.holdout_start.isoformat(),
        "holdout_status": "sealed_unopened",
        "coverage": coverage,
        "status": "backfill_required" if missing else "ready",
        "blocking_tables": missing,
        "required_action": (
            "Create a separately reviewed Bronze replay migration; the frozen L0 base must not "
            "be mutated by a production CLI command."
        ) if missing else None,
        "note": "Stress window 2015-2018 is out_of_model and excluded from all estimation.",
    }
    output = args.data_root / "metadata" / "research_window_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
