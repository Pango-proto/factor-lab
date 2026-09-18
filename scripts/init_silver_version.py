#!/usr/bin/env python3
"""Publish the initial no-delta Silver version over the validated base."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from factor_matrix.revisioned_silver import SilverVersionLedger
from factor_matrix.storage import DataLake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    lake = DataLake(args.data_root)
    base_path = lake.metadata / "base_manifests" / "legacy_base_v1.json"
    state_path = lake.metadata / "base_manifests" / "security_daily_state_v1.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    ledger = SilverVersionLedger(lake)
    current = ledger.current()
    if current is None:
        current = ledger.publish(
            base_id=base["base_id"],
            base_snapshot_sha256=base["snapshot_sha256"],
            observed_at=datetime.now(UTC),
            changes=[
                {
                    "table": "security_daily_state",
                    **state["output"],
                }
            ],
            quality_gate={
                "status": "passed",
                "base_validation": base["return_validation"],
                "security_daily_state": state["quality_gate"],
            },
        )
    print(json.dumps(current, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
