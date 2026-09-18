#!/usr/bin/env python3
"""Validate the frozen market base and materialize one-time BSE facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factor_matrix.base_migration import (
    materialize_bse_transfer_batch,
    materialize_security_daily_state,
    validate_legacy_base,
)
from factor_matrix.storage import DataLake


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    lake = DataLake(args.data_root)
    with lake.pipeline_lock():
        base = validate_legacy_base(lake, args.snapshot_id)
        transfer = materialize_bse_transfer_batch(lake, args.snapshot_id)
        result = {
            "base": base,
            "bse_transfer_batch": transfer,
            "security_daily_state": materialize_security_daily_state(
                lake, args.snapshot_id
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
