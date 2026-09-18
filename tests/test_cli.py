from datetime import date

from factor_matrix.cli import (
    main,
    snapshot_matches,
    universe_snapshot_covers,
)
from factor_matrix.storage import DataLake
import json
import sys

import polars as pl


def test_snapshot_match_requires_requested_as_of_date(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"as_of_date": "2026-08-07"}))
    assert snapshot_matches(snapshot, date(2026, 8, 7))
    assert not snapshot_matches(snapshot, date(2026, 8, 6))


def test_daily_current_requires_registered_universe_covering_target(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.write_manifest(
        "tradable_universe_v1_old",
        {
            "job": "tradable_universe_v1",
            "start": "2026-08-06",
            "end": "2026-08-07",
            "quality_gate": {"status": "passed"},
        },
    )
    assert universe_snapshot_covers(lake, date(2026, 8, 7))
    assert not universe_snapshot_covers(lake, date(2026, 8, 10))


def test_daily_update_skips_closed_market_without_token(tmp_path, monkeypatch, capsys) -> None:
    lake = DataLake(tmp_path / "data")
    lake.replace(
        "trade_calendar",
        pl.DataFrame(
            {
                "exchange": ["SSE"],
                "cal_date": [date(2026, 8, 8)],
                "is_open": [0],
                "pretrade_date": [date(2026, 8, 7)],
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "factor-matrix",
            "daily-update",
            "--date",
            "2026-08-08",
            "--data-root",
            str(lake.root),
        ],
    )
    assert main() == 0
    assert "not a China trading day" in capsys.readouterr().out
