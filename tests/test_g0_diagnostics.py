from __future__ import annotations

from factor_matrix.diagnostics.diff import compare_reports


def test_g0_diff_tracks_operational_metrics_and_observation_states() -> None:
    previous = {
        "run_id": "old",
        "history": {"maximum_trade_date": "2026-08-13"},
        "universe": {"latest_count": 5000, "minimum_count": 3600, "maximum_count": 5500},
        "financial_pit": {"revision_count": 100, "conservative_first_seen_count": 2},
        "transition_first_day_returns": {"non_null_count": 80, "absolute_extreme_count": 20},
        "resumption_returns_excluding_exchange_first_day": {
            "non_null_count": 300, "absolute_extreme_count": 10,
        },
        "observation_states": {"totals": {"TRADED": 1000}},
    }
    current = {
        **previous,
        "run_id": "new",
        "history": {"maximum_trade_date": "2026-08-14"},
        "universe": {"latest_count": 5002, "minimum_count": 3600, "maximum_count": 5500},
        "observation_states": {"totals": {"TRADED": 1002, "SOURCE_INCOMPLETE": 0}},
    }
    result = compare_reports(current, previous)
    assert result["previous_run_id"] == "old"
    metrics = {row["metric"] for row in result["changes"]}
    assert "history.maximum_trade_date" in metrics
    assert "universe.latest_count" in metrics
    assert "observation_states.totals.TRADED" in metrics


def test_g0_diff_without_previous_report_is_explicit() -> None:
    assert compare_reports({"run_id": "first"}, None) == {
        "previous_run_id": None,
        "changes": [],
    }
