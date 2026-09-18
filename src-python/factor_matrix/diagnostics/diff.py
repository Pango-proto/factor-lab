"""Small deterministic diffs for successive G0 diagnostic reports."""

from __future__ import annotations

from typing import Any


TRACKED_PATHS = (
    ("history", "maximum_trade_date"),
    ("universe", "latest_count"),
    ("universe", "minimum_count"),
    ("universe", "maximum_count"),
    ("financial_pit", "revision_count"),
    ("financial_pit", "conservative_first_seen_count"),
    ("transition_first_day_returns", "non_null_count"),
    ("transition_first_day_returns", "absolute_extreme_count"),
    ("resumption_returns_excluding_exchange_first_day", "non_null_count"),
    ("resumption_returns_excluding_exchange_first_day", "absolute_extreme_count"),
)


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def compare_reports(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    if previous is None:
        return {"previous_run_id": None, "changes": []}
    changes = []
    for path in TRACKED_PATHS:
        before, after = _get(previous, path), _get(current, path)
        if before != after:
            changes.append({"metric": ".".join(path), "before": before, "after": after})

    current_states = current.get("observation_states", {}).get("totals", {})
    previous_states = previous.get("observation_states", {}).get("totals", {})
    for state in sorted(set(current_states) | set(previous_states)):
        before, after = previous_states.get(state, 0), current_states.get(state, 0)
        if before != after:
            changes.append({
                "metric": f"observation_states.totals.{state}",
                "before": before,
                "after": after,
            })
    return {"previous_run_id": previous.get("run_id"), "changes": changes}
