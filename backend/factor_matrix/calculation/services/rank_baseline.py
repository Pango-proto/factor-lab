"""M2b point-in-time signal adapter and pure rank strategy, fixture scope only.

These functions receive no execution opportunities, closing prices or labels.
The strategy returns targets/intents; it cannot book trades or alter an account.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .accounting import decimal, instant, money

ASSETS = ("FIXTURE.SH", "FIXTURE.SZ")
STRATEGY = {
    "strategy_id": "rank_long_only_fixture_v1", "top_n": 1, "gross_target": "0.80",
    "rebalance": "each_explicit_session", "tie_break": "asset_id_ascending",
    "missing_all": "hold_existing", "signal_freshness": "same_session_only",
    "lot_size": 10, "limit_price": "decision_reference_price",
    "execution": "sell_then_buy_actual_cash_cap", "unfilled": "cancel_before_close",
}


def validate_signals(rows: list[dict]):
    keys = {"signal_id", "session", "asset_id", "value", "eligible", "observed_at", "available_at"}
    seen, ids = set(), set()
    for r in rows:
        if set(r) != keys or r["asset_id"] not in ASSETS or type(r["eligible"]) is not bool:
            raise ValueError("SIGNAL_SCHEMA_UNSUPPORTED")
        pair = (r["session"], r["asset_id"])
        if pair in seen or not isinstance(r["signal_id"], str) or not r["signal_id"] or r["signal_id"] in ids:
            raise ValueError("DUPLICATE_SIGNAL")
        seen.add(pair)
        ids.add(r["signal_id"])
        observed, available = instant(r["observed_at"]), instant(r["available_at"])
        if observed > available or observed.date().isoformat() != r["session"]:
            raise ValueError("SIGNAL_CLOCK_INVALID")
        if r["value"] is not None:
            decimal(r["value"])


def signal_snapshot(rows: list[dict], decision_time: str) -> list[dict]:
    """Filter availability BEFORE selection; never carry yesterday's signal."""
    now = instant(decision_time)
    session = now.date().isoformat()
    result = []
    for asset in ASSETS:
        candidates = [r for r in rows if r["asset_id"] == asset and r["session"] == session
                      and instant(r["available_at"]) <= now]
        if len(candidates) > 1:
            raise ValueError("DUPLICATE_AVAILABLE_SIGNAL")
        r = candidates[0] if candidates else None
        reason = ("not_available" if r is None else "ineligible" if not r["eligible"]
                  else "missing_value" if r["value"] is None else "eligible")
        result.append({"session": session, "decision_time": decision_time, "asset_id": asset,
            "signal_id": r["signal_id"] if r else None, "value": r["value"] if r and reason == "eligible" else None,
            "available_at": r["available_at"] if r else None, "reason": reason})
    return result


@dataclass(frozen=True)
class DecisionAccount:
    """Read-only values supplied by the backtest at the strategy decision time."""
    cash: Decimal
    receivables: Decimal
    holdings: dict[str, int]


def rank_targets(snapshot: list[dict], account: DecisionAccount, quotes: dict, *, decision_time: str):
    if set(quotes) != set(ASSETS) or set(account.holdings) != set(ASSETS):
        raise ValueError("DECISION_ASSET_AXIS_MISMATCH")
    if (len(snapshot) != len(ASSETS) or {r["asset_id"] for r in snapshot} != set(ASSETS)
        or any(r["decision_time"] != decision_time or r["session"] != instant(decision_time).date().isoformat()
               or r["reason"] not in ("eligible", "not_available", "ineligible", "missing_value")
               or (r["available_at"] is not None and instant(r["available_at"]) > instant(decision_time))
               or (r["reason"] == "eligible" and (r["available_at"] is None or r["value"] is None)) for r in snapshot)):
        raise ValueError("SIGNAL_SNAPSHOT_BINDING_MISMATCH")
    prices = {}
    for asset, q in quotes.items():
        observed = instant(q["observed_at"])
        if (set(q) != {"price", "observed_at"} or observed > instant(decision_time) or observed.date() != instant(decision_time).date()
            or decimal(q["price"]) <= 0 or decimal(q["price"]) != money(q["price"])):
            raise ValueError("INVALID_DECISION_QUOTE")
        prices[asset] = decimal(q["price"])
    if account.cash < 0 or account.receivables < 0 or any(type(v) is not int or v < 0 for v in account.holdings.values()):
        raise ValueError("INVALID_DECISION_ACCOUNT")
    nav = account.cash + account.receivables + sum(prices[a] * account.holdings[a] for a in ASSETS)
    eligible = sorted((r for r in snapshot if r["reason"] == "eligible"),
                      key=lambda r: (-decimal(r["value"]), r["asset_id"]))
    ranks = {r["asset_id"]: i + 1 for i, r in enumerate(eligible)}
    winner = eligible[0]["asset_id"] if eligible else None
    targets, intents = [], []
    session = instant(decision_time).date().isoformat()
    for asset in ASSETS:
        desired = (int(nav * Decimal("0.80") / prices[asset] / 10) * 10 if asset == winner
                   else 0 if winner else account.holdings[asset])
        delta = desired - account.holdings[asset]
        target_id = f"{session}:{asset}"
        targets.append({"target_id": target_id, "session": session, "decision_time": decision_time,
            "asset_id": asset, "rank": ranks.get(asset), "selected": asset == winner,
            "reason": "rank_target" if winner else "hold_no_eligible_signal",
            "decision_nav": nav, "reference_price": prices[asset], "reference_observed_at": quotes[asset]["observed_at"],
            "opening_quantity": account.holdings[asset], "target_quantity": desired,
            "target_weight": prices[asset] * desired / nav if nav else Decimal(0),
            "risk_exposure_snapshot": None, "risk_forecast_snapshot": None, "risk_status": "unavailable"})
        if delta:
            intents.append({"intent_id": target_id, "target_id": target_id, "decision_time": decision_time,
                "asset_id": asset, "side": "buy" if delta > 0 else "sell",
                "requested_quantity": abs(delta), "limit_price": prices[asset]})
    return targets, intents
