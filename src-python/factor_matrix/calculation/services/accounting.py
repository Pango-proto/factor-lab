"""Deterministic, long-only fixture accounting. No research or market-data reads.

Money is Decimal CNY; serialized tables retain exact decimal strings. The v1
fixture policy charges commission cumulatively per order, rounding half-up to
cents. This aggregation/rounding policy is NOT a verified broker convention.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal
import re

ZERO = Decimal("0.00")
RATE = Decimal("0.0001854")


def decimal(value) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError("EXACT_DECIMAL_STRING_REQUIRED")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("NONFINITE_AMOUNT")
    return result


def money(value) -> Decimal:
    return decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def instant(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    return result


def quantity(value: int, *, allow_zero=False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError("INVALID_QUANTITY")
    return value


def commission(notional: Decimal) -> Decimal:
    if notional < 0:
        raise ValueError("NEGATIVE_NOTIONAL")
    return max(Decimal("5.00"), money(notional * RATE)) if notional else ZERO


@dataclass
class Order:
    order_id: str
    charge_unit_id: str
    asset_id: str
    side: Literal["buy", "sell"]
    quantity: int
    limit_price: Decimal
    decision_time: str
    earliest_execution: str
    filled_quantity: int = 0
    notional: Decimal = ZERO
    commission_paid: Decimal = ZERO
    status: str = "open"
    reason: str | None = None

    @property
    def remaining(self):
        return self.quantity - self.filled_quantity if self.status == "open" else 0


@dataclass
class Lot:
    asset_id: str
    quantity: int
    sellable_at: str
    fill_id: str


class AccountingEngine:
    """Atomic ordered events; exact duplicates are no-ops, conflicts are errors."""

    def __init__(self, initial_cash: str, *, lot_size: int, asset_scope: tuple[str, ...] = ("FIXTURE.SH", "FIXTURE.SZ")):
        if (not asset_scope or len(set(asset_scope)) != len(asset_scope)
            or any(a not in ("FIXTURE.SH", "FIXTURE.SZ") and re.fullmatch(r"[0-9]{6}\.(SH|SZ)", a) is None for a in asset_scope)):
            raise ValueError("UNSUPPORTED_EQUITY_SCOPE")
        if type(self) is AccountingEngine and any(a not in ("FIXTURE.SH", "FIXTURE.SZ") for a in asset_scope):
            raise ValueError("REAL_EQUITY_REQUIRES_RULE_AWARE_ENGINE")
        self.asset_scope = tuple(asset_scope)
        self.initial_cash = money(initial_cash)
        if self.initial_cash < 0:
            raise ValueError("NEGATIVE_INITIAL_CASH")
        self.cash = self.initial_cash
        self.lot_size = quantity(lot_size)
        self.orders: dict[str, Order] = {}
        self.lots: list[Lot] = []
        self.receivables: dict[str, dict] = {}
        self.seen: dict[str, dict] = {}
        self.last_time: datetime | None = None
        self.tables = {name: [] for name in (
            "events", "fills", "fees", "cash_ledger", "share_ledger",
            "corporate_actions", "positions", "daily_nav",
        )}

    def total(self, asset, at=None):
        return sum(lot.quantity for lot in self.lots if lot.asset_id == asset
                   and (at is None or instant(lot.sellable_at) <= instant(at)))

    def buy_reserve(self, order):
        value = money(order.limit_price * order.remaining)
        return value + commission(order.notional + value) - order.commission_paid if order.remaining else ZERO

    @property
    def reserved_cash(self):
        return sum((self.buy_reserve(o) for o in self.orders.values() if o.side == "buy"), ZERO)

    def reserved_shares(self, asset, except_order=None):
        return sum(o.remaining for o in self.orders.values()
                   if o.asset_id == asset and o.side == "sell" and o.order_id != except_order)

    def _quantity_reason(self, order, event):
        return "quantity_constraint" if order.quantity % self.lot_size else None

    def _submission_sellable_time(self, order, event):
        return event['time']

    def _fill_quantity(self, order, capacity):
        return min(order.remaining, capacity // self.lot_size * self.lot_size)

    def _fee_deltas(self, order, value, event):
        return {"commission": commission(order.notional + value) - order.commission_paid}

    def apply(self, event: dict):
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("INVALID_EVENT_ID")
        if event_id in self.seen:
            if self.seen[event_id] != event:
                raise ValueError("EVENT_ID_CONFLICT")
            return "duplicate_ignored"
        now = instant(event["time"])
        if self.last_time is not None and now < self.last_time:
            raise ValueError("EVENT_TIME_REVERSED")
        working = deepcopy(self)
        if event["kind"] in ("dividend_record", "dividend_pay", "split"):
            if not isinstance(event["action_id"], str) or not event["action_id"]:
                raise ValueError("INVALID_ACTION_ID")
            if event["kind"] != "dividend_pay" and event["asset_id"] not in self.asset_scope:
                raise ValueError("UNSUPPORTED_FIXTURE_INSTRUMENT")
        handlers = {"submit": working._submit, "execute": working._execute,
                    "cancel": working._cancel, "dividend_record": working._dividend_record,
                    "dividend_pay": working._dividend_pay, "split": working._split,
                    "mark": working._mark}
        if hasattr(working, "_external_flow"):
            handlers["external_flow"] = working._external_flow
        if event["kind"] not in handlers:
            raise ValueError("UNKNOWN_EVENT_KIND")
        reason = handlers[event["kind"]](event)
        if working.cash < 0 or working.cash < working.reserved_cash:
            raise ValueError("CASH_RESERVATION_INVARIANT")
        working.tables["events"].append({"event_id": event_id, "time": event["time"],
                                        "kind": event["kind"], "result": reason})
        working.last_time = now
        working.seen[event_id] = deepcopy(event)
        self.__dict__ = working.__dict__
        return reason

    def _submit(self, e):
        oid = e["order_id"]
        if not isinstance(oid, str) or not oid or oid in self.orders:
            raise ValueError("ORDER_ID_CONFLICT")
        if e["side"] not in ("buy", "sell") or e["asset_id"] not in self.asset_scope:
            raise ValueError("UNSUPPORTED_FIXTURE_INSTRUMENT_OR_SIDE")
        price = decimal(e["limit_price"])
        if price <= 0 or price != money(price):
            raise ValueError("INVALID_CENT_PRICE")
        if instant(e["decision_time"]) != instant(e["time"]) or instant(e["earliest_execution"]) <= instant(e["time"]):
            raise ValueError("EXECUTION_MUST_FOLLOW_DECISION")
        o = Order(oid, oid, e["asset_id"], e["side"], quantity(e["quantity"]), price,
                  e["decision_time"], e["earliest_execution"])
        reason = None
        if self._quantity_reason(o, e):
            reason = self._quantity_reason(o, e)
        elif o.side == "buy" and self.buy_reserve(o) > self.cash - self.reserved_cash:
            reason = "insufficient_cash"
        elif o.side == "sell" and o.quantity > self.total(o.asset_id, self._submission_sellable_time(o, e)) - self.reserved_shares(o.asset_id):
            reason = "insufficient_sellable"
        # A sell can cost more than its proceeds at very small notionals. Execute
        # checks cash atomically; it never silently creates a negative balance.
        if reason:
            o.status, o.reason = "rejected", reason
        self.orders[oid] = o
        return reason or "accepted"

    def _execute(self, e):
        o = self.orders[e["order_id"]]
        if not isinstance(e["fill_id"], str) or not e["fill_id"]:
            raise ValueError("INVALID_FILL_ID")
        if any(row["fill_id"] == e["fill_id"] for row in self.tables["fills"]):
            raise ValueError("FILL_ID_CONFLICT")
        if instant(e["quote_time"]) != instant(e["time"]):
            raise ValueError("EXECUTION_REQUIRES_CONTEMPORANEOUS_QUOTE")
        price = decimal(e["price"])
        if price <= 0 or price != money(price):
            raise ValueError("INVALID_CENT_PRICE")
        capacity = quantity(e["capacity"], allow_zero=True)
        if type(e["halted"]) is not bool or type(e["side_blocked"]) is not bool:
            raise ValueError("EXPLICIT_MARKET_STATE_REQUIRED")
        if instant(e["sellable_at"]) < instant(e["time"]):
            raise ValueError("SETTLEMENT_BEFORE_FILL")
        if o.status != "open":
            return "order_not_open"
        if instant(e["time"]) < instant(o.earliest_execution):
            return "before_earliest_execution"
        if e["halted"]:
            return "halted"
        if e["side_blocked"]:
            return "price_limit_blocked"
        if (o.side == "buy" and price > o.limit_price) or (o.side == "sell" and price < o.limit_price):
            return "outside_order_limit"
        qty = self._fill_quantity(o, capacity)
        if not qty:
            return "no_capacity"
        value = money(price * qty)
        fees = self._fee_deltas(o, value, e)
        fee = fees["commission"]
        cash_delta = (-value if o.side == "buy" else value) - sum(fees.values(), ZERO)
        other_reserve = self.reserved_cash - (self.buy_reserve(o) if o.side == "buy" else ZERO)
        if self.cash + cash_delta < other_reserve:
            return "insufficient_cash_at_fill"
        if o.side == "sell":
            if qty > self.total(o.asset_id, e["time"]) - self.reserved_shares(o.asset_id, o.order_id):
                return "insufficient_sellable_at_fill"
            remaining = qty
            for lot in self.lots:
                if lot.asset_id == o.asset_id and instant(lot.sellable_at) <= instant(e["time"]):
                    taken = min(lot.quantity, remaining)
                    lot.quantity -= taken
                    remaining -= taken
        else:
            self.lots.append(Lot(o.asset_id, qty, e["sellable_at"], e["fill_id"]))
        o.filled_quantity += qty
        o.notional += value
        o.commission_paid += fee
        if o.filled_quantity == o.quantity:
            o.status = "filled"
        row = {"event_id": e["event_id"], "time": e["time"], "fill_id": e["fill_id"],
               "order_id": o.order_id, "charge_unit_id": o.charge_unit_id, "asset_id": o.asset_id,
               "side": o.side, "quantity": qty, "price": price, "notional": value,
               "sellable_at": e["sellable_at"]}
        self.tables["fills"].append(row)
        self._cash(e, "trade", -value if o.side == "buy" else value, e["fill_id"])
        for component, amount in fees.items():
            extra = {"cumulative_commission": o.commission_paid} if component == "commission" else {}
            self.tables["fees"].append({**row, "component": component, "amount": amount, **extra})
            self._cash(e, component, -amount, e["fill_id"])
        self._shares(e, o.asset_id, qty if o.side == "buy" else -qty, e["fill_id"])
        return "filled" if o.status == "filled" else "partial_fill"

    def _cash(self, e, kind, delta, reference):
        self.cash += delta
        self.tables["cash_ledger"].append({"event_id": e["event_id"], "time": e["time"],
            "kind": kind, "reference": reference, "delta": delta, "balance": self.cash})

    def _shares(self, e, asset, delta, reference):
        self.tables["share_ledger"].append({"event_id": e["event_id"], "time": e["time"],
            "asset_id": asset, "reference": reference, "delta": delta, "balance": self.total(asset)})

    def _cancel(self, e):
        o = self.orders[e["order_id"]]
        if o.status != "open":
            return "order_not_open"
        o.status, o.reason = "cancelled", "explicit_cancel"
        return "cancelled"

    def _dividend_record(self, e):
        aid = e["action_id"]
        if any(row["action_id"] == aid for row in self.tables["corporate_actions"]):
            raise ValueError("ACTION_ID_CONFLICT")
        rate = decimal(e["cash_per_share"])
        if rate < 0:
            raise ValueError("NEGATIVE_DIVIDEND")
        shares = self.total(e["asset_id"])
        record = {"action_id": aid, "asset_id": e["asset_id"], "entitled_quantity": shares, "cash_per_share": rate,
                  "amount": money(rate * shares), "paid": False}
        self.receivables[aid] = record
        self.tables["corporate_actions"].append({**record, "event_id": e["event_id"],
            "time": e["time"], "kind": "dividend_record"})
        return "dividend_receivable_recorded"

    def _dividend_pay(self, e):
        record = self.receivables[e["action_id"]]
        if record["paid"]:
            raise ValueError("DIVIDEND_ALREADY_PAID")
        self._cash(e, "dividend", record["amount"], e["action_id"])
        record["paid"] = True
        self.tables["corporate_actions"].append({**record, "event_id": e["event_id"],
            "time": e["time"], "kind": "dividend_pay"})
        return "dividend_paid"

    def _split(self, e):
        if any(row["action_id"] == e["action_id"] for row in self.tables["corporate_actions"]):
            raise ValueError("ACTION_ID_CONFLICT")
        numerator, denominator = quantity(e["numerator"]), quantity(e["denominator"])
        lots = [lot for lot in self.lots if lot.asset_id == e["asset_id"]]
        if any(lot.quantity * numerator % denominator for lot in lots):
            raise ValueError("FRACTIONAL_SPLIT_NOT_SUPPORTED")
        before = self.total(e["asset_id"])
        for o in self.orders.values():
            if o.asset_id == e["asset_id"] and o.status == "open":
                o.status, o.reason = "cancelled", "split_cancel_policy"
        for lot in lots:
            lot.quantity = lot.quantity * numerator // denominator
        delta = self.total(e["asset_id"]) - before
        self._shares(e, e["asset_id"], delta, e["action_id"])
        self.tables["corporate_actions"].append({"event_id": e["event_id"], "time": e["time"],
            "kind": "split", "action_id": e["action_id"], "asset_id": e["asset_id"], "delta": delta,
            "quantity_before": before, "numerator": numerator, "denominator": denominator})
        return "split_booked"

    def _mark(self, e):
        day = instant(e["time"]).date().isoformat()
        if any(row["date"] == day for row in self.tables["daily_nav"]):
            raise ValueError("DUPLICATE_DAILY_NAV")
        market_value = ZERO
        for asset in sorted({lot.asset_id for lot in self.lots if lot.quantity}):
            if asset not in e["quotes"]:
                raise ValueError("MISSING_HELD_ASSET_MARK")
            quote = e["quotes"][asset]
            observed = instant(quote["observed_at"])
            if observed > instant(e["time"]):
                raise ValueError("FUTURE_MARK")
            price = decimal(quote["price"])
            if price <= 0:
                raise ValueError("INVALID_MARK")
            stale = observed < instant(e["time"])
            if stale and quote.get("carry_reason") != "synthetic_halt_last_mark":
                raise ValueError("UNDECLARED_STALE_MARK")
            # A corporate action invalidates pre-action marks, including stale
            # halt quotes. The fixture must supply a post-action price.
            if any(r["asset_id"] == asset and r["kind"] in ("split", "dividend_record")
                   and observed < instant(r["time"]) for r in self.tables["corporate_actions"]):
                raise ValueError("PRE_ACTION_MARK")
            held, sellable = self.total(asset), self.total(asset, e["time"])
            value = money(price * held)
            market_value += value
            self.tables["positions"].append({"event_id": e["event_id"], "date": day, "asset_id": asset,
                "quantity": held, "sellable_quantity": sellable, "locked_quantity": held - sellable,
                "reserved_quantity": self.reserved_shares(asset), "mark": price, "market_value": value,
                "observed_at": quote["observed_at"], "stale": stale})
        receivable = sum((r["amount"] for r in self.receivables.values() if not r["paid"]), ZERO)
        self.tables["daily_nav"].append({"event_id": e["event_id"], "time": e["time"], "date": day,
            "cash": self.cash, "reserved_cash": self.reserved_cash, "available_cash": self.cash - self.reserved_cash,
            "market_value": market_value, "receivables": receivable, "nav": self.cash + market_value + receivable,
            "cash_ledger_count": len(self.tables["cash_ledger"]), "share_ledger_count": len(self.tables["share_ledger"]),
            "corporate_action_count": len(self.tables["corporate_actions"]),
            "risk_exposure_snapshot": None, "risk_forecast_snapshot": None, "risk_status": "unavailable"})
        return "marked"

    def export(self):
        return {**deepcopy(self.tables), "orders": [asdict(o) for o in self.orders.values()],
                "position_lots": [asdict(lot) for lot in self.lots],
                "receivables": list(deepcopy(self.receivables).values())}
