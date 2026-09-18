import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from factor_matrix.calculation.services.accounting import AccountingEngine, commission
from factor_matrix.calculation.services.accounting_fixture import audit_tables, run_accounting_fixture
from factor_matrix.storage import DataLake, file_sha256

CONFIG = Path(__file__).resolve().parents[1] / "config/accounting_fixture_v1.json"


def stamp(day=1, hour=10):
    return f"2001-01-{day:02}T{hour:02}:00:00+00:00"


def submit(oid="o1", *, day=1, hour=10, qty=100, price="10.00", side="buy"):
    return dict(event_id="submit_" + oid, kind="submit", time=stamp(day, hour), order_id=oid,
                asset_id="FIXTURE.SH", side=side, quantity=qty, limit_price=price,
                decision_time=stamp(day, hour), earliest_execution=stamp(day, hour+1))


def fill(fid="f1", oid="o1", *, day=1, hour=11, qty=100, price="10.00", **kw):
    return dict(event_id="execute_" + fid, kind="execute", time=stamp(day, hour), order_id=oid,
                fill_id=fid, quote_time=stamp(day, hour), price=price, capacity=qty,
                halted=False, side_blocked=False, sellable_at=stamp(day+1, 0), **kw)


def event(kind, eid, day=1, hour=12, **kw):
    return dict(kind=kind, event_id=eid, time=stamp(day, hour), **kw)


def fixture_engine():
    cfg = json.loads(CONFIG.read_text())
    engine = AccountingEngine(cfg["initial_cash"], lot_size=cfg["lot_size"])
    for e in cfg["events"]:
        engine.apply(e)
    return engine


def serialized(engine):
    return json.loads(json.dumps(engine.export(), default=str))


@pytest.mark.parametrize("value,expected", [("0", "0"), ("1", "5"), ("10000", "5"), ("30000", "5.56"), ("100000", "18.54")])
def test_confirmed_commission_rate_and_minimum(value, expected):
    assert commission(Decimal(value)) == Decimal(expected)


def test_partial_fills_reserve_and_cumulative_fee():
    engine = AccountingEngine("40000", lot_size=10)
    assert engine.apply(submit(qty=3000)) == "accepted"
    assert engine.reserved_cash == Decimal("30005.56")
    engine.apply(fill(qty=1000))
    assert engine.cash == Decimal("29995")
    assert engine.reserved_cash == Decimal("20000.56")
    engine.apply(fill("f2", hour=12, qty=2000))
    assert [r["amount"] for r in engine.tables["fees"]] == [Decimal("5"), Decimal("0.56")]
    assert engine.cash == Decimal("9994.44")
    assert engine.reserved_cash == 0


def test_partial_cancel_retains_only_actual_fee_and_releases_cash():
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit(qty=500))
    engine.apply(fill(qty=100))
    engine.apply(event("cancel", "cancel", order_id="o1"))
    assert engine.cash == Decimal("4995") and engine.reserved_cash == 0
    assert engine.total("FIXTURE.SH") == 100
    assert engine.orders["o1"].commission_paid == 5


def test_zero_fills_have_no_fees_and_no_cash_movement():
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    assert engine.apply(fill(qty=0)) == "no_capacity"
    engine.apply(event("cancel", "cancel", order_id="o1"))
    assert engine.cash == 6000 and not engine.tables["fees"] and not engine.tables["cash_ledger"]


def test_duplicate_event_noop_conflict_and_failed_event_are_atomic():
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    e = fill()
    engine.apply(e)
    before = deepcopy(engine.__dict__)
    assert engine.apply(e) == "duplicate_ignored"
    assert engine.__dict__ == before
    for bad in ({**e, "capacity": 50}, {**e, "event_id": "another"},
                event("mark", "bad_mark", quotes={})):
        with pytest.raises(ValueError):
            engine.apply(bad)
        assert engine.__dict__ == before


@pytest.mark.parametrize("mutation,message", [
    ({"time": stamp(1, 9)}, "EVENT_TIME_REVERSED"),
    ({"quote_time": stamp(1, 12)}, "CONTEMPORANEOUS"),
    ({"sellable_at": stamp(1, 9)}, "SETTLEMENT_BEFORE_FILL"),
    ({"price": "NaN"}, "NONFINITE"),
    ({"price": "10.001"}, "CENT_PRICE"),
    ({"price": 10.0}, "DECIMAL_STRING"),
    ({"capacity": 10.5}, "QUANTITY"),
    ({"halted": "false"}, "MARKET_STATE"),
])
def test_invalid_execution_fails_without_mutation(mutation, message):
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    before = deepcopy(engine.__dict__)
    with pytest.raises(ValueError, match=message):
        engine.apply({**fill(), **mutation})
    assert engine.__dict__ == before


def test_cash_and_shares_cannot_be_reserved_twice():
    engine = AccountingEngine("2000", lot_size=10)
    engine.apply(submit())
    assert engine.apply(submit("o2")) == "insufficient_cash"
    engine.apply(fill())
    assert engine.apply(submit("s0", hour=12, side="sell")) == "insufficient_sellable"
    assert engine.apply(submit("s1", day=2, side="sell")) == "accepted"
    assert engine.apply(submit("s2", day=2, side="sell")) == "insufficient_sellable"
    assert engine.total("FIXTURE.SH") == 100


def test_sale_fee_cannot_make_available_cash_negative():
    engine = AccountingEngine("6", lot_size=1)
    engine.apply(submit(qty=1, price="1.00"))
    engine.apply(fill(qty=1, price="1.00"))
    engine.apply(submit("s", day=2, qty=1, price="1.00", side="sell"))
    assert engine.apply(fill("sale", "s", day=2, qty=1, price="1.00")) == "insufficient_cash_at_fill"
    assert engine.total("FIXTURE.SH") == 1 and engine.cash == 0


def test_dividend_uses_record_entitlement_after_sale_and_split():
    engine = fixture_engine()
    record = engine.receivables["div_sh"]
    assert record["entitled_quantity"] == 500 and record["amount"] == 100 and record["paid"]
    assert engine.total("FIXTURE.SH") == 800
    nav = engine.tables["daily_nav"]
    assert nav[1]["receivables"] == 100 and nav[2]["receivables"] == 0
    assert nav[1]["nav"] == nav[2]["nav"]  # Split/payment alone don't create wealth.
    assert nav[1]["reserved_cash"] == 905 and nav[2]["reserved_cash"] == 0
    assert engine.orders["split_cancel"].reason == "split_cancel_policy"


def test_halt_keeps_position_and_resumption_loss_is_not_cash_flow():
    engine = fixture_engine()
    nav = engine.tables["daily_nav"]
    assert nav[3]["cash"] == nav[4]["cash"]
    assert nav[4]["nav"] - nav[3]["nav"] == Decimal("-1520")
    halted = [r for r in engine.tables["positions"] if r["date"] == "2001-01-04" and r["asset_id"] == "FIXTURE.SH"][0]
    assert halted["quantity"] == 800 and halted["stale"]
    assert all(r["risk_status"] == "unavailable" and r["risk_forecast_snapshot"] is None for r in nav)


@pytest.mark.parametrize("quote,message", [
    ({"price": "10", "observed_at": stamp(2)}, "FUTURE_MARK"),
    ({"price": "10", "observed_at": stamp(1, 10)}, "STALE_MARK"),
    ({"price": "NaN", "observed_at": stamp(1, 12)}, "NONFINITE"),
])
def test_marks_cannot_use_future_or_implicit_stale_prices(quote, message):
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    engine.apply(fill())
    with pytest.raises(ValueError, match=message):
        engine.apply(event("mark", "m", quotes={"FIXTURE.SH": quote}))
    assert not engine.tables["daily_nav"] and not engine.tables["positions"]


def test_fractional_split_rejected_without_cancelling_order():
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    engine.apply(fill())
    engine.apply(submit("s", day=2, side="sell"))
    before = deepcopy(engine.__dict__)
    with pytest.raises(ValueError, match="FRACTIONAL_SPLIT"):
        engine.apply(event("split", "split", day=2, action_id="sp", asset_id="FIXTURE.SH", numerator=1, denominator=3))
    assert engine.__dict__ == before


def test_split_preserves_lock_and_invalidates_old_mark():
    engine = AccountingEngine("6000", lot_size=10)
    engine.apply(submit())
    engine.apply(fill())
    engine.apply(event("split", "split", action_id="sp", asset_id="FIXTURE.SH", numerator=2, denominator=1))
    assert engine.total("FIXTURE.SH") == 200 and engine.total("FIXTURE.SH", stamp(1, 13)) == 0
    with pytest.raises(ValueError, match="PRE_ACTION_MARK"):
        engine.apply(event("mark", "m", hour=13, quotes={"FIXTURE.SH": dict(price="10", observed_at=stamp(1, 11), carry_reason="synthetic_halt_last_mark")}))


def test_fixture_covers_execution_reasons_and_independent_reconciliation():
    engine = fixture_engine()
    reasons = {r["result"] for r in engine.tables["events"]}
    assert {"accepted", "filled", "partial_fill", "insufficient_cash", "quantity_constraint",
            "insufficient_sellable", "before_earliest_execution", "halted", "price_limit_blocked",
            "outside_order_limit", "no_capacity", "cancelled"} <= reasons
    report = audit_tables(serialized(engine), "50000.00")
    assert report["status"] == "passed" and len(report["daily"]) == 5


@pytest.mark.parametrize("table,field,value", [
    ("fees", "amount", "6"), ("cash_ledger", "delta", "-999"),
    ("share_ledger", "delta", 110), ("daily_nav", "nav", "50000"),
    ("daily_nav", "cash_ledger_count", 0), ("positions", "quantity", 99),
    ("corporate_actions", "entitled_quantity", 400),
    ("fills", "order_id", "unknown"),
])
def test_independent_audit_detects_table_corruption(table, field, value):
    tables = serialized(fixture_engine())
    tables[table][0][field] = value
    assert audit_tables(tables, "50000.00")["status"] == "failed"


def test_immutable_run_reuses_same_manifest_and_detects_tamper(tmp_path):
    lake = DataLake(tmp_path)
    path = run_accounting_fixture(lake, config_path=CONFIG)
    digest = file_sha256(path)
    assert run_accounting_fixture(lake, config_path=CONFIG) == path and file_sha256(path) == digest
    manifest = json.loads(path.read_text())
    assert manifest["summary"] == dict(final_nav="45464.44", commission_total="15.56", fills=5, days=5)
    assert manifest["research_status"] == "not_eligible" and not manifest["real_cost_verified"]
    assert not list(tmp_path.rglob("_CURRENT.json")) and not list(tmp_path.rglob("*.sqlite"))
    for record in manifest["outputs"].values():
        assert file_sha256(tmp_path / record["path"]) == record["sha256"]
    (path.parent / "fees.json").write_text("[]")
    with pytest.raises(RuntimeError, match="IMMUTABLE_RECORD_CONFLICT"):
        run_accounting_fixture(lake, config_path=CONFIG)


@pytest.mark.parametrize("key,value", [("purpose", "real_research"), ("real_cost_verified", True), ("policy", {})])
def test_runner_rejects_unverified_research_or_fee_policy(tmp_path, key, value):
    cfg = json.loads(CONFIG.read_text())
    cfg[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="CONTRACT_UNSUPPORTED"):
        run_accounting_fixture(DataLake(tmp_path / "data"), config_path=path)
