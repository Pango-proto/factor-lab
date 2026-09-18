"""M2a immutable engineering runner and independent exported-table audit."""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from .accounting import AccountingEngine

POLICY = {
    "commission_rate": "0.0001854", "minimum_commission_cny": "5.00",
    "charge_aggregation": "cumulative_per_order_fixture_assumption",
    "rounding": "CNY_cent_half_up_fixture_assumption",
    "other_fees": "excluded_unverified_not_a_real_total_cost",
    "settlement": "explicit_sellable_at_per_fill_fixture_assumption",
    "split_open_orders": "cancel_fixture_assumption",
    "dividend": "record_receivable_then_pay_gross_fixture_assumption",
}


def audit_tables(tables: dict, initial_cash: str) -> dict:
    """Recompute identities without using engine state or its fee function."""
    d = Decimal
    cent = lambda x: x.quantize(d("0.01"), rounding=ROUND_HALF_UP)
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    fills, fees = tables["fills"], tables["fees"]
    cash, shares, actions = tables["cash_ledger"], tables["share_ledger"], tables["corporate_actions"]
    event_order = {r["event_id"]: i for i, r in enumerate(tables["events"])}
    check(len(event_order) == len(tables["events"]), "duplicate_event_id")
    for name in ("fills", "fees", "cash_ledger", "share_ledger", "corporate_actions", "positions", "daily_nav"):
        check(all(r["event_id"] in event_order for r in tables[name]), "orphan_event:" + name)
    if errors:
        return {"status": "failed", "errors": errors, "daily": []}
    for name in ("fills", "fees", "cash_ledger", "share_ledger", "corporate_actions", "daily_nav"):
        check(all(r["time"] == tables["events"][event_order[r["event_id"]]]["time"] for r in tables[name]), "event_time_binding:" + name)
        sequence = [event_order[r["event_id"]] for r in tables[name]]
        check(sequence == sorted(sequence), "ledger_event_order:" + name)
    check(len({r["fill_id"] for r in fills}) == len(fills), "duplicate_fill_id")
    check(len({r["order_id"] for r in tables["orders"]}) == len(tables["orders"]), "duplicate_order_id")
    for order in tables["orders"]:
        fs = [f for f in fills if f["order_id"] == order["order_id"]]
        value = sum((d(f["notional"]) for f in fs), d(0))
        expected = max(d("5.00"), cent(value * d("0.0001854"))) if fs else d(0)
        paid = sum((d(f["amount"]) for f in fees if f["order_id"] == order["order_id"]), d(0))
        check(value == d(order["notional"]) and expected == paid == d(order["commission_paid"]), "order_fee:" + order["order_id"])
        check(sum(f["quantity"] for f in fs) == order["filled_quantity"] <= order["quantity"], "order_quantity:" + order["order_id"])
        for f in fs:
            check(f["asset_id"] == order["asset_id"] and f["side"] == order["side"]
                  and f["charge_unit_id"] == order["charge_unit_id"], "fill_order_binding:" + f["fill_id"])
            check(datetime.fromisoformat(order["decision_time"]) < datetime.fromisoformat(order["earliest_execution"])
                  <= datetime.fromisoformat(f["time"]) <= datetime.fromisoformat(f["sellable_at"]), "fill_clock:" + f["fill_id"])
        cumulative_value, cumulative_paid = d(0), d(0)
        for f in fs:
            cumulative_value += d(f["notional"])
            due = max(d("5.00"), cent(cumulative_value * d("0.0001854")))
            matching = [r for r in fees if r["fill_id"] == f["fill_id"]]
            if len(matching) == 1:
                check(d(matching[0]["amount"]) == due - cumulative_paid
                      and d(matching[0]["cumulative_commission"]) == due, "incremental_fee:" + f["fill_id"])
            cumulative_paid = due
    for f in fills:
        fid = f["fill_id"]
        ff = [r for r in fees if r["fill_id"] == fid]
        cl = [r for r in cash if r["reference"] == fid and r["kind"] in ("trade", "commission")]
        sl = [r for r in shares if r["reference"] == fid and r["event_id"] == f["event_id"]]
        signed = f["quantity"] * (1 if f["side"] == "buy" else -1)
        check(d(f["notional"]) == cent(d(f["price"]) * f["quantity"]), "fill_notional:" + fid)
        check(len(ff) == 1 and len(cl) == 2 and len(sl) == 1, "fill_ledger_cardinality:" + fid)
        if len(ff) == 1 and len(cl) == 2 and len(sl) == 1:
            expected_trade = -d(f["notional"]) if f["side"] == "buy" else d(f["notional"])
            check(sum((d(r["delta"]) for r in cl if r["kind"] == "trade"), d(0)) == expected_trade,
                  "trade_cash:" + fid)
            check(sum((d(r["delta"]) for r in cl if r["kind"] == "commission"), d(0)) == -d(ff[0]["amount"]), "fee_cash:" + fid)
            check(sl[0]["delta"] == signed and sl[0]["asset_id"] == f["asset_id"], "fill_shares:" + fid)
    fill_ids = {f["fill_id"] for f in fills}
    order_ids = {o["order_id"] for o in tables["orders"]}
    check(all(f["order_id"] in order_ids for f in fills), "orphan_fill")
    check(all(f["fill_id"] in fill_ids for f in fees), "orphan_fee")
    pay_actions = [a for a in actions if a["kind"] == "dividend_pay"]
    records = {a["action_id"]: a for a in actions if a["kind"] == "dividend_record"}
    for a in actions:
        before = sum(r["delta"] for r in shares if r["asset_id"] == a["asset_id"]
                     and event_order[r["event_id"]] < event_order[a["event_id"]])
        if a["kind"] == "dividend_record":
            check(before == a["entitled_quantity"] and d(a["amount"]) == cent(d(a["cash_per_share"]) * before), "dividend_entitlement")
        if a["kind"] == "split":
            check(before == a["quantity_before"] and before * a["numerator"] == (before + a["delta"]) * a["denominator"], "split_ratio")
    check(len({a["action_id"] for a in pay_actions}) == len(pay_actions), "duplicate_dividend_payment")
    for a in pay_actions:
        movements = [r for r in cash if r["kind"] == "dividend" and r["reference"] == a["action_id"]]
        check(a["action_id"] in records and len(movements) == 1, "dividend_binding")
        if a["action_id"] in records and len(movements) == 1:
            check(d(movements[0]["delta"]) == d(a["amount"]) == d(records[a["action_id"]]["amount"]), "dividend_cash")
    balance = d(initial_cash)
    for r in cash:
        balance += d(r["delta"])
        check(balance == d(r["balance"]) and balance >= 0, "cash_running_balance")
        check((r["kind"] in ("trade", "commission") and r["reference"] in fill_ids)
              or (r["kind"] == "dividend" and any(a["action_id"] == r["reference"] for a in pay_actions)), "unexplained_cash")
    running = {}
    for r in shares:
        asset = r["asset_id"]
        running[asset] = running.get(asset, 0) + r["delta"]
        check(running[asset] == r["balance"] and running[asset] >= 0, "share_running_balance")
        check(r["reference"] in fill_ids or any(a["kind"] == "split" and a["action_id"] == r["reference"]
              and a["delta"] == r["delta"] and a["asset_id"] == asset for a in actions), "unexplained_share_change")
    for asset in set(running) | {r["asset_id"] for r in tables["position_lots"]}:
        check(running.get(asset, 0) == sum(r["quantity"] for r in tables["position_lots"] if r["asset_id"] == asset), "final_lots")
    daily = []
    for nav in tables["daily_nav"]:
        nc, ns, na = nav["cash_ledger_count"], nav["share_ledger_count"], nav["corporate_action_count"]
        check(0 <= nc <= len(cash) and 0 <= ns <= len(shares) and 0 <= na <= len(actions), "snapshot_offsets")
        check(all(n == sum(event_order[r["event_id"]] < event_order[nav["event_id"]] for r in rows)
                  for n, rows in ((nc, cash), (ns, shares), (na, actions))), "snapshot_event_boundary")
        expected_cash = d(initial_cash) + sum((d(r["delta"]) for r in cash[:nc]), d(0))
        holdings = {}
        for r in shares[:ns]:
            holdings[r["asset_id"]] = holdings.get(r["asset_id"], 0) + r["delta"]
        ps = [r for r in tables["positions"] if r["event_id"] == nav["event_id"]]
        check(len({p["asset_id"] for p in ps}) == len(ps), "duplicate_position_snapshot")
        check({p["asset_id"]: p["quantity"] for p in ps} == {a: q for a, q in holdings.items() if q}, "nav_share_identity")
        for p in ps:
            check(p["sellable_quantity"] + p["locked_quantity"] == p["quantity"]
                  and 0 <= p["reserved_quantity"] <= p["sellable_quantity"] <= p["quantity"], "position_availability")
            check(d(p["market_value"]) == cent(d(p["mark"]) * p["quantity"]), "position_value")
        mv = sum((cent(d(p["mark"]) * p["quantity"]) for p in ps), d(0))
        ar = sum((d(a["amount"]) * (1 if a["kind"] == "dividend_record" else -1)
                  for a in actions[:na] if a["kind"] in ("dividend_record", "dividend_pay")), d(0))
        cash_residual, nav_residual = d(nav["cash"]) - expected_cash, d(nav["nav"]) - expected_cash - mv - ar
        check(cash_residual == nav_residual == 0 and mv == d(nav["market_value"]) and ar == d(nav["receivables"]), "daily_accounting_identity")
        check(d(nav["available_cash"]) + d(nav["reserved_cash"]) == expected_cash
              and d(nav["available_cash"]) >= 0 and d(nav["reserved_cash"]) >= 0, "daily_cash_reservation")
        daily.append({"date": nav["date"], "cash_residual": str(cash_residual), "nav_residual": str(nav_residual)})
    check(bool(daily), "missing_daily_nav")
    check(bool(tables["events"]) and tables["events"][-1]["kind"] == "mark", "unmarked_terminal_state")
    return {"status": "passed" if not errors else "failed", "errors": errors, "daily": daily}


def run_accounting_fixture(lake: DataLake, *, config_path: Path) -> Path:
    config_hash = file_sha256(config_path)
    config = json.loads(config_path.read_text())
    if (config.get("contract_id") != "accounting_fixture_v1"
        or config.get("purpose") != "engineering_fixture_only"
        or config.get("real_cost_verified") is not False or config.get("policy") != POLICY):
        raise ValueError("ACCOUNTING_FIXTURE_CONTRACT_UNSUPPORTED")
    code_hash = source_tree_hash()
    definition = {"config_sha256": config_hash, "input_sha256": json_hash(config["events"]),
                  "code_hash": code_hash, "contract_id": config["contract_id"]}
    run_id = "accounting_fixture_v1_" + json_hash(definition)[:16]
    run_dir = lake.root / "diagnostics" / "accounting_fixture_v1" / f"run_id={run_id}"
    engine = AccountingEngine(config["initial_cash"], lot_size=config["lot_size"])
    for event in config["events"]:
        engine.apply(event)
    # Audit the serialized representation, exactly as a downstream consumer reads it.
    tables = json.loads(json.dumps(engine.export(), default=str))
    reconciliation = audit_tables(tables, config["initial_cash"])
    if reconciliation["status"] != "passed":
        raise ValueError("ACCOUNTING_RECONCILIATION_FAILED " + str(reconciliation["errors"]))
    actual = {"final_nav": tables["daily_nav"][-1]["nav"],
              "commission_total": str(sum((Decimal(r["amount"]) for r in tables["fees"]), Decimal("0.00"))),
              "fills": len(tables["fills"]), "days": len(tables["daily_nav"])}
    if actual != config["expected"]:
        raise ValueError("FIXTURE_EXPECTATION_MISMATCH " + str(actual))
    outputs = {}
    for name, value in {**tables, "reconciliation": reconciliation, "input_config": config}.items():
        path = lake.write_immutable_json(run_dir / f"{name}.json", value)
        outputs[name] = lake.artifact_record(path)
    if file_sha256(config_path) != config_hash or source_tree_hash() != code_hash:
        raise ValueError("ACCOUNTING_INPUT_OR_CODE_CHANGED")
    return lake.write_immutable_json(run_dir / "_MANIFEST.json", {
        "schema_version": 1, "run_id": run_id, "status": "engineering_passed",
        "purpose": "engineering_fixture_only", "research_status": "not_eligible",
        "real_cost_verified": False, "holdout_touched": False, "parent_run_ids": [],
        "risk_basis_id": None, "risk_set_version": None, "risk_status": "unavailable",
        "scope": {"assets": ["FIXTURE.SH", "FIXTURE.SZ"],
                  "start": config["events"][0]["time"], "end": config["events"][-1]["time"]},
        "definition": definition, "summary": actual, "outputs": outputs,
        "table_encoding": "JSON arrays; monetary values are exact Decimal strings in CNY",
        "publication_policy": "diagnostics_only_no_current_no_registry",
    })
