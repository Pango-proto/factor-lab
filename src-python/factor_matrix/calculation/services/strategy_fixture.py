"""Three-layer engineering runner. Frozen intents feed the M2a account engine."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from .accounting import AccountingEngine, commission, decimal, instant, money
from .accounting_fixture import POLICY, audit_tables
from .rank_baseline import ASSETS, STRATEGY, DecisionAccount, rank_targets, signal_snapshot, validate_signals


def plain(value):
    return json.loads(json.dumps(value, default=str))


def affordable_quantity(requested: int, price: Decimal, cash: Decimal, lot_size: int) -> int:
    """Largest whole-lot order covered by ACTUAL available cash plus commission."""
    lo, hi = 0, requested // lot_size
    while lo < hi:
        mid = (lo + hi + 1) // 2
        value = money(mid * lot_size * price)
        if value + commission(value) <= cash:
            lo = mid
        else:
            hi = mid - 1
    return lo * lot_size


def publish_stage(lake, directory, *, layer, definition, parents, tables):
    stage_definition = {**definition, "layer": layer, "parents": parents}
    run_id = f"m2b_{layer}_" + json_hash(stage_definition)[:16]
    stage_dir = directory / run_id
    outputs = {name: lake.artifact_record(lake.write_immutable_json(stage_dir / f"{name}.json", plain(rows)))
               for name, rows in tables.items()}
    path = lake.write_immutable_json(stage_dir / "_MANIFEST.json", {
        "schema_version": 1, "run_id": run_id, "status": "engineering_passed",
        "purpose": "engineering_fixture_only", "research_status": "not_eligible",
        "definition": stage_definition, "parent_run_ids": [p["run_id"] for p in parents],
        "risk_basis_id": None, "risk_set_version": None, "risk_status": "unavailable",
        "outputs": outputs,
    })
    return {"run_id": run_id, **lake.artifact_record(path)}


def load_stage(lake, ref):
    path = lake.root / ref["path"]
    if file_sha256(path) != ref["sha256"]:
        raise ValueError("M2B_MANIFEST_CHECKSUM_MISMATCH")
    manifest = json.loads(path.read_text())
    if (manifest["run_id"] != ref["run_id"] or manifest["status"] != "engineering_passed"
        or manifest["purpose"] != "engineering_fixture_only" or manifest["research_status"] != "not_eligible"):
        raise ValueError("M2B_STAGE_NOT_ELIGIBLE")
    tables = {}
    for name, record in manifest["outputs"].items():
        path = lake.root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("M2B_TABLE_CHECKSUM_MISMATCH")
        tables[name] = json.loads(path.read_text())
    return tables


def validate_config(config):
    if (set(config) != {"contract_id", "purpose", "real_cost_verified", "initial_cash", "strategy", "fee_policy", "signals", "sessions"}
        or config["contract_id"] != "strategy_fixture_v1" or config["purpose"] != "engineering_fixture_only"
        or config["real_cost_verified"] is not False or config["strategy"] != STRATEGY or config["fee_policy"] != POLICY
        or money(config["initial_cash"]) <= 0 or decimal(config["initial_cash"]) != money(config["initial_cash"])):
        raise ValueError("M2B_FIXTURE_CONTRACT_UNSUPPORTED")
    validate_signals(config["signals"])
    if not config["sessions"]:
        raise ValueError("MISSING_SESSIONS")
    previous = None
    session_ids = set()
    for s in config["sessions"]:
        if set(s) != {"session", "decision_time", "sell_submit_time", "sell_execution_time", "buy_submit_time",
                      "buy_execution_time", "cancel_time", "close_time", "decision_quotes", "opportunities", "close_quotes"}:
            raise ValueError("M2B_SESSION_SCHEMA_UNSUPPORTED")
        clocks = [instant(s[k]) for k in ("decision_time", "sell_submit_time", "sell_execution_time",
                  "buy_submit_time", "buy_execution_time", "cancel_time", "close_time")]
        if (any(a >= b for a, b in zip(clocks, clocks[1:]))
            or any(t.date().isoformat() != s["session"] for t in clocks)
            or (previous is not None and clocks[0] <= previous) or s["session"] in session_ids):
            raise ValueError("M2B_SESSION_CLOCK_INVALID")
        previous = clocks[-1]
        session_ids.add(s["session"])
        if set(s["opportunities"]) != {"buy", "sell"} or set(s["close_quotes"]) != set(ASSETS):
            raise ValueError("M2B_MARKET_AXIS_MISMATCH")
        for side in ("sell", "buy"):
            if set(s["opportunities"][side]) != set(ASSETS):
                raise ValueError("M2B_MARKET_AXIS_MISMATCH")
            for q in s["opportunities"][side].values():
                if (set(q) != {"quote_time", "price", "capacity", "halted", "side_blocked", "sellable_at"}
                    or q["quote_time"] != s[f"{side}_execution_time"]
                    or instant(q["sellable_at"]) < instant(q["quote_time"])
                    or decimal(q["price"]) <= 0 or decimal(q["price"]) != money(q["price"])
                    or type(q["capacity"]) is not int or q["capacity"] < 0
                    or type(q["halted"]) is not bool or type(q["side_blocked"]) is not bool):
                    raise ValueError("M2B_OPPORTUNITY_INVALID")
        for q in s["close_quotes"].values():
            if (set(q) != {"price", "observed_at"} or q["observed_at"] != s["close_time"] or decimal(q["price"]) <= 0):
                raise ValueError("M2B_CLOSE_QUOTE_INVALID")
        # Validate the narrowed strategy input contract before publishing any
        # stage. These dry-run targets are discarded; live targets use prior
        # published actual holdings and cash, never this initial account.
        rank_targets(signal_snapshot(config["signals"], s["decision_time"]),
                     DecisionAccount(decimal(config["initial_cash"]), Decimal(0), {a: 0 for a in ASSETS}),
                     s["decision_quotes"], decision_time=s["decision_time"])
    if any(r["session"] not in session_ids for r in config["signals"]):
        raise ValueError("M2B_SIGNAL_SESSION_OUTSIDE_SCOPE")


def execute_intents(engine, intents, session):
    """Buy cash-commit decision follows sell fills; original targets stay frozen."""
    admissions, executed = [], []

    def apply(e):
        outcome = engine.apply(e)
        executed.append(e)
        return outcome

    for side in ("sell", "buy"):
        pending = []
        for intent in sorted((i for i in intents if i["side"] == side), key=lambda i: i["asset_id"]):
            asset, requested = intent["asset_id"], intent["requested_quantity"]
            price = decimal(intent["limit_price"])
            submit_time = session[f"{side}_submit_time"]
            available_cash = engine.cash - engine.reserved_cash
            sellable = engine.total(asset, submit_time) - engine.reserved_shares(asset)
            admitted = (min(requested, sellable) // engine.lot_size * engine.lot_size if side == "sell"
                        else affordable_quantity(requested, price, available_cash, engine.lot_size))
            reason = "full_intent" if admitted == requested else "sellable_cap" if side == "sell" else "cash_cap"
            oid = "order:" + intent["intent_id"] if admitted else None
            admission = {**intent, "order_id": oid, "submission_time": submit_time,
                         "available_cash_at_submission": available_cash, "sellable_at_submission": sellable,
                         "submitted_quantity": admitted, "admission_reason": reason, "execution_result": "not_submitted"}
            admissions.append(admission)
            if oid:
                result = apply({"event_id": "submit:" + oid, "kind": "submit", "time": submit_time,
                    "order_id": oid, "asset_id": asset, "side": side, "quantity": admitted,
                    "limit_price": str(price), "decision_time": submit_time,
                    "earliest_execution": session[f"{side}_execution_time"]})
                admission["submission_result"] = result
                pending.append(admission)
        for a in pending:
            oid = a["order_id"]
            quote = session["opportunities"][side][a["asset_id"]]
            a["execution_result"] = apply({"event_id": "execute:" + oid, "kind": "execute",
                "time": session[f"{side}_execution_time"], "order_id": oid, "fill_id": "fill:" + oid, **quote})
    for a in admissions:
        oid = a["order_id"]
        if oid and engine.orders[oid].status == "open":
            apply({"event_id": "cancel:" + oid, "kind": "cancel", "time": session["cancel_time"], "order_id": oid})
    apply({"event_id": "mark:" + session["session"], "kind": "mark", "time": session["close_time"], "quotes": session["close_quotes"]})
    return admissions, executed


def compare_targets(targets, admissions, ledger, session):
    """Derive gaps from actual fills/closing positions, never from target weights."""
    positions = {r["asset_id"]: r for r in ledger["positions"] if r["date"] == session}
    final_nav = decimal(ledger["daily_nav"][-1]["nav"])
    output = []
    for t in targets:
        a = next((a for a in admissions if a["target_id"] == t["target_id"]), None)
        fs = [f for f in ledger["fills"] if a and f["order_id"] == a["order_id"]]
        p = positions.get(t["asset_id"])
        actual = p["quantity"] if p else 0
        signed = sum(f["quantity"] * (1 if f["side"] == "buy" else -1) for f in fs)
        if t["opening_quantity"] + signed != actual:
            raise ValueError("M2B_TARGET_ACTUAL_SHARE_RECONCILIATION_FAILED")
        output.append({"target_id": t["target_id"], "session": session, "asset_id": t["asset_id"],
            "target_quantity": t["target_quantity"], "actual_quantity": actual,
            "quantity_gap": actual - t["target_quantity"], "filled_quantity": sum(f["quantity"] for f in fs),
            "requested_quantity": a["requested_quantity"] if a else 0,
            "submitted_quantity": a["submitted_quantity"] if a else 0,
            "order_id": a["order_id"] if a else None,
            "admission_reason": a["admission_reason"] if a else "no_trade_intent",
            "execution_result": a["execution_result"] if a else "no_trade_intent",
            "target_weight_at_decision": t["target_weight"],
            "actual_weight_at_close": decimal(p["market_value"]) / final_nav if p and final_nav else Decimal(0)})
    return output


def run_strategy_fixture(lake: DataLake, *, config_path: Path) -> Path:
    config_hash, code_hash = file_sha256(config_path), source_tree_hash()
    config = json.loads(config_path.read_text())
    validate_config(config)
    definition = {"contract_id": config["contract_id"], "config_sha256": config_hash,
                  "code_hash": code_hash, "strategy_id": STRATEGY["strategy_id"],
                  "scope": {"assets": list(ASSETS), "start": config["sessions"][0]["session"], "end": config["sessions"][-1]["session"]}}
    run_id = "strategy_fixture_v1_" + json_hash(definition)[:16]
    directory = lake.root / "diagnostics" / "strategy_fixture_v1" / f"run_id={run_id}"
    source = publish_stage(lake, directory, layer="signal", definition=definition, parents=[],
                           tables={"signals": config["signals"], "input_config": config})
    input_tables = load_stage(lake, source)
    engine = AccountingEngine(config["initial_cash"], lot_size=10)
    previous, refs = None, [source]
    all_targets, all_intents, all_admissions, all_gaps, all_events = [], [], [], [], []
    for session in input_tables["input_config"]["sessions"]:
        if previous:
            prior = load_stage(lake, previous)
            account = DecisionAccount(decimal(prior["daily_nav"][-1]["cash"]), decimal(prior["daily_nav"][-1]["receivables"]),
                {a: sum(lot["quantity"] for lot in prior["position_lots"] if lot["asset_id"] == a) for a in ASSETS})
        else:
            account = DecisionAccount(decimal(config["initial_cash"]), Decimal(0), {a: 0 for a in ASSETS})
        snapshot = signal_snapshot(input_tables["signals"], session["decision_time"])
        targets, intents = rank_targets(snapshot, account, session["decision_quotes"], decision_time=session["decision_time"])
        strategy = publish_stage(lake, directory, layer="strategy", definition={**definition, "session": session["session"]},
            parents=[source] + ([previous] if previous else []), tables={"signal_snapshot": snapshot,
                "decision_account": {"cash": account.cash, "receivables": account.receivables, "holdings": account.holdings},
                "targets": targets, "intents": intents})
        refs.append(strategy)
        planned = load_stage(lake, strategy)
        admissions, events = execute_intents(engine, planned["intents"], session)
        ledger = plain(engine.export())
        reconciliation = audit_tables(ledger, config["initial_cash"])
        if reconciliation["status"] != "passed":
            raise ValueError("M2B_ACCOUNTING_RECONCILIATION_FAILED " + str(reconciliation["errors"]))
        gaps = compare_targets(planned["targets"], admissions, ledger, session["session"])
        all_targets.extend(planned["targets"])
        all_intents.extend(planned["intents"])
        all_admissions.extend(admissions)
        all_gaps.extend(gaps)
        all_events.extend(events)
        previous = publish_stage(lake, directory, layer="backtest", definition={**definition, "session": session["session"]},
            parents=[strategy] + ([previous] if previous else []), tables={**ledger, "admissions": admissions,
                "target_actual": gaps, "executed_events": events, "reconciliation": reconciliation})
        refs.append(previous)
    final = load_stage(lake, previous)
    summary = {"sessions": len(config["sessions"]), "strategy_variants": 1, "parameter_search": False,
               "orders": len(final["orders"]), "fills": len(final["fills"]),
               "commission_total": str(sum((decimal(f["amount"]) for f in final["fees"]), Decimal("0.00"))),
               "final_nav": final["daily_nav"][-1]["nav"], "nonzero_quantity_gaps": sum(g["quantity_gap"] != 0 for g in all_gaps)}
    outputs = {name: lake.artifact_record(lake.write_immutable_json(directory / f"{name}.json", plain(rows)))
               for name, rows in {"targets": all_targets, "intents": all_intents, "admissions": all_admissions,
                                  "target_actual": all_gaps, "executed_events": all_events}.items()}
    # Re-verify all explicitly bound stage outputs before publication.
    for ref in refs:
        load_stage(lake, ref)
    if file_sha256(config_path) != config_hash or source_tree_hash() != code_hash:
        raise ValueError("M2B_INPUT_OR_CODE_CHANGED")
    return lake.write_immutable_json(directory / "_MANIFEST.json", {
        "schema_version": 1, "run_id": run_id, "status": "engineering_passed",
        "purpose": "engineering_fixture_only", "research_status": "not_eligible", "real_cost_verified": False,
        "holdout_touched": False, "risk_status": "unavailable", "risk_basis_id": None, "risk_set_version": None,
        "definition": definition, "parent_run_ids": [r["run_id"] for r in refs], "stages": refs,
        "final_backtest": previous, "outputs": outputs, "summary": summary,
        "publication_policy": "diagnostics_only_no_current_no_registry",
    })
