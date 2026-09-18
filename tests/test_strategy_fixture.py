import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from factor_matrix.calculation.services.accounting import AccountingEngine
from factor_matrix.calculation.services.accounting_fixture import audit_tables
from factor_matrix.calculation.services.rank_baseline import (
    ASSETS, DecisionAccount, rank_targets, signal_snapshot, validate_signals,
)
from factor_matrix.calculation.services.strategy_fixture import (
    affordable_quantity, load_stage, plain, run_strategy_fixture, validate_config,
)
from factor_matrix.cli_parser import build_parser
from factor_matrix.storage import DataLake, file_sha256

CONFIG = Path(__file__).resolve().parents[1] / "config/strategy_fixture_v1.json"


def config():
    return json.loads(CONFIG.read_text())


def run(tmp_path, cfg=None):
    path = tmp_path / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config() if cfg is None else cfg))
    lake = DataLake(tmp_path / "data")
    manifest = run_strategy_fixture(lake, config_path=path)
    root = json.loads(manifest.read_text())
    tables = {k: json.loads((lake.root / r["path"]).read_text()) for k, r in root["outputs"].items()}
    return lake, manifest, root, tables


def test_cli_exposes_versioned_default_without_strategy_search_options():
    args = build_parser().parse_args(["run-strategy-fixture"])
    assert args.config == Path("config/strategy_fixture_v1.json")


def test_fixed_three_layer_chain_uses_actual_state_and_reports_gaps(tmp_path):
    lake, path, root, tables = run(tmp_path)
    assert root["summary"] == {"sessions": 5, "strategy_variants": 1, "parameter_search": False,
        "orders": 6, "fills": 4, "commission_total": "20.00", "final_nav": "9380.00", "nonzero_quantity_gaps": 5}
    assert len(root["stages"]) == 11
    stages = [json.loads((lake.root / r["path"]).read_text()) for r in root["stages"]]
    assert [m["definition"]["layer"] for m in stages] == ["signal"] + ["strategy", "backtest"] * 5
    for index in range(3, 11, 2):
        assert stages[index - 1]["run_id"] in stages[index]["parent_run_ids"]
    target = {(r["session"], r["asset_id"]): r for r in tables["targets"]}
    gap = {(r["session"], r["asset_id"]): r for r in tables["target_actual"]}
    assert target[("2001-01-01", ASSETS[0])]["target_quantity"] == 800
    assert gap[("2001-01-01", ASSETS[0])]["actual_quantity"] == 300
    assert target[("2001-01-02", ASSETS[0])]["opening_quantity"] == 300  # Not the previous 800 target.
    assert gap[("2001-01-02", ASSETS[0])]["execution_result"] == "halted"
    assert gap[("2001-01-02", ASSETS[1])]["admission_reason"] == "cash_cap"
    assert gap[("2001-01-02", ASSETS[1])]["submitted_quantity"] == 690
    assert gap[("2001-01-03", ASSETS[1])]["order_id"] is None
    assert gap[("2001-01-03", ASSETS[0])]["execution_result"] == "price_limit_blocked"
    assert all(r["risk_status"] == "unavailable" and r["risk_forecast_snapshot"] is None for r in tables["targets"])
    final = load_stage(lake, root["final_backtest"])
    assert audit_tables(final, "10000.00")["status"] == "passed"
    assert all(r["reserved_cash"] == "0.00" for r in final["daily_nav"])
    assert final["daily_nav"][-1]["nav"] == final["daily_nav"][-2]["nav"]
    assert not list(lake.root.rglob("_CURRENT.json")) and not list(lake.root.rglob("*.sqlite"))


def test_exported_execution_events_replay_exactly_to_final_ledger(tmp_path):
    lake, _, root, tables = run(tmp_path)
    engine = AccountingEngine("10000.00", lot_size=10)
    for e in tables["executed_events"]:
        engine.apply(e)
    final = load_stage(lake, root["final_backtest"])
    for name, rows in plain(engine.export()).items():
        assert final[name] == rows
    before = plain(engine.export())
    for e in tables["executed_events"]:
        assert engine.apply(e) == "duplicate_ignored"
    assert plain(engine.export()) == before


def test_missing_and_future_signals_hold_existing_without_liquidation(tmp_path):
    lake, _, root, tables = run(tmp_path)
    strategy = load_stage(lake, root["stages"][-2])
    assert [r["reason"] for r in strategy["signal_snapshot"]] == ["not_available", "missing_value"]
    assert not strategy["intents"]
    assert all(r["reason"] == "hold_no_eligible_signal" for r in strategy["targets"])
    assert strategy["targets"][0]["target_quantity"] == 930
    assert not any(a["decision_time"].startswith("2001-01-05") for a in tables["admissions"])


def test_future_signal_values_cannot_change_any_decision(tmp_path):
    a = config()
    b = deepcopy(a)
    b["signals"][-2]["value"] = "-999999"
    _, _, _, baseline = run(tmp_path / "a", a)
    _, _, _, changed = run(tmp_path / "b", b)
    assert baseline == changed


def test_later_close_and_execution_cannot_change_earlier_targets(tmp_path):
    a, b = config(), config()
    b["sessions"][0]["opportunities"]["buy"][ASSETS[0]]["capacity"] = 200
    b["sessions"][0]["close_quotes"][ASSETS[0]]["price"] = "9.00"
    _, _, _, original = run(tmp_path / "a", a)
    _, _, _, altered = run(tmp_path / "b", b)
    assert original["targets"][:2] == altered["targets"][:2]
    assert original["intents"][0] == altered["intents"][0]
    assert original["targets"][2:4] != altered["targets"][2:4]  # New actual holdings become known next day.


def test_rank_tie_break_is_stable_and_does_not_mutate_account():
    cfg = config()
    session = cfg["sessions"][3]
    account = DecisionAccount(Decimal("10000"), Decimal(0), dict.fromkeys(ASSETS, 0))
    snapshot = signal_snapshot(cfg["signals"], session["decision_time"])
    before = deepcopy(account)
    targets, intents = rank_targets(list(reversed(snapshot)), account, session["decision_quotes"], decision_time=session["decision_time"])
    assert targets[0]["selected"] and not targets[1]["selected"]
    assert targets[0]["target_quantity"] == 1000  # 80% / 8.00
    assert account == before


@pytest.mark.parametrize("cash,requested,price,expected", [
    ("1005", 100, "10", 100), ("1004.99", 100, "10", 90),
    ("4.99", 100, "10", 0), ("30005.55", 3000, "10", 2990),
    ("30005.56", 3000, "10", 3000), ("100000", 100, "10", 100),
])
def test_buy_cap_accounts_for_minimum_and_proportional_commission(cash, requested, price, expected):
    assert affordable_quantity(requested, Decimal(price), Decimal(cash), 10) == expected


@pytest.mark.parametrize("mutation", ["labels", "duplicates", "nan", "observed_after_available", "real_asset"])
def test_signal_contract_rejects_bad_or_label_inputs(mutation):
    rows = config()["signals"]
    if mutation == "labels": rows[0]["forward_return"] = "0.20"
    if mutation == "duplicates": rows.append(deepcopy(rows[0]))
    if mutation == "nan": rows[0]["value"] = "NaN"
    if mutation == "observed_after_available": rows[0]["observed_at"] = "2001-01-01T10:00:00+00:00"
    if mutation == "real_asset": rows[0]["asset_id"] = "000001.SZ"
    with pytest.raises(ValueError):
        validate_signals(rows)


@pytest.mark.parametrize("mutation", ["future_quote", "stale_quote", "future_snapshot", "asset_axis"])
def test_strategy_checks_its_point_in_time_boundary(mutation):
    cfg = config()
    session = cfg["sessions"][0]
    snapshot = signal_snapshot(cfg["signals"], session["decision_time"])
    if mutation == "future_quote": session["decision_quotes"][ASSETS[0]]["observed_at"] = session["close_time"]
    if mutation == "stale_quote": session["decision_quotes"][ASSETS[0]]["observed_at"] = "2000-12-31T16:00:00+00:00"
    if mutation == "future_snapshot": snapshot[0]["available_at"] = session["close_time"]
    if mutation == "asset_axis": snapshot.pop()
    with pytest.raises(ValueError):
        rank_targets(snapshot, DecisionAccount(Decimal(10000), Decimal(0), dict.fromkeys(ASSETS, 0)), session["decision_quotes"], decision_time=session["decision_time"])


@pytest.mark.parametrize("mutation", ["real_purpose", "real_cost", "search", "policy", "clock", "empty", "duplicate_session", "bad_quote_clock"])
def test_runner_fails_closed_before_any_artifact_publication(tmp_path, mutation):
    cfg = config()
    if mutation == "real_purpose": cfg["purpose"] = "real_research"
    if mutation == "real_cost": cfg["real_cost_verified"] = True
    if mutation == "search": cfg["strategy"]["top_n"] = [1, 2]
    if mutation == "policy": cfg["fee_policy"]["minimum_commission_cny"] = "0"
    if mutation == "clock": cfg["sessions"][0]["buy_execution_time"] = cfg["sessions"][0]["decision_time"]
    if mutation == "empty": cfg["sessions"] = []
    if mutation == "duplicate_session": cfg["sessions"].append(cfg["sessions"][0])
    if mutation == "bad_quote_clock": cfg["sessions"][0]["opportunities"]["buy"][ASSETS[0]]["quote_time"] = cfg["sessions"][0]["close_time"]
    with pytest.raises(ValueError):
        run(tmp_path, cfg)
    assert not list((tmp_path / "data").rglob("_MANIFEST.json"))


def test_sellable_cap_preserves_locked_position(tmp_path):
    cfg = config()
    cfg["sessions"][0]["opportunities"]["buy"][ASSETS[0]]["sellable_at"] = "2001-01-04T00:00:00+00:00"
    _, _, _, tables = run(tmp_path, cfg)
    sells = [a for a in tables["admissions"] if a["side"] == "sell" and a["asset_id"] == ASSETS[0]]
    assert sells and all(a["admission_reason"] == "sellable_cap" and a["submitted_quantity"] == 0 for a in sells)
    assert tables["target_actual"][2]["actual_quantity"] == 300


def test_identical_rerun_reuses_all_artifacts_and_tamper_is_rejected(tmp_path):
    lake, path, root, _ = run(tmp_path)
    hashes = {str(p): file_sha256(p) for p in lake.root.rglob("*.json")}
    run_strategy_fixture(lake, config_path=tmp_path / "config.json")
    assert hashes == {str(p): file_sha256(p) for p in lake.root.rglob("*.json")}
    ref = root["stages"][1]
    m = json.loads((lake.root / ref["path"]).read_text())
    target = lake.root / m["outputs"]["targets"]["path"]
    target.write_text("[]")
    with pytest.raises(ValueError, match="TABLE_CHECKSUM"):
        load_stage(lake, ref)
    with pytest.raises(RuntimeError, match="IMMUTABLE_RECORD_CONFLICT"):
        run_strategy_fixture(lake, config_path=tmp_path / "config.json")
