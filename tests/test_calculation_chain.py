"""End-to-end tests for the realized-return calculation chain."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace

import polars as pl
import pytest

from factor_matrix.calculation.components.market import (
    MarketReturnDispersion,
    MaterializeRealizedReturns,
)
from factor_matrix.calculation.components.labels import BuildForwardLabels
from factor_matrix.calculation.l2.label_policy import DEFAULT_PATH
from factor_matrix.calculation.core.contracts import (
    CalculationRequest,
    CalculationResult,
    QualityReport,
    QualityStatus,
    Stage,
    TableOutput,
)
from factor_matrix.calculation.core.executor import CalculationExecutor
from factor_matrix.calculation.core.policy import architecture_boundary_policy
from factor_matrix.calculation.core.registry import CalculationRegistry
from factor_matrix.calculation.publishing.current_pointer import CurrentPointer
from factor_matrix.calculation.publishing.local import LocalArtifactStore
from factor_matrix.storage import DataLake


START = dt.date(2026, 1, 5)
END = dt.date(2026, 1, 6)


def _universe_frame() -> pl.DataFrame:
    rows = []
    for day in (START, END):
        for asset in range(40):
            suspended = asset == 39
            rows.append({
                "trade_date": day,
                "asset_id": f"{600000 + asset}.SH",
                "board_id": "main",
                "observation_state": (
                    "SUSPENDED_CONFIRMED" if suspended else "TRADED"
                ),
                "is_suspended": suspended,
                "is_exchange_first_day": False,
            })
    return pl.DataFrame(rows)


def _silver_frame() -> pl.DataFrame:
    rows = []
    for day_offset, day in enumerate((START, END)):
        for asset in range(40):
            suspended = asset == 39
            rows.append({
                "trade_date": day,
                "asset_id": f"{600000 + asset}.SH",
                "total_return": (
                    None
                    if suspended
                    else (asset - 20) / 1000.0 + day_offset / 10000.0
                ),
                "return_source": "suspension" if suspended else "price",
                "missing_return_policy": "suspended" if suspended else "observed",
                "is_suspended": suspended,
            })
    return pl.DataFrame(rows)


def _publish_universe(lake: DataLake, frame: pl.DataFrame):
    store = LocalArtifactStore(lake)
    materializer = MaterializeRealizedReturns(lake)
    spec = replace(
        materializer.spec,
        operation_id="fixture_tradable_universe",
        stage=Stage.UNIVERSE,
        input_artifact_types=(),
        input_artifact_versions={},
        output_artifact_types=("tradable_universe",),
        output_artifact_versions={"tradable_universe": "1"},
    )
    result = CalculationResult(
        outputs=(TableOutput(
            artifact_type="tradable_universe",
            tables={"tradable_universe_v1": frame},
        ),),
        quality=QualityReport(status=QualityStatus.PASSED, checks=()),
    )
    request = CalculationRequest(
        operation_id=spec.operation_id,
        operation_version=spec.version,
        as_of_date=END,
        scope_key="market:all",
        parameters={},
        inputs=(),
    )
    return store.publish(spec, request, result)[0]


def _build(tmp_path):
    lake = DataLake(tmp_path)
    lake.replace("returns_daily", _silver_frame())
    universe = _publish_universe(lake, _universe_frame())
    store = LocalArtifactStore(lake)
    registry = CalculationRegistry((
        MaterializeRealizedReturns(lake),
        MarketReturnDispersion(),
    ))
    executor = CalculationExecutor(
        registry=registry,
        loader=store,
        publisher=store,
        policy=architecture_boundary_policy(),
    )
    return lake, executor, CurrentPointer(lake), universe


def _materialize_request(inputs, *, start="2026-01-05", end="2026-01-06"):
    return CalculationRequest(
        operation_id="materialize_realized_returns",
        operation_version="1",
        as_of_date=END,
        scope_key="market:all",
        parameters={"start_date": start, "end_date": end},
        inputs=inputs,
    )


def test_calculation_chain_publishes_and_is_consumable(tmp_path) -> None:
    lake, executor, pointer, universe = _build(tmp_path)

    produced = executor.execute(_materialize_request((universe,)))
    assert len(produced) == 1
    assert produced[0].quality_status is QualityStatus.PASSED
    pointer.publish(produced[0])

    upstream = pointer.resolve("realized_returns", "1", "market:all")
    assert upstream.key.run_id == produced[0].key.run_id

    dispersion = executor.execute(CalculationRequest(
        operation_id="market_return_dispersion",
        operation_version="1",
        as_of_date=END,
        scope_key="market:all",
        parameters={},
        inputs=(upstream,),
    ))
    assert dispersion[0].quality_status is QualityStatus.PASSED
    pointer.publish(dispersion[0])

    published = pointer.resolve("market_return_dispersion", "1", "market:all")
    table = LocalArtifactStore(lake).load(published).tables[
        "market_return_dispersion_v1"
    ]
    expected = (
        _universe_frame()
        .filter(pl.col("observation_state") == "TRADED")
        .join(_silver_frame(), on=["trade_date", "asset_id"])
        .group_by("trade_date")
        .agg(
            pl.len().alias("n_valid"),
            pl.col("total_return").std(ddof=1).alias("sigma_r"),
        )
        .sort("trade_date")
    )
    assert table["n_valid"].to_list() == expected["n_valid"].to_list()
    for observed, reference in zip(table["sigma_r"], expected["sigma_r"]):
        assert observed == pytest.approx(reference, rel=1e-12)


def test_quality_failure_reports_failed_check_and_preserves_pointer(tmp_path) -> None:
    _, executor, pointer, universe = _build(tmp_path)
    produced = executor.execute(_materialize_request((universe,)))
    pointer.publish(produced[0])
    good_run_id = pointer.resolve("realized_returns", "1", "market:all").key.run_id

    with pytest.raises(RuntimeError, match=r"CALCULATION_QUALITY_GATE_FAILED.*rows_present"):
        executor.execute(_materialize_request(
            (universe,), start="2020-01-01", end="2020-01-02"
        ))

    assert pointer.resolve("realized_returns", "1", "market:all").key.run_id == good_run_id


def test_input_contract_mismatch_is_rejected(tmp_path) -> None:
    _, executor, _, _ = _build(tmp_path)

    with pytest.raises(ValueError, match="CALCULATION_INPUT_MISMATCH"):
        executor.execute(CalculationRequest(
            operation_id="market_return_dispersion",
            operation_version="1",
            as_of_date=END,
            scope_key="market:all",
            parameters={},
            inputs=(),
        ))


@pytest.mark.parametrize("plugin_class", [MaterializeRealizedReturns, BuildForwardLabels])
def test_return_components_resolve_policy_independently_of_data_location(tmp_path, plugin_class):
    lake = DataLake(tmp_path / "independent" / "data")
    expected = json.loads(DEFAULT_PATH.read_text())
    assert plugin_class(lake)._label_policy() == expected
    local = lake.root / "config" / DEFAULT_PATH.name
    local.parent.mkdir(parents=True)
    changed = {**expected, "policy_id": "local_override"}
    local.write_text(json.dumps(changed))
    assert plugin_class(lake)._label_policy() == changed
    with pytest.raises(FileNotFoundError):
        plugin_class(lake, label_policy_path=tmp_path / "missing.json")._label_policy()
    # A malformed selected override must not fall back to the default policy.
    local.write_text("{partial")
    with pytest.raises(json.JSONDecodeError):
        plugin_class(lake)._label_policy()


def test_changed_resumption_policy_still_blocks_publication(tmp_path):
    lake, executor, pointer, universe = _build(tmp_path)
    produced = executor.execute(_materialize_request((universe,)))
    pointer.publish(produced[0])
    local = lake.root / "config" / DEFAULT_PATH.name
    local.parent.mkdir(parents=True)
    changed = json.loads(DEFAULT_PATH.read_text())
    changed["l2b"]["resumption_day"]["action"] = "include"
    local.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="resumption_policy_frozen"):
        executor.execute(_materialize_request((universe,)))
    assert pointer.resolve("realized_returns", "1", "market:all").key.run_id == produced[0].key.run_id


def test_forward_label_operation_consumes_materialized_returns(tmp_path):
    lake, executor, _, universe = _build(tmp_path)
    realized = executor.execute(_materialize_request((universe,)))[0]
    store = LocalArtifactStore(lake)
    labels_executor = CalculationExecutor(
        CalculationRegistry((BuildForwardLabels(lake),)), store, store,
        architecture_boundary_policy(),
    )
    label = labels_executor.execute(CalculationRequest(
        operation_id="build_forward_labels", operation_version="1",
        as_of_date=END, scope_key="market:all", parameters={"horizons": [1]},
        inputs=(realized, universe),
    ))[0]
    assert label.quality_status is QualityStatus.PASSED
    frame = store.load(label).tables["forward_labels_v1"]
    row = frame.filter((pl.col("asset_id") == "600000.SH") & (pl.col("trade_date") == START))
    assert row["target_return"][0] == pytest.approx(-0.02 + 0.0001)
