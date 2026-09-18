import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def test_g6_freeze_protocol_never_uses_r_squared_as_membership_score() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g6_freeze_protocol_v1.json").read_text()
    )
    descriptive = protocol["descriptive_expectations_non_decisional"]
    assert descriptive["r_squared_statistic"] == (
        "mean_of_daily_full_label_weighted_r_squared"
    )
    assert descriptive["decision_role"] == (
        "diagnostic_and_investigation_trigger_not_membership_score"
    )
    assert protocol["acceptance_gates"]["factor_membership"][
        "threshold_relaxation_after_results"
    ] == "forbidden"


def test_g6_known_residual_registry_is_complete() -> None:
    registry = json.loads(
        (PROJECT / "config" / "g6_known_residual_registry_v1.json").read_text()
    )
    required = {"id", "magnitude", "evidence", "reason_not_fixed", "disposition"}
    entries = registry["entries"]
    assert len(entries) >= 4
    assert len({entry["id"] for entry in entries}) == len(entries)
    assert "raw_condition_parameterization_defect_resolved" in {
        entry["id"] for entry in entries
    }
    assert "ragged_factor_return_panel_not_supported" in {
        entry["id"] for entry in entries
    }
    for entry in entries:
        assert required <= set(entry)
        assert isinstance(entry["evidence"], str) and entry["evidence"]


@pytest.mark.local_data
def test_g6_registered_evidence_exists_locally() -> None:
    registry = json.loads((PROJECT / "config/g6_known_residual_registry_v1.json").read_text())
    for entry in registry["entries"]:
        assert (PROJECT / entry["evidence"]).exists()


def test_g6_bias_and_decision_cutoff_are_frozen() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g6_freeze_protocol_v1.json").read_text()
    )
    assert protocol["decision_data_cutoff_inclusive"] == "2025-02-28"
    bias = protocol["acceptance_gates"]["bias_test"]
    assert bias["overall_and_each_preregistered_stratum_interval"] == [0.91, 1.09]
    assert bias["independent_axes"] == [
        "predicted_specific_risk_decile", "liquidity_decile",
    ]
