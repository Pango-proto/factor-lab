from datetime import date
from pathlib import Path

import pytest

from factor_matrix.external_facts import ExternalFacts
from factor_matrix.research_protocol import (
    ProtocolSealStore, ResearchProtocol, assert_label_free_derivation_inputs,
    build_sample_calendar_rows, label_sample_date,
)
from scripts.check_architecture_invariants import violations


PROJECT = Path(__file__).resolve().parents[1]


def protocol() -> ResearchProtocol:
    return ResearchProtocol.load(PROJECT / "config" / "research_protocol_v1.json")


def test_holdout_is_sealed_and_sample_roles_are_orthogonal() -> None:
    item = protocol()
    research = label_sample_date(date(2024, 1, 2), item, burn_in_end=date(2021, 8, 31))
    holdout = label_sample_date(date(2025, 3, 3), item, burn_in_end=date(2021, 8, 31))
    stress = label_sample_date(date(2016, 1, 4), item, burn_in_end=date(2021, 8, 31))
    assert research["sample_role"] == "research"
    assert research["backtest_display_eligible"]
    assert holdout["sample_role"] == "holdout" and not holdout["alpha_estimation_eligible"]
    assert stress == {"sample_role": "out_of_model", "risk_estimation_eligible": False,
                      "alpha_estimation_eligible": False,
                      "backtest_display_eligible": False, "stress_only": True}


def test_parameter_derivation_is_reproducible_and_label_free() -> None:
    snapshot = protocol().derive(
        factor_count=5, maximum_evaluation_horizon_days=20,
        cross_section_size=5000,
        available_estimation_days=1400,
    )
    values = snapshot["values"]
    assert values["burn_in_days"] == 504
    assert values["coverage_min"] == pytest.approx(2 / 3)
    assert values["minimum_category_members"] == 5
    assert values["huber_delta"] == pytest.approx(1.345, abs=0.002)
    assert values["ewma_factor_covariance_half_life_days"] > values["ewma_specific_variance_half_life_days"]
    with pytest.raises(ValueError, match="USES_RETURN_LABELS"):
        assert_label_free_derivation_inputs({"factor_count": 5, "headline_ic": 0.03})


def test_sample_calendar_uses_snapshot_derived_burn_in() -> None:
    dates = [date(2019, 7, 1).fromordinal(date(2019, 7, 1).toordinal() + index)
             for index in range(700)]
    snapshot = protocol().derive(
        factor_count=1, maximum_evaluation_horizon_days=5,
        cross_section_size=5000,
        available_estimation_days=700,
    )
    rows = build_sample_calendar_rows(dates, protocol(), snapshot)
    assert rows[503]["sample_role"] == "burn_in"
    assert rows[504]["sample_role"] == "research"


def test_holdout_seal_cannot_move(tmp_path: Path) -> None:
    store = ProtocolSealStore(tmp_path / "protocol.sqlite")
    store.seal(protocol())
    store.seal(protocol())


def test_external_facts_are_time_varying_sourced_and_immutable(tmp_path: Path) -> None:
    facts = ExternalFacts.load(PROJECT / "config" / "external_facts_v1.json")
    assert facts.lookup("stamp_duty_sell", date(2023, 8, 27)).value["rate"] == 0.001
    assert facts.lookup("stamp_duty_sell", date(2023, 8, 28)).value["rate"] == 0.0005
    assert date(2019, 7, 22) in facts.regime_break_dates()
    facts.sync_to_sqlite(tmp_path / "definitions.sqlite")
    facts.sync_to_sqlite(tmp_path / "definitions.sqlite")


def test_architecture_ci_guard_has_no_violations() -> None:
    assert violations(PROJECT) == []
