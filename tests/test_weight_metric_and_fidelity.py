import json
from pathlib import Path

import pytest

from factor_matrix.calculation.l2 import (
    NeutralizationFidelityPoint, PredictivityPoint, SemanticProbeLedger,
    neutralization_fidelity, weighted_residualize_against_frozen_composite,
)
from factor_matrix.weight_metric_contract import load_weight_metric_contract


PROJECT = Path(__file__).resolve().parents[1]


def point(index: int, value: float, *, trained: bool = True) -> PredictivityPoint:
    return PredictivityPoint(
        evaluation_index=index,
        training_end_index=index - 1 if trained else None,
        raw_ic=value,
        shrunk_ic=value,
        standard_error=0.01,
        confidence_low=value - 0.02,
        confidence_high=value + 0.02,
        factor_weight=1.0,
        status_signal="active" if trained else "burn_in",
    )


def test_weight_metric_contract_binds_two_semantics_to_one_artifact() -> None:
    contract = load_weight_metric_contract(
        PROJECT / "config" / "weight_metric_contract_v1.json"
    )
    assert contract.is_bound
    assert contract.regression_base_scheme_id == contract.orthogonalization_scheme_id
    assert contract.risk_metric_id.startswith("risk_metric_")


@pytest.mark.local_data
def test_weight_metric_bound_local_artifact_matches_hash() -> None:
    contract = load_weight_metric_contract(PROJECT / "config/weight_metric_contract_v1.json")
    assert contract.resolve_artifact(PROJECT / "data").is_file()


def test_neutralization_fidelity_handles_floor_sign_and_retention() -> None:
    rows = neutralization_fidelity(
        (point(0, 0.04), point(1, 0.002), point(2, 0.04)),
        (point(0, 0.02), point(1, 0.001), point(2, -0.02)),
        minimum_absolute_raw_ic=0.005,
        minimum_retention_ratio=0.30,
    )
    assert isinstance(rows[0], NeutralizationFidelityPoint)
    assert rows[0].status == "passed"
    assert rows[0].absolute_ic_retention_ratio == pytest.approx(0.5)
    assert rows[1].status == "raw_ic_below_ratio_floor"
    assert rows[2].status == "degraded_neutralization_fidelity"


def test_semantic_probe_ledger_is_append_only_and_separate(tmp_path: Path) -> None:
    ledger = SemanticProbeLedger(tmp_path / "metadata.sqlite")
    kwargs = dict(
        event_id="probe-1", risk_basis_id="basis-1",
        descriptor_manifest_sha="a" * 64, protocol_sha="b" * 64,
        created_at="2026-08-16T00:00:00Z",
    )
    ledger.record(**kwargs)
    assert ledger.count() == 1
    assert ledger.count_for_basis("basis-1") == 1
    assert ledger.count_for_basis("basis-2") == 0
    with pytest.raises(RuntimeError, match="APPEND_ONLY_CONFLICT"):
        ledger.record(**kwargs)


def test_incremental_exposure_is_orthogonal_to_frozen_composite() -> None:
    composite = (-2.0, -1.0, 0.0, 1.0, 2.0)
    candidate = (-3.0, -1.0, 1.0, 2.0, 6.0)
    weights = (1.0, 2.0, 1.0, 2.0, 1.0)
    residual = weighted_residualize_against_frozen_composite(
        candidate, composite, weights,
    )
    assert sum(w * value for w, value in zip(weights, residual)) == pytest.approx(0.0)
    assert sum(
        w * x * value for w, x, value in zip(weights, composite, residual)
    ) == pytest.approx(0.0)


def test_g6a_requires_weight_metric_coherence() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g6_freeze_protocol_v1.json").read_text()
    )
    prerequisites = " ".join(
        protocol["split_gates"]["G6a_basis_freeze"]["hard_prerequisites"]
    )
    assert "weight_metric_contract" in prerequisites
    assert "independently recomputes orthogonality" in prerequisites
