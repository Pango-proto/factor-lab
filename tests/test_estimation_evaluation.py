import json

import pytest

from factor_matrix.calculation.l2.estimation_evaluation import (
    EvaluationContract,
    centered_peak_gate,
    permutation_null_gate,
    placebo_zero_gate,
    rolling_mean,
    serial_autocorrelation,
    shifted_exposure_diagnostic,
    summarize_ic,
)


def test_contract_derives_purge_gap_instead_of_accepting_manual_value(tmp_path) -> None:
    path = tmp_path / "contract.json"
    payload = {
        "framework_id": "test",
        "sample": {
            "development_start": "2019-10-08",
            "holdout_start": "2025-03-01",
            "horizon_unit": "trading_days",
            "reported_horizons": [1, 5, 20],
            "primary_horizon": 5,
            "embargo_trading_days": 5,
            "purge_gap_trading_days": 24,
        },
        "negative_controls": {"seed": 7, "zero_threshold_standard_errors": 2},
        "inference": {"rolling_window_trading_days": 252},
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="PURGE_GAP_NOT_DERIVED"):
        EvaluationContract.from_json(path)


def test_ic_summary_uses_newey_west_lag_at_least_horizon() -> None:
    result = summarize_ic([0.01, 0.02, -0.01, 0.03, 0.00], horizon_days=3)
    assert result["newey_west_lag"] == 3
    assert result["mean"] == pytest.approx(0.01)
    assert result["ic_ir"] > 0


def test_placebo_gate_and_shift_diagnostic_are_explicit() -> None:
    placebo = placebo_zero_gate(
        [-0.01, 0.01, -0.02, 0.02, 0.0, 0.0],
        horizon_days=1,
        threshold_standard_errors=2,
    )
    shifted = shifted_exposure_diagnostic(
        [0.04, 0.03, 0.05, 0.04, 0.06, 0.05],
        [0.01, 0.00, 0.01, 0.00, 0.01, 0.00],
        horizon_days=1,
    )
    assert placebo["passed"] is True
    assert shifted["status"] == "passed_significant_decline"


def test_rolling_mean_has_burn_in() -> None:
    assert rolling_mean([1.0, 2.0, 3.0], window=2) == [None, 1.5, 2.5]


def test_serial_autocorrelation_reports_requested_lags() -> None:
    result = serial_autocorrelation([1.0, -1.0, 1.0, -1.0, 1.0], lags=[1, 2, 10])
    assert result["1"] == pytest.approx(-1.0)
    assert result["2"] == pytest.approx(1.0)
    assert result["10"] is None


def test_alignment_peak_and_permutation_distribution_gates_are_falsifiable() -> None:
    assert centered_peak_gate({-2: 0.01, -1: 0.02, 0: 0.03, 1: 0.02, 2: 0.01})[
        "passed"
    ]
    assert not centered_peak_gate({-2: 0.03, -1: 0.02, 0: 0.01, 1: 0.0, 2: -0.01})[
        "passed"
    ]
    spanning = permutation_null_gate([value / 10_000 for value in range(-10, 10)])
    positive = permutation_null_gate([value / 10_000 for value in range(1, 21)])
    assert spanning["passed"]
    assert not positive["passed"]
