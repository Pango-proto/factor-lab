import json
from pathlib import Path

import numpy as np

from factor_matrix.calculation.l2.g4_spectrum import _principal_angle_degrees
from factor_matrix.calculation.l2.g4_diagnostics import (
    INCREMENTAL_R2_IMPLEMENTATION_ID, run_g4_attribution_diagnostics,
)
from factor_matrix.calculation.l2.risk_modeling import (
    SPECIFIC_VARIANCE_IMPLEMENTATION_ID, validate_specific_variance_config,
)
from factor_matrix.calculation.l2.g4_weight_selection import run_g4_wls_weight_selection
from factor_matrix.storage import DataLake
import pytest


PROJECT = Path(__file__).resolve().parents[1]


def test_statistical_factor_stability_is_preregistered_as_subspace_not_columns() -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    policy = payload["g4_1_statistical_factor_protocol"]
    assert policy["preregistered"] is True
    assert policy["factor_count_rule"] == (
        "eigenvalue_strictly_above_marchenko_pastur_upper_edge"
    )
    assert policy["loading_alignment"] == "orthogonal_procrustes_to_previous_window"
    assert policy["single_column_persistence_test"] == "exempt_rotation_indeterminate"
    assert policy["stability_test"] == (
        "largest_principal_angle_between_loading_subspaces"
    )
    assert policy["per_board_estimation"] == "forbidden"
    assert policy["alpha_registry_overlap"] == "forbidden"


def test_g4_diagnostics_cannot_auto_select_wls_weight_exponent() -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    diagnostic = payload["g4_0_attribution_diagnostics"]
    assert diagnostic["wls_exponent_diagnostic"]["selection_gate"] is False
    assert diagnostic["group_dimensions"] == [
        "board", "market_cap_decile", "industry_l1", "liquidity_decile"
    ]
    assert diagnostic["incremental_r2"] == {
        "implementation_id": INCREMENTAL_R2_IMPLEMENTATION_ID,
        "scope": "full_label_domain",
        "weighting": "weight_metric_contract_v1.regression_base_weight_artifact",
        "huber_multiplier_included": False,
        "diagnostic_unweighted_version": True,
        "baseline": ["country", "industry_sw1", "board"],
    }


def test_specific_variance_robustification_is_preregistered() -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_risk_validation_v1.json").read_text()
    )
    policy = payload["specific_variance"]
    assert policy["implementation_id"] == SPECIFIC_VARIANCE_IMPLEMENTATION_ID
    assert policy["estimator"] == "winsorized_residual_squared_ewma_with_group_shrinkage"
    assert policy["robustification"]["clip"] == "center_plus_or_minus_5_times_scale"
    assert policy["robustification"]["uses_only_strictly_prior_residuals_for_scale"] is True
    assert policy["shrinkage"]["target"] == "liquidity_decile_mean_specific_variance"
    assert policy["shrinkage"]["prior_effective_observations"] == 120
    assert "market-cap" in policy["shrinkage"][
        "departure_from_conventional_market_cap_grouping"
    ]


def test_g5_bias_test_has_independent_liquidity_axis() -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_risk_validation_v1.json").read_text()
    )
    bias = payload["bias_test"]
    assert bias["independent_stratification_axes"] == [
        "predicted_specific_risk_decile", "liquidity_decile",
    ]
    assert bias["liquidity_decile_test"]["window_trading_days"] == 250
    assert bias["liquidity_decile_test"]["lower_bound"] == 0.91
    assert bias["liquidity_decile_test"]["upper_bound"] == 1.09


def test_specific_variance_config_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    path.write_text(json.dumps({"specific_variance": {"implementation_id": "wrong"}}))
    with pytest.raises(ValueError, match="SPECIFIC_VARIANCE_IMPLEMENTATION_CONFIG_MISMATCH"):
        validate_specific_variance_config(path)


def test_g4_weight_implementation_config_mismatch_fails(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    payload["g4_1a_wls_weight_protocol"]["implementation_id"] = "wrong"
    path = tmp_path / "g4.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="G4_WEIGHT_IMPLEMENTATION_CONFIG_MISMATCH"):
        run_g4_wls_weight_selection(DataLake(tmp_path / "data"), config_path=path)


def test_incremental_r2_implementation_config_mismatch_fails(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    payload["g4_0_attribution_diagnostics"]["incremental_r2"]["implementation_id"] = "wrong"
    path = tmp_path / "g4.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="G4_INCREMENTAL_R2_IMPLEMENTATION_CONFIG_MISMATCH"):
        run_g4_attribution_diagnostics(DataLake(tmp_path / "data"), config_path=path)


def test_statistical_spectrum_evaluation_schedule_is_preregistered() -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    protocol = payload["g4_1_statistical_factor_protocol"]
    assert protocol["rolling_window_days"] == 252
    assert protocol["evaluation_stride_trading_days"] == 21
    assert protocol["stable_factor_count_rule"] == {
        "aggregation": "median_across_adjacent_21_trading_day_windows",
        "maximum_largest_principal_angle_degrees": 30.0,
        "definition": "largest_dimension_with_median_largest_principal_angle_at_or_below_threshold",
        "frozen_before_full_dimension_rerun": True,
    }


def test_g4_2_uses_frozen_median_incremental_r_squared_not_greedy() -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    protocol = payload["g4_2_preregistered_member_protocol"]
    assert protocol["candidate_order"] == [
        "size", "beta", "residual_volatility", "liquidity",
        "nonlinear_size", "listing_age",
    ]
    assert protocol["evaluation_order_policy"] == "fixed_order_not_incremental_r2_greedy"
    assert protocol["incremental_r2"]["primary_aggregation"] == (
        "median_across_daily_incremental_r_squared"
    )
    assert protocol["incremental_r2"]["reported_diagnostic_aggregation"] == (
        "mean_across_daily_incremental_r_squared"
    )
    assert protocol["report_field_order"][-2:] == [
        "incremental_r2_median_primary", "incremental_r2_mean_diagnostic",
    ]
    assert protocol["report_field_order"].index("alpha_dimension_cost") < (
        protocol["report_field_order"].index("incremental_r2_median_primary")
    )


def test_future_alpha_policy_does_not_block_g6_and_forbids_all_risk() -> None:
    policy = json.loads(
        (PROJECT / "config" / "future_alpha_neutralization_policy_v1.json").read_text()
    )
    assert policy["g6_risk_freeze_dependency"] is False
    assert policy["all_risk_shorthand"] == "forbidden"
    assert set(policy["required_explicit_boolean_decisions"]) == {
        "beta", "residual_volatility", "liquidity", "listing_age",
        "statistical_factors",
    }
    legacy = json.loads(
        (PROJECT / "config" / "alpha_dimension_budget_v1.json").read_text()
    )
    assert legacy["status"] == "superseded_legacy_probe_deleted_not_an_active_g6_gate"
    assert legacy["historical_probe_run_id"] is None


def test_g6_freezes_only_risk_model_not_future_alpha_choices() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g6_freeze_protocol_v1.json").read_text()
    )
    boundary = protocol["alpha_stage_boundary"]
    assert boundary["registered_alpha_candidate_count_at_freeze"] == 0
    assert boundary["future_alpha_neutralization_decisions_block_g6"] is False
    assert boundary["future_registration_policy"] == (
        "config/future_alpha_neutralization_policy_v1.json"
    )


def test_statistical_semantic_probe_is_explicitly_not_alpha() -> None:
    probe = json.loads(
        (PROJECT / "config" / "g4_statistical_loading_semantic_probe_v1.json").read_text()
    )
    assertions = probe["explicit_non_alpha_assertions"]
    assert probe["status"] == "pending_one_run_on_current_risk_basis"
    assert probe["completed_run_id"] is None
    assert probe["risk_basis_id"] == "risk_basis_5649d34a30c03cef"
    assert assertions["factor_specs_registered"] == 0
    assert assertions["alpha_exposures_published"] == 0
    assert assertions["forward_return_labels_used"] is False
    assert assertions["IC_or_portfolio_metrics_produced"] is False


def test_semantic_probe_reentry_has_a_cost_and_cannot_repeat_descriptors() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "semantic_probe_reentry_protocol_v1.json").read_text()
    )
    assert "semantic_probe_decision_count increments" in " ".join(
        protocol["preconditions"]
    )
    assert protocol["multiplicity_boundary"]["included_in_l2a_bh_denominator"] is False
    assert any("previously tested descriptor" in item for item in protocol["forbidden"])
    assert "availability timestamps" in protocol["analyst_coverage_unlock"]


def test_second_style_probe_is_frozen_and_non_decisional() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g4_style_loading_semantic_probe_v1.json").read_text()
    )
    assert protocol["run_policy"] == "exactly_once_per_risk_basis"
    assert protocol["risk_basis_id"] == "risk_basis_5649d34a30c03cef"
    assert protocol["risk_metric_id"] == "risk_metric_524163ea01abfe28"
    assert protocol["l1_run_id"] == "l1_risk_exposure_history_20260814_e3b192fa61031760"
    assert protocol["spectrum_run_id"] is None
    assert protocol["basis_status"] == "pending_valid_g3_and_statistical_spectrum"
    assert set(protocol["style_bases"]) == {
        "beta", "residual_volatility", "liquidity", "listing_age",
    }
    assert protocol["forward_return_labels_used"] is False
    assert protocol["automatic_neutralization_decision"] is False


def test_g6_is_split_and_theme_monitor_blocks_only_optimizer_path() -> None:
    g6 = json.loads(
        (PROJECT / "config" / "g6_freeze_protocol_v1.json").read_text()
    )
    gates = g6["split_gates"]
    assert "future alpha registration" in gates["G6a_basis_freeze"]["unblocks"]
    assert "L4 optimizer comparison" in gates["G6b_V_acceptance"]["blocks_until_passed"]
    theme = json.loads(
        (PROJECT / "config" / "theme_observation_v1.json").read_text()
    )
    assert theme["hard_gate"] == "must_be_operational_before_L4_or_any_production_optimizer"
    assert "G6a basis freeze" in theme["does_not_block"]


def test_risk_v2_and_g4_termination_cannot_be_driven_by_alpha_results() -> None:
    versioning = json.loads(
        (PROJECT / "config" / "risk_set_versioning_v1.json").read_text()
    )
    assert "alpha IC" in versioning["forbidden_trigger"]
    termination = json.loads(
        (PROJECT / "config" / "g4_termination_rule_v1.json").read_text()
    )
    assert "R2 reaches a target" in termination["explicitly_not_required"]
    assert any("no new problem category" in item for item in termination["all_required"])


def test_statistical_alpha_overlap_is_one_shot_and_return_free() -> None:
    protocol = json.loads(
        (PROJECT / "config" / "g4_statistical_alpha_overlap_v1.json").read_text()
    )
    assert protocol["run_policy"] == "exactly_once_per_risk_basis"
    assert protocol["risk_basis_id"] == "risk_basis_5649d34a30c03cef"
    assert protocol["risk_metric_id"] == "risk_metric_524163ea01abfe28"
    assert protocol["g3_run_id"] is None
    assert protocol["activation_status"] == "pending_valid_g3_and_statistical_spectrum"
    assert protocol["decision_sample_end_inclusive"] == "2025-02-28"
    assert protocol["automatic_change_to_J"] is False
    assert protocol["automatic_alpha_neutralization_decision"] is False
    assert {"IC", "return_spread", "portfolio_return", "performance_metric"} == set(
        protocol["forbidden_outputs"]
    )


def test_bias_failure_cannot_be_used_to_increase_statistical_factor_count() -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_risk_validation_v1.json").read_text()
    )
    branch = payload["freeze_policy"]["statistical_factor_escalation_on_bias_failure"]
    assert branch["increase_J"] == "forbidden"
    assert branch["g6_disposition"].startswith("blocked_under_current_protocol")


def test_statistical_subspace_angle_is_rotation_invariant() -> None:
    assets = ["a", "b", "c", "d"]
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    angle, common = _principal_angle_degrees(assets, basis, assets, basis @ rotation)
    assert common == 4
    assert angle is not None and angle < 1e-6


def test_wls_weight_protocol_uses_direct_weight_units_and_non_degenerate_gate() -> None:
    payload = json.loads(
        (PROJECT / "config" / "g4_validation_protocol_v1.json").read_text()
    )
    protocol = payload["g4_1a_wls_weight_protocol"]
    assert protocol["weight_semantics"] == "diagonal_W_in_X_transpose_W_X"
    assert protocol["selection_rule"] == "minimum_out_of_sample_heteroskedasticity_score"
    assert protocol["forbidden_selection_rule"] == "maximize_r_squared"
    formulas = {item["id"]: item["formula"] for item in protocol["candidates"]}
    assert formulas["structural"] == "1/exp(2*predicted_log_sigma_u)"
    assert "-2*baseline_log_sigma_cap_slope" in formulas["empirical_cap"]
    assert protocol["implementation_id"] == "g4_1a_pit_weight_selection_v1"
    assert protocol["evaluation"]["decision_sample_end_inclusive"] == "2025-02-28"
    assert protocol["evaluation"]["score"] == (
        "coefficient_of_variation_of_group_mean(W_i*u_i^2)"
    )
    assert protocol["candidate_publication"]["publish_current_argument_exposed"] is False
    assert protocol["secondary_guard"]["frozen_before_any_candidate_run"] is True
