from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from factor_matrix.calculation import (
    OperationSpec, ParameterSchema, ScopeMode, Stage, architecture_boundary_policy,
)
from factor_matrix.calculation.l2 import (
    AuditedPredictivityRunner, L2AtomicPublisher, L2BParquetInputs,
    RegressionInput, RegressionMode, ResearchEvent,
    ResearchEventStore, attribution_identity_error, configuration_sha,
    build_daily_categorical_constrained_design,
    build_daily_industry_constrained_design,
    decompose_cross_section, load_predictivity_config,
    load_return_decomposition_config, purged_k_fold_indices,
    rolling_fama_macbeth, rolling_out_of_sample_ic, bias_test,
    ewma_factor_covariance, load_current_l2_product, normalize_diagonal_icir_weights,
    ewma_specific_variance,
    specific_variance_winsorization_comparison,
    summarize_specific_risk_ratio_by_group,
    load_current_g3_manifest,
    risk_set_freeze_decision, run_l2b_from_parquet,
    stratified_residual_permutation_test,
)
import polars as pl
import numpy as np
from factor_matrix.storage import DataLake
from factor_matrix.factor_engine import FactorRegistryStore
from factor_matrix.research_protocol import ResearchProtocol
from factor_matrix.calculation.l2.return_decomposition import (
    HUBER_IMPLEMENTATION_ID, _whitened_mad_scale,
)
from factor_matrix.calculation.l2.risk_modeling import _ewma_weights


PROJECT = Path(__file__).resolve().parents[1]


def derived_params():
    return ResearchProtocol.load(PROJECT / "config" / "research_protocol_v1.json").derive(
        factor_count=2, maximum_evaluation_horizon_days=20,
        cross_section_size=5000,
        available_estimation_days=1400,
    )["values"]


def predictivity_config():
    protocol = ResearchProtocol.load(PROJECT / "config" / "research_protocol_v1.json")
    return load_predictivity_config(
        PROJECT / "config" / "l2_predictivity_v1.json",
        derived_params=derived_params(), frozen_choices=dict(protocol.frozen_choices),
    )


def issue_l2_test_attempt(
    store: ResearchEventStore, *, factor_id: str = "factor_a", horizon_days: int = 1,
) -> str:
    """Create the minimal frozen registry state required by the audited L2a API."""
    FactorRegistryStore(store.path).initialize()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            INSERT INTO feature_registry(
              feature_id, version, feature_key, display_name, formula_expr,
              source_tables_json, source_fields_json, params_json,
              family_root_id, proposed_date
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                factor_id, 1, f"test:{factor_id}:v1", factor_id, "x",
                "[]", "[]", "{}", factor_id, "2026-08-14",
            ),
        )
        connection.execute(
            """
            INSERT INTO risk_set(
              risk_set_version, status, purpose, rationale, created_at
            ) VALUES (?,?,?,?,?)
            """,
            (1, "frozen", "research", "test fixture", "2026-08-14T09:00:00"),
        )
    return store.issue_attempt(
        family_root_id=factor_id,
        feature_id=factor_id,
        feature_version=1,
        feature_spec_hash="b" * 64,
        risk_set_version=1,
        horizon_days=horizon_days,
        code_sha="c" * 64,
        config_sha="d" * 64,
    )


def decomposition_config():
    return load_return_decomposition_config(
        PROJECT / "config" / "l2_return_decomposition_v1.json",
        derived_params=derived_params(),
    )


def test_production_canonical_solver_is_invariant_to_industry_column_scale() -> None:
    l2_payload = json.loads(
        (PROJECT / "config" / "l2_return_decomposition_v1.json").read_text()
    )
    assert l2_payload["solver"] == "scale_invariant_canonicalized_equality_constrained_wls_v1"
    assert l2_payload["numerical_preconditioning"]["implementation_id"] == (
        "full_factor_space_weighted_column_canonicalization_v1"
    )
    base = RegressionInput(
        asset_ids=("a", "b", "c", "d", "e", "f"),
        factor_ids=("risk_country", "risk_industry_a", "risk_industry_b", "risk_size"),
        factor_families=("risk",) * 4,
        exposures=(
            (1.0, 1.0, 0.0, -1.2), (1.0, 1.0, 0.0, -0.5),
            (1.0, 1.0, 0.0, 0.2), (1.0, 0.0, 1.0, -0.1),
            (1.0, 0.0, 1.0, 0.7), (1.0, 0.0, 1.0, 1.4),
        ),
        realized_returns=(0.01, 0.02, 0.00, -0.01, 0.03, 0.02),
        base_weights=(1.0, 1.2, 0.8, 1.1, 0.9, 1.3),
        is_tradable=(True,) * 6,
        equality_constraints=((0.0, 0.55, 0.45, 0.0),),
    )
    left = decompose_cross_section(
        base, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
    )
    left_fit = np.asarray(base.exposures) @ np.asarray(left.factor_returns)
    for scale in (0.1, 1.0, 10.0):
        scaled = RegressionInput(
            asset_ids=base.asset_ids, factor_ids=base.factor_ids,
            factor_families=base.factor_families,
            exposures=tuple(
                (row[0], scale * row[1], scale * row[2], row[3])
                for row in base.exposures
            ),
            realized_returns=base.realized_returns, base_weights=base.base_weights,
            is_tradable=base.is_tradable,
            equality_constraints=((0.0, scale * 0.55, scale * 0.45, 0.0),),
        )
        right = decompose_cross_section(
            scaled, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
        )
        right_fit = np.asarray(scaled.exposures) @ np.asarray(right.factor_returns)
        assert np.max(np.abs(left_fit - right_fit)) < 1e-12
        assert right.condition_number == pytest.approx(left.condition_number, rel=1e-12)
        assert right.r_squared == pytest.approx(left.r_squared, abs=1e-12)
        assert right.estimation_domain_r_squared == pytest.approx(
            left.estimation_domain_r_squared, abs=1e-12
        )
        assert right.specific_returns == pytest.approx(left.specific_returns, abs=1e-12)
        assert right.huber_weight_multipliers == pytest.approx(
            left.huber_weight_multipliers, abs=1e-12
        )
        assert right.constraint_error == pytest.approx(left.constraint_error, abs=1e-12)
        assert np.asarray(right.factor_returns)[1:3] * scale == pytest.approx(
            np.asarray(left.factor_returns)[1:3], abs=1e-12
        )


def test_canonical_condition_gate_fires_on_near_collinear_columns() -> None:
    protocol = ResearchProtocol.load(PROJECT / "config" / "research_protocol_v1.json")
    derived = protocol.derive(
        factor_count=3, maximum_evaluation_horizon_days=1,
        cross_section_size=400, available_estimation_days=1400,
    )["values"]
    config = load_return_decomposition_config(
        PROJECT / "config" / "l2_return_decomposition_v1.json",
        derived_params=derived,
    )
    rng = np.random.default_rng(20260817)
    values = rng.normal(size=400)
    noise = rng.normal(size=400)
    inputs = RegressionInput(
        asset_ids=tuple(f"asset_{index}" for index in range(400)),
        factor_ids=("risk_country", "risk_size", "risk_size_near_collinear"),
        factor_families=("risk", "risk", "risk"),
        exposures=tuple(
            (1.0, float(value), float(value + 1e-8 * perturbation))
            for value, perturbation in zip(values, noise)
        ),
        realized_returns=tuple(
            float(0.01 * value + rng.normal(scale=0.001)) for value in values
        ),
        base_weights=(1.0,) * 400,
        is_tradable=(True,) * 400,
    )

    result = decompose_cross_section(
        inputs, mode=RegressionMode.RISK_ONLY, config=config
    )

    assert result.condition_number > config.maximum_condition_number
    assert result.status == "invalid"


def test_l2_config_records_scale_invariant_condition_gate() -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_return_decomposition_v1.json").read_text()
    )
    preconditioner = payload["numerical_preconditioning"]
    assert preconditioner["free_parameters"] == 0
    assert preconditioner["published_exposure_units_changed"] is False
    assert preconditioner["published_factor_return_units_changed"] is False
    assert "Q_transpose_Z_transpose_W_Z_Q" in payload["condition_number_matrix"]


def test_singular_gram_recovery_cannot_masquerade_as_a_valid_solve(monkeypatch):
    from factor_matrix.calculation.l2.linear_algebra import solve_prepared_weighted_least_squares

    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)
    x = np.column_stack((np.ones(20), np.arange(20, dtype=float)))
    y = x @ np.array([0.01, 0.02])
    coefficients, condition = solve_prepared_weighted_least_squares(
        x, y, np.ones(20), np.eye(2),
    )
    assert x @ coefficients == pytest.approx(y)
    assert np.isinf(condition)
    # SVD recovery cannot silently identify a truly rank-deficient model.
    x[:, 1] = x[:, 0]
    with pytest.raises(ValueError, match="LINEAR_SYSTEM_SINGULAR"):
        solve_prepared_weighted_least_squares(x, y, np.ones(20), np.eye(2))


def test_huber_config_matches_whitened_implementation() -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_return_decomposition_v1.json").read_text()
    )
    assert payload["robustifier"]["implementation_id"] == HUBER_IMPLEMENTATION_ID


def test_huber_config_mismatch_fails(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT / "config" / "l2_return_decomposition_v1.json").read_text()
    )
    payload["robustifier"]["implementation_id"] = "wrong"
    path = tmp_path / "l2.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="HUBER_IMPLEMENTATION_CONFIG_MISMATCH"):
        load_return_decomposition_config(path, derived_params=derived_params())


def test_specific_variance_is_pit_winsorized_and_group_shrunk() -> None:
    normal = [0.01, -0.01] * 50
    values = {
        "a": [*normal, 1.0],
        "b": [*normal, 0.01],
    }
    result = ewma_specific_variance(
        values, 20.0, group_by_asset={"a": "L1", "b": "L1"},
        minimum_observations=60, prior_effective_observations=120,
    )
    raw_outlier_ewma = sum(
        weight * value * value
        for weight, value in zip(_ewma_weights(len(values["a"]), 20.0), values["a"])
    )
    assert result["a"] < raw_outlier_ewma
    assert result["a"] > result["b"]


def test_specific_variance_reports_like_for_like_winsorization_cost() -> None:
    normal = [0.01, -0.01] * 50
    values = {"d1": [*normal, 1.0], "d10": [*normal, 0.01]}
    groups = {"d1": "D1", "d10": "D10"}
    comparison = specific_variance_winsorization_comparison(
        values, 20.0, group_by_asset=groups,
        minimum_observations=60, prior_effective_observations=0,
    )
    assert comparison["d1"]["winsorized_sigma"] < comparison["d1"]["nonwinsorized_sigma"]
    assert comparison["d10"]["winsorized_to_nonwinsorized_sigma_ratio"] == pytest.approx(1.0)
    summary = summarize_specific_risk_ratio_by_group(comparison, groups)
    assert summary["D1"]["median_winsorized_to_nonwinsorized_sigma_ratio"] < 1.0
    assert summary["D10"]["median_winsorized_to_nonwinsorized_sigma_ratio"] == pytest.approx(1.0)


def test_l2a_uses_only_labels_known_before_horizon_and_embargo() -> None:
    config = predictivity_config()
    asset_axis = tuple(float(index) for index in range(20))
    exposures = [asset_axis for _ in range(650)]
    labels = [tuple(0.0 for _ in asset_axis) for _ in range(550)]
    labels.extend(asset_axis for _ in range(100))

    points = rolling_out_of_sample_ic(
        exposures, labels, horizon_days=20, config=config
    )

    assert all(point.shrunk_ic == 0.0 for point in points[:600])
    assert all(point.training_end_index is None for point in points[:600])


def test_purged_fold_removes_overlapping_and_embargoed_observations() -> None:
    folds = purged_k_fold_indices(100, folds=5, horizon_days=10, embargo_days=5)
    train, test = folds[2]
    assert test == tuple(range(40, 60))
    assert not set(range(30, 65)) & set(train)


def test_l2a_cannot_run_without_an_append_only_research_event(tmp_path: Path) -> None:
    store = ResearchEventStore(tmp_path / "metadata.sqlite")
    config = predictivity_config()
    attempt_id = issue_l2_test_attempt(store)
    runner = AuditedPredictivityRunner(
        store, config, {"factor_a": 1}, {"factor_a": 1}
    )
    event = ResearchEvent(
        event_id="event-1",
        run_at=datetime(2026, 8, 14, 9, 0),
        factor_ids=("factor_a",),
        config_sha=configuration_sha({"config_id": "l2_predictivity_v1"}),
        data_end_date=date(2026, 8, 13),
        touched_holdout=False,
        headline_ic=None,
        decision="Run the preregistered quarterly evaluation.",
        operator="researcher",
        payload={"groups": ["board"], "evaluation_horizon_days": 1},
    )
    values = [tuple(float(index) for index in range(5)) for _ in range(5)]
    runner.run_single_factor(
        event=event, factor_id="factor_a", attempt_id=attempt_id,
        exposures_by_date=values, forward_returns_by_date=values,
    )
    assert store.count() == 1
    with pytest.raises(RuntimeError, match="RESEARCH_EVENT_APPEND_ONLY_CONFLICT"):
        runner.run_single_factor(
            event=event, factor_id="factor_a", attempt_id=attempt_id,
            exposures_by_date=values, forward_returns_by_date=values,
        )


def test_l2a_admission_snapshot_reads_labels_once_for_all_derived_outputs(
    tmp_path: Path,
) -> None:
    store = ResearchEventStore(tmp_path / "metadata.sqlite")
    attempt_id = issue_l2_test_attempt(store)
    runner = AuditedPredictivityRunner(
        store, predictivity_config(), {"factor_a": 1}, {"factor_a": 1}
    )
    event = ResearchEvent(
        event_id="admission-snapshot-1", run_at=datetime(2026, 8, 14, 9, 0),
        factor_ids=("factor_a",), config_sha="a" * 64,
        data_end_date=date(2026, 8, 13), touched_holdout=False,
        headline_ic=None, decision="Run one bundled admission evaluation.",
        operator="researcher", payload={
            "evaluation_horizon_days": 1,
            "factor_evaluation_snapshot": {
                "outputs": [
                    "raw_ic", "neutralized_ic", "neutralization_retention", "incremental_ic",
                ],
                "variant_count_charge": 1,
            },
            "neutralization_fidelity": {
                "minimum_absolute_raw_ic": 0.005,
                "minimum_retention_ratio": 0.30,
            },
        },
    )
    values = [tuple(float(index) for index in range(5)) for _ in range(5)]
    outputs = runner.run_factor_evaluation_snapshot(
        event=event, factor_id="factor_a", attempt_id=attempt_id,
        raw_exposures_by_date=values,
        neutralized_exposures_by_date=values,
        incremental_exposures_by_date=values, forward_returns_by_date=values,
        minimum_absolute_raw_ic=0.005, minimum_retention_ratio=0.30,
    )
    assert len(outputs) == 4
    assert store.count() == 1


def test_holdout_touch_permanently_downgrades_evaluation_status(tmp_path: Path) -> None:
    store = ResearchEventStore(tmp_path / "metadata.sqlite")
    store.append(ResearchEvent(
        event_id="holdout-event", run_at=datetime(2026, 8, 14, 9, 0),
        factor_ids=("factor_a",), config_sha="a" * 64,
        data_end_date=date(2026, 8, 13), touched_holdout=True,
        headline_ic=0.01, decision="Holdout was inspected.", operator="researcher",
        payload={},
    ))
    assert store.evaluation_status() == "degraded_holdout_contaminated"


def test_fama_macbeth_is_zero_until_trailing_burn_in_is_available() -> None:
    config = predictivity_config()
    count = config.minimum_burn_in_days + config.embargo_days + 1
    daily_x = [((1.0, -1.0), (1.0, 0.0), (1.0, 1.0)) for _ in range(count)]
    daily_y = [(-0.01, 0.01, 0.03) for _ in range(count)]
    daily_w = [(1.0, 1.0, 1.0) for _ in range(count)]
    points = rolling_fama_macbeth(
        daily_x, daily_y, daily_w, equality_constraints=(),
        horizon_days=1, config=config,
    )
    assert points[-1].coefficients == pytest.approx((0.01, 0.02), abs=1e-12)
    assert points[-1].training_end_index == config.minimum_burn_in_days - 1


def test_negative_ic_cannot_silently_reverse_factor_direction() -> None:
    with pytest.raises(ValueError, match="NEGATIVE_IC_INVALIDATE_FACTOR"):
        normalize_diagonal_icir_weights((1.0, -0.2))


def test_l2b_risk_only_rejects_alpha_columns() -> None:
    inputs = RegressionInput(
        asset_ids=tuple("ABCDEF"), factor_ids=("country", "alpha_a"),
        factor_families=("risk", "alpha"),
        exposures=tuple((1.0, float(index)) for index in range(6)),
        realized_returns=tuple(float(index) for index in range(6)),
        base_weights=(1.0,) * 6, is_tradable=(True,) * 6,
    )
    with pytest.raises(ValueError, match="ALPHA_COLUMN_FORBIDDEN"):
        decompose_cross_section(
            inputs, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
        )


def test_superseded_g3_current_is_not_consumable(tmp_path: Path) -> None:
    current_dir = tmp_path / "data" / "gold" / "l2b_risk_only"
    current_dir.mkdir(parents=True)
    (current_dir / "_CURRENT.json").write_text(
        '{"status":"superseded","run_id":"old","manifest":"unused.json"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="G3_CURRENT_NOT_CONSUMABLE status=superseded"):
        load_current_g3_manifest(DataLake(tmp_path / "data"))


def test_daily_industry_constraints_use_cap_not_sqrt_cap_and_drop_empty_industries() -> None:
    frame = pl.DataFrame({
        "asset_id": ["A", "B", "C"],
        "industry_id": ["BANK", "BANK", "TECH"],
        "realized_return": [0.01, 0.02, -0.01],
        "float_mkt_cap": [100.0, 300.0, 600.0],
        "is_tradable": [True, True, True],
        "country": [1.0, 1.0, 1.0],
        "size": [-1.0, 0.0, 1.0],
    })
    design = build_daily_industry_constrained_design(
        frame, exposure_columns=("country", "size"),
        family_by_column={"country": "risk", "size": "risk"},
    )
    assert design.present_industries == ("BANK", "TECH")
    assert design.industry_constraint_weights == pytest.approx({"BANK": 0.4, "TECH": 0.6})
    assert design.industry_wls_weight_sums["BANK"] == pytest.approx(10 + 300 ** 0.5)
    assert design.regression_input.equality_constraints[0] == pytest.approx(
        (0.0, 0.0, 0.4, 0.6)
    )
    assert len(design.warnings) == 2


def test_categorical_design_reports_cap_constraint_and_wls_aggregates_separately() -> None:
    frame = pl.DataFrame({
        "asset_id": list("ABCDEF"),
        "realized_return": [0.01, 0.02, 0.03, -0.01, -0.02, -0.03],
        "float_mkt_cap": [100.0, 300.0, 600.0, 100.0, 300.0, 600.0],
        "model_eligible": [True] * 6,
        "in_estimation_domain": [True] * 6,
        "risk_country": [1.0] * 6,
        "risk_industry_BANK": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "risk_industry_TECH": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    })
    design = build_daily_categorical_constrained_design(
        frame,
        exposure_columns=("risk_country", "risk_industry_BANK", "risk_industry_TECH"),
        family_by_column={
            "risk_country": "risk", "risk_industry_BANK": "risk",
            "risk_industry_TECH": "risk",
        },
        categorical_blocks={
            "industry": ("risk_industry_BANK", "risk_industry_TECH"),
        },
        estimation_column="in_estimation_domain",
    )
    assert design.industry_constraint_weights == pytest.approx({
        "risk_industry_BANK": 0.5, "risk_industry_TECH": 0.5,
    })
    expected_wls_sum = 10.0 + 300.0 ** 0.5 + 600.0 ** 0.5
    assert design.industry_wls_weight_sums == pytest.approx({
        "risk_industry_BANK": expected_wls_sum,
        "risk_industry_TECH": expected_wls_sum,
    })
    assert design.category_wls_weight_sums["industry"] == pytest.approx(
        design.industry_wls_weight_sums
    )


def test_l2b_exactly_decomposes_a_well_conditioned_cross_section() -> None:
    x_values = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    returns = tuple(0.01 + 0.02 * value for value in x_values)
    inputs = RegressionInput(
        asset_ids=tuple("ABCDEF"), factor_ids=("country", "size"),
        factor_families=("risk", "risk"),
        exposures=tuple((1.0, value) for value in x_values),
        realized_returns=returns, base_weights=(1.0,) * 6,
        is_tradable=(True,) * 6,
    )
    result = decompose_cross_section(
        inputs, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
    )
    assert result.status == "passed"
    assert result.factor_returns == pytest.approx((0.01, 0.02), abs=1e-12)
    assert result.specific_returns == pytest.approx((0.0,) * 6, abs=1e-12)
    assert result.r_squared == pytest.approx(1.0)


def test_l2b_estimation_domain_is_strict_subset_of_saved_residual_domain() -> None:
    x_values = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    returns = tuple(0.01 + 0.02 * value for value in x_values[:-1]) + (0.50,)
    inputs = RegressionInput(
        asset_ids=tuple("ABCDEFG"), factor_ids=("country", "size"),
        factor_families=("risk", "risk"),
        exposures=tuple((1.0, value) for value in x_values),
        realized_returns=returns, base_weights=(1.0,) * 7,
        is_tradable=(True,) * 6 + (False,),
    )
    result = decompose_cross_section(
        inputs, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
    )
    assert result.included_assets == tuple("ABCDEFG")
    assert result.sample_count == 6
    assert result.label_sample_count == 7
    assert result.in_estimation_domain == (True,) * 6 + (False,)
    assert result.estimation_weights[-1] is None
    assert result.huber_weight_multipliers[-1] is None
    assert result.specific_returns[-1] == pytest.approx(0.41, abs=1e-12)
    assert result.estimation_domain_r_squared == pytest.approx(1.0)
    assert result.r_squared < result.estimation_domain_r_squared
    assert result.unweighted_r_squared < result.estimation_domain_unweighted_r_squared
    assert result.regression_identity_error < 1e-12


def test_huber_downweights_but_never_hard_deletes_estimation_observations() -> None:
    x_values = tuple(float(value) for value in range(-5, 6))
    returns = tuple(0.01 + 0.02 * value for value in x_values[:-1]) + (2.0,)
    inputs = RegressionInput(
        asset_ids=tuple(f"A{index}" for index in range(len(x_values))),
        factor_ids=("country", "size"), factor_families=("risk", "risk"),
        exposures=tuple((1.0, value) for value in x_values),
        realized_returns=returns, base_weights=(1.0,) * len(x_values),
        is_tradable=(True,) * len(x_values),
    )
    result = decompose_cross_section(
        inputs, mode=RegressionMode.RISK_ONLY, config=decomposition_config()
    )
    assert result.label_sample_count == result.sample_count == len(x_values)
    assert result.excluded_assets == ()
    assert len(result.specific_returns) == len(x_values)
    assert any(value < 1.0 for value in result.huber_weight_multipliers if value is not None)


def test_huber_scale_is_computed_in_whitened_space_about_zero() -> None:
    residuals = np.asarray([-10.0, 1.0, 1.0, 1.0])
    base_weights = np.asarray([0.01, 1.0, 4.0, 9.0])
    assert _whitened_mad_scale(residuals, base_weights) == pytest.approx(
        1.5 / 0.6744897501960817
    )


def test_constrained_industry_model_has_exact_expected_rank_deficiency() -> None:
    frame = pl.DataFrame({
        "asset_id": list("ABCDEF"),
        "industry_id": ["BANK"] * 3 + ["TECH"] * 3,
        "float_mkt_cap": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0],
        "is_tradable": [True] * 6,
        "country": [1.0] * 6,
        "size": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    })
    bank_return, tech_return = 0.03, -0.02
    frame = frame.with_columns(pl.Series("realized_return", [
        0.01 + 0.02 * size + (bank_return if industry == "BANK" else tech_return)
        for size, industry in zip(frame["size"], frame["industry_id"])
    ]))
    design = build_daily_industry_constrained_design(
        frame, exposure_columns=("country", "size"),
        family_by_column={"country": "risk", "size": "risk"},
    )
    result = decompose_cross_section(
        design.regression_input, mode=RegressionMode.RISK_ONLY,
        config=decomposition_config(),
    )
    assert result.matrix_rank == len(result.factor_returns) - 1
    constraint = design.regression_input.equality_constraints[0]
    assert abs(sum(value * coefficient for value, coefficient in zip(
        constraint, result.factor_returns
    ))) < 1e-12
    assert result.status == "passed"


def test_multiple_categorical_blocks_are_jointly_identified_and_constrained() -> None:
    industries = ["BANK"] * 6 + ["TECH"] * 6
    boards = ["MAIN", "MAIN", "MAIN", "STAR", "STAR", "STAR"] * 2
    sizes = [-2.3, -0.9, 0.1, -1.7, 0.2, 0.8, -0.4, 1.1, 1.9, 0.7, 2.5, 3.1]
    caps = [100.0, 150.0, 130.0, 120.0, 180.0, 160.0,
            200.0, 250.0, 230.0, 220.0, 300.0, 280.0]
    frame = pl.DataFrame({
        "asset_id": [f"A{index}" for index in range(12)],
        "realized_return": [
            0.01 + 0.02 * size
            + (0.03 if industry == "BANK" else -0.02)
            + (0.015 if board == "MAIN" else -0.01)
            for size, industry, board in zip(sizes, industries, boards)
        ],
        "float_mkt_cap": caps,
        "model_eligible": [True] * 12,
        "risk_country": [1.0] * 12,
        "risk_industry_BANK": [float(value == "BANK") for value in industries],
        "risk_industry_TECH": [float(value == "TECH") for value in industries],
        "risk_board_MAIN": [float(value == "MAIN") for value in boards],
        "risk_board_STAR": [float(value == "STAR") for value in boards],
        "risk_size": sizes,
    })
    factor_columns = (
        "risk_country", "risk_industry_BANK", "risk_industry_TECH",
        "risk_board_MAIN", "risk_board_STAR", "risk_size",
    )
    design = build_daily_categorical_constrained_design(
        frame,
        exposure_columns=factor_columns,
        family_by_column={column: "risk" for column in factor_columns},
        categorical_blocks={
            "industry": ("risk_industry_BANK", "risk_industry_TECH"),
            "board": ("risk_board_MAIN", "risk_board_STAR"),
        },
    )
    result = decompose_cross_section(
        design.regression_input, mode=RegressionMode.RISK_ONLY,
        config=decomposition_config(),
    )

    assert design.constraint_rank == 2
    assert design.matrix_rank == len(factor_columns) - 2
    assert design.stacked_rank == len(factor_columns)
    assert design.kkt_rank == len(factor_columns) + 2
    assert design.constraint_identification_passed
    for constraint in design.regression_input.equality_constraints:
        assert abs(sum(value * coefficient for value, coefficient in zip(
            constraint, result.factor_returns
        ))) < 1e-12
    assert result.status == "passed"


def test_l2_atomic_publisher_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    dates = [date(2026, 8, 12), date(2026, 8, 13)]
    products = {
        "factor_returns_v1": pl.DataFrame({
            "trade_date": [dates[0], dates[0], dates[1], dates[1]],
            "factor_id": ["country", "size"] * 2,
            "regression_mode": ["risk_only"] * 4,
            "factor_family": ["risk"] * 4,
            "factor_return": [0.01, 0.02, 0.0, -0.01],
        }),
        "specific_returns_v1": pl.DataFrame({
            "trade_date": [dates[0], dates[0], dates[1], dates[1]],
            "asset_id": ["A", "B"] * 2,
            "regression_mode": ["risk_only"] * 4,
            "specific_return": [0.0] * 4,
        }),
        "factor_regression_quality_v1": pl.DataFrame({
            "trade_date": dates, "regression_mode": ["risk_only"] * 2,
            "status": ["passed"] * 2, "sample_count": [2, 2],
            "excluded_count": [0, 0], "factor_count": [2, 2],
            "constraint_count": [0, 0], "expected_matrix_rank": [2, 2],
            "constraint_identification_passed": [True, True],
            "matrix_rank": [2, 2], "condition_number": [2.0, 2.0],
            "r_squared": [0.4, 0.4], "r_squared_in_expected_range": [True, True],
            "constraint_error": [0.0, 0.0],
            "regression_identity_error": [0.0, 0.0],
            "maximum_group_residual_correlation": [None, None],
        }),
        "cross_section_stats_v1": pl.DataFrame({
            "trade_date": dates, "board_id": ["MAIN", "MAIN"],
            "regression_mode": ["risk_only"] * 2, "sigma_r": [0.02, 0.03],
            "n_valid": [2, 2], "mean_abs_return": [0.01, 0.02],
            "skew": [0.0, 0.0], "kurtosis": [1.0, 1.0],
        }),
    }
    input_path = tmp_path / "input.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(input_path)
    publisher = L2AtomicPublisher(DataLake(tmp_path / "data"))
    arguments = dict(
        products=products, mode="risk_only", configuration={"version": 1},
        risk_set_version=1, input_paths={"exposure_matrix_v1": input_path},
        date_range=(dates[0], dates[1]),
        derived_params={**derived_params(), "unresolved_required": []},
        code_sha="code-fixture",
    )
    first = publisher.publish_l2b(**arguments)
    second = publisher.publish_l2b(**arguments)
    assert first == second
    assert (tmp_path / "data" / "gold" / "l2" / "_CURRENT").read_text().strip() == first
    manifest, loaded = load_current_l2_product(
        DataLake(tmp_path / "data"), "factor_returns_v1"
    )
    assert manifest["run_id"] == first
    assert loaded.height == 4


def test_l2b_parquet_runner_joins_only_next_trading_day_returns(tmp_path: Path) -> None:
    exposure_dates = [date(2026, 8, 10), date(2026, 8, 11)]
    return_dates = [date(2026, 8, 11), date(2026, 8, 12)]
    assets = list("ABCDEF")
    sizes = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    exposure = pl.DataFrame({
        "trade_date": [day for day in exposure_dates for _ in assets],
        "asset_id": assets * 2, "country": [1.0] * 12, "size": sizes * 2,
    })
    universe = pl.DataFrame({
        "trade_date": [day for day in exposure_dates for _ in assets],
        "asset_id": assets * 2, "is_tradable": [True] * 12,
        "board_id": ["MAIN"] * 12,
        "sw_l1_code": (["BANK"] * 3 + ["TECH"] * 3) * 2,
        "exchange_list_date": [date(2020, 1, 1)] * 12,
    })
    valuation = pl.DataFrame({
        "trade_date": [day for day in exposure_dates for _ in assets],
        "asset_id": assets * 2,
        "float_mkt_cap": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0] * 2,
    })
    realized = []
    for day in return_dates:
        for asset, size, industry in zip(assets, sizes, ["BANK"] * 3 + ["TECH"] * 3):
            realized.append({
                "trade_date": day, "asset_id": asset,
                "total_return": 0.01 + 0.02 * size + (0.03 if industry == "BANK" else -0.02),
                "return_source": "price",
            })
    paths = {}
    for name, frame in {
        "exposure": exposure, "universe": universe,
        "returns": pl.DataFrame(realized), "valuation": valuation,
    }.items():
        path = tmp_path / f"{name}.parquet"
        frame.write_parquet(path)
        paths[name] = path
    products = run_l2b_from_parquet(
        L2BParquetInputs(paths["exposure"], paths["universe"], paths["returns"], paths["valuation"]),
        exposure_columns=("country", "size"),
        family_by_column={"country": "risk", "size": "risk"},
        mode=RegressionMode.RISK_ONLY, config=decomposition_config(),
    )
    quality = products["factor_regression_quality_v1"]
    assert quality.get_column("trade_date").to_list() == return_dates
    assert quality.get_column("exposure_date").to_list() == exposure_dates
    assert quality.get_column("status").to_list() == ["passed", "passed"]


def test_l2b_parquet_runner_forwards_frozen_base_weights_to_canonical_design(
    tmp_path: Path,
) -> None:
    exposure = pl.DataFrame({
        "trade_date": [date(2026, 8, 10)] * 6,
        "asset_id": list("ABCDEF"), "country": [1.0] * 6,
        "size": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    })
    universe = pl.DataFrame({
        "trade_date": [date(2026, 8, 10)] * 6,
        "asset_id": list("ABCDEF"), "is_tradable": [True] * 6,
        "board_id": ["MAIN"] * 6, "sw_l1_code": ["BANK"] * 3 + ["TECH"] * 3,
        "exchange_list_date": [date(2020, 1, 1)] * 6,
    })
    returns = pl.DataFrame({
        "trade_date": [date(2026, 8, 11)] * 6,
        "asset_id": list("ABCDEF"),
        "total_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "return_source": ["price"] * 6,
    })
    valuation = pl.DataFrame({
        "trade_date": [date(2026, 8, 10)] * 6,
        "asset_id": list("ABCDEF"), "float_mkt_cap": [100.0] * 6,
    })
    frozen_weights = pl.DataFrame({
        "exposure_date": [date(2026, 8, 10)] * 6,
        "asset_id": list("ABCDEF"), "candidate_weight": [0.0] * 6,
    })
    paths = {}
    for name, frame in {
        "exposure": exposure, "universe": universe, "returns": returns,
        "valuation": valuation,
    }.items():
        path = tmp_path / f"{name}.parquet"
        frame.write_parquet(path)
        paths[name] = path
    weights_path = tmp_path / "weights.parquet"
    frozen_weights.write_parquet(weights_path)

    with pytest.raises(ValueError, match="L2B_BASE_WEIGHT_INVALID"):
        run_l2b_from_parquet(
            L2BParquetInputs(
                paths["exposure"], paths["universe"],
                paths["returns"], paths["valuation"],
            ),
            exposure_columns=("country", "size"),
            family_by_column={"country": "risk", "size": "risk"},
            mode=RegressionMode.RISK_ONLY,
            config=decomposition_config(),
            base_weight_path=weights_path,
        )


def test_l2b_attribution_identity_is_machine_precision() -> None:
    weights = (0.4, 0.6)
    exposures = ((1.0, -1.0), (1.0, 2.0))
    factor_returns = (0.01, 0.02)
    specific = (0.003, -0.002)
    returns = tuple(
        sum(value * coefficient for value, coefficient in zip(row, factor_returns)) + residual
        for row, residual in zip(exposures, specific)
    )
    assert attribution_identity_error(
        weights, returns, exposures, factor_returns, specific
    ) < 1e-15


def test_l2c_covariance_bias_and_freeze_gate_are_separate_from_l2b() -> None:
    covariance = ewma_factor_covariance({
        "country": [0.01, -0.01, 0.02], "size": [0.0, 0.01, -0.01],
    }, half_life_days=2)
    assert covariance[("country", "size")] == pytest.approx(
        covariance[("size", "country")]
    )
    result = bias_test(
        [1.0, -1.0] * 30, [1.0] * 60,
        lower_bound=0.95, upper_bound=1.05, minimum_observations=60,
    )
    assert result.passed
    assert risk_set_freeze_decision(
        bias_passed=True, residual_results=(),
        significant_dimensions_are_pit=False,
        statistical_factor_already_added=False,
    ) == "sufficient_can_freeze"


def test_residual_correlation_uses_stratified_permutation_null() -> None:
    residuals = {
        "A": [1.0, 2.0, 3.0, 4.0], "B": [1.1, 2.1, 3.1, 4.1],
        "C": [4.0, 1.0, 3.0, 2.0], "D": [1.0, 4.0, 2.0, 3.0],
    }
    results = stratified_residual_permutation_test(
        residuals,
        {"A": "theme", "B": "theme"},
        {asset: ("IND", "SIZE") for asset in residuals},
        permutations=20, random_seed=7, bh_q=0.05,
    )
    assert results[0].group_id == "theme"
    assert 0 < results[0].permutation_p_value <= 1


def test_only_l2_stages_may_declare_return_label_inputs() -> None:
    bad = OperationSpec(
        operation_id="illegal_label_consumer", version="1", stage=Stage.EXPOSURE,
        input_artifact_types=("forward_labels",),
        input_artifact_versions={"forward_labels": "1"},
        output_artifact_types=("risk_exposure_matrix",),
        output_artifact_versions={"risk_exposure_matrix": "1"},
        parameters=ParameterSchema(fields=(), allow_extra=False),
        scope_mode=ScopeMode.PER_BOARD, accepts_benchmark_inputs=False,
        description="must fail",
    )
    with pytest.raises(ValueError, match="ARTIFACT_REQUIRED_INPUT_MISSING|LAYER_INPUT_FORBIDDEN"):
        policy = architecture_boundary_policy()
        policy.validate_spec(bad)
        policy.validate_inputs(bad, ())

    policy = architecture_boundary_policy()
    for stage in Stage:
        if stage not in {Stage.L2_PREDICTIVITY, Stage.L2_RETURN_DECOMPOSITION}:
            assert {"realized_returns", "forward_labels"} <= policy.forbidden_input_types_by_stage[stage]
    assert "forward_labels" in policy.forbidden_input_types_by_stage[
        Stage.L2_RETURN_DECOMPOSITION
    ]
    assert {"factor_returns", "specific_returns"} <= policy.forbidden_input_types_by_stage[
        Stage.L2_PREDICTIVITY
    ]
