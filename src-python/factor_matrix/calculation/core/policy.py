from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ArtifactRef, OperationSpec, Stage


# Operations permitted to publish an artifact without consuming its declared
# required inputs, because they adopt an artifact that was already built
# outside the calculation core rather than computing it.
#
# The required-input rule targets construction: an operation that computes an
# exposure matrix without reading the tradable universe has produced something
# out of thin air. Adoption is a different act. L1 already built and published
# its exposures with the right inputs; the adapter only makes that artifact
# addressable here. A historical base_matrix artifact has never existed in this
# project, so there is no way to satisfy the rule literally either.
#
# This list is deliberately a module constant rather than a parameter. Adding an
# entry asserts that the artifact was produced correctly somewhere else and that
# its provenance is recorded in output metadata. That assertion should be
# reviewed, not configured at call time. Adapters are transitional: if L1 is
# ever rebuilt inside the calculation core, its entry must be removed rather
# than left standing next to the builder.
ADOPTION_OPERATIONS = frozenset({
    "adopt_risk_exposure_matrix",
    "adopt_tradable_universe",
})


@dataclass(frozen=True)
class BoundaryPolicy:
    producer_stage_by_artifact_type: Mapping[str, Stage]
    forbidden_input_types_by_stage: Mapping[Stage, frozenset[str]]
    required_input_types_by_output_type: Mapping[str, frozenset[str]]
    adoption_operations: frozenset[str] = frozenset()

    def validate_spec(self, spec: OperationSpec) -> None:
        for artifact_type in spec.output_artifact_types:
            expected = self.producer_stage_by_artifact_type.get(artifact_type)
            if expected is not None and expected is not spec.stage:
                raise ValueError(
                    f"ARTIFACT_STAGE_CONFLICT type={artifact_type} "
                    f"expected={expected.value} observed={spec.stage.value}"
                )
        declared_inputs = frozenset(spec.input_artifact_types)
        forbidden_declared = sorted(
            declared_inputs & self.forbidden_input_types_by_stage.get(spec.stage, frozenset())
        )
        if forbidden_declared:
            raise ValueError(
                f"CALCULATION_LAYER_INPUT_FORBIDDEN stage={spec.stage.value} "
                f"types={','.join(forbidden_declared)}"
            )
        # Adoption waives the required-input rule and nothing else. Stage
        # ownership and forbidden inputs above still apply, and an adapter that
        # starts declaring inputs is no longer a pure boundary operation, so it
        # loses the exemption instead of quietly keeping it.
        if spec.operation_id in self.adoption_operations:
            if spec.input_artifact_types:
                raise ValueError(
                    f"ADOPTION_OPERATION_MUST_DECLARE_NO_INPUTS "
                    f"operation={spec.operation_id} "
                    f"types={','.join(sorted(spec.input_artifact_types))}"
                )
        else:
            for artifact_type in spec.output_artifact_types:
                required = self.required_input_types_by_output_type.get(
                    artifact_type, frozenset()
                )
                missing = sorted(required - declared_inputs)
                if missing:
                    raise ValueError(
                        f"ARTIFACT_REQUIRED_INPUT_MISSING output={artifact_type} "
                        f"types={','.join(missing)}"
                    )
        l2b_outputs = {"factor_returns", "specific_returns", "factor_regression_quality"}
        if set(spec.output_artifact_types) & l2b_outputs:
            design_inputs = declared_inputs & {"risk_exposure_matrix", "exposure_matrix"}
            if len(design_inputs) != 1:
                raise ValueError("L2B_EXACTLY_ONE_DESIGN_MATRIX_REQUIRED")

    def validate_inputs(self, spec: OperationSpec, inputs: tuple[ArtifactRef, ...]) -> None:
        forbidden = self.forbidden_input_types_by_stage.get(spec.stage, frozenset())
        conflicts = sorted(
            reference.key.artifact_type for reference in inputs
            if reference.key.artifact_type in forbidden
        )
        if conflicts:
            raise ValueError(
                f"CALCULATION_LAYER_INPUT_FORBIDDEN stage={spec.stage.value} "
                f"types={','.join(conflicts)}"
            )


def architecture_boundary_policy() -> BoundaryPolicy:
    """Hard layer boundary: only L2 may consume return labels."""
    producers = {
        "base_matrix": Stage.UNIVERSE,
        "tradable_universe": Stage.UNIVERSE,
        "liquidity_facts": Stage.UNIVERSE,
        "risk_exposure_matrix": Stage.EXPOSURE,
        "risk_factor_set": Stage.L2_RISK_MODEL,
        "exposure_matrix": Stage.EXPOSURE,
        "realized_returns": Stage.L2_RETURN_DECOMPOSITION,
        "market_return_dispersion": Stage.L2_RETURN_DECOMPOSITION,
        "forward_labels": Stage.L2_PREDICTIVITY,
        "probe_signal_panel": Stage.L2_PREDICTIVITY,
        "factor_predictivity": Stage.L2_PREDICTIVITY,
        "fm_regression": Stage.L2_PREDICTIVITY,
        "factor_evaluation_snapshot": Stage.L2_PREDICTIVITY,
        "factor_evaluation_panel": Stage.L2_PREDICTIVITY,
        "factor_returns": Stage.L2_RETURN_DECOMPOSITION,
        "specific_returns": Stage.L2_RETURN_DECOMPOSITION,
        "factor_regression_quality": Stage.L2_RETURN_DECOMPOSITION,
        "cross_section_stats": Stage.L2_RETURN_DECOMPOSITION,
        "factor_covariance": Stage.L2_RISK_MODEL,
        "specific_risk": Stage.L2_RISK_MODEL,
        "security_covariance": Stage.L2_RISK_MODEL,
        "bias_test_report": Stage.L2_RISK_MODEL,
        "residual_correlation_report": Stage.L2_RISK_MODEL,
        "cross_chain_validation": Stage.L2_VALIDATOR,
        "alpha_forecast": Stage.ALPHA,
        "cost_estimate": Stage.COST,
        "stock_score": Stage.SCORING,
        "score_panel_daily": Stage.SCORING,
        "target_weights": Stage.PORTFOLIO,
        "backtest_daily": Stage.BACKTEST,
        "execution_log": Stage.EXECUTION,
        "attribution": Stage.ATTRIBUTION,
        "benchmark_factor_returns": Stage.BENCHMARK,
    }
    downstream = frozenset({
        "factor_covariance", "specific_risk", "security_covariance", "alpha_forecast",
        "stock_score", "score_panel_daily", "target_weights", "backtest_daily",
        "execution_log", "attribution",
    })
    labels = frozenset({"realized_returns", "forward_labels"})
    return BoundaryPolicy(
        producer_stage_by_artifact_type=producers,
        required_input_types_by_output_type={
            "risk_exposure_matrix": frozenset({"base_matrix", "tradable_universe"}),
            "exposure_matrix": frozenset({
                "base_matrix", "tradable_universe", "risk_exposure_matrix", "risk_factor_set",
            }),
            "forward_labels": frozenset({"realized_returns", "tradable_universe"}),
            "factor_predictivity": frozenset({
                "exposure_matrix", "forward_labels", "tradable_universe",
            }),
            "fm_regression": frozenset({
                "exposure_matrix", "forward_labels", "tradable_universe",
            }),
            "factor_evaluation_snapshot": frozenset({"factor_predictivity", "fm_regression"}),
            # The evaluation panel deliberately reads risk_exposure_matrix rather
            # than exposure_matrix. exposure_matrix requires a frozen
            # risk_factor_set, which does not exist before G6a, and a probe
            # evaluation has no business claiming a frozen basis. Consumers must
            # record risk_set_version as a candidate. After G6a the first real
            # alpha switches to exposure_matrix, which is where alpha and risk
            # exposures share one aligned source.
            "factor_evaluation_panel": frozenset({
                "risk_exposure_matrix", "forward_labels", "tradable_universe",
                "probe_signal_panel",
            }),
            "probe_signal_panel": frozenset({"realized_returns", "tradable_universe"}),
            "factor_returns": frozenset({
                "realized_returns", "tradable_universe",
            }),
            "specific_returns": frozenset({
                "realized_returns", "tradable_universe",
            }),
            "factor_regression_quality": frozenset({
                "realized_returns", "tradable_universe",
            }),
            "cross_section_stats": frozenset({"realized_returns", "tradable_universe"}),
            "market_return_dispersion": frozenset({"realized_returns"}),
            "risk_factor_set": frozenset({
                "factor_regression_quality", "bias_test_report",
                "residual_correlation_report",
            }),
            "factor_covariance": frozenset({"factor_returns"}),
            "specific_risk": frozenset({"specific_returns"}),
            "security_covariance": frozenset({
                "risk_exposure_matrix", "factor_covariance", "specific_risk",
            }),
            "bias_test_report": frozenset({"factor_covariance", "specific_risk"}),
            "residual_correlation_report": frozenset({
                "specific_returns", "tradable_universe",
            }),
            "cross_chain_validation": frozenset({
                "factor_evaluation_snapshot", "factor_returns",
            }),
            "alpha_forecast": frozenset({
                "exposure_matrix", "factor_evaluation_snapshot", "cross_section_stats",
            }),
            "cost_estimate": frozenset({"liquidity_facts", "tradable_universe"}),
            "stock_score": frozenset({"alpha_forecast", "cost_estimate"}),
            "score_panel_daily": frozenset({"stock_score", "tradable_universe"}),
            "target_weights": frozenset({"stock_score", "cost_estimate", "tradable_universe"}),
            "backtest_daily": frozenset({"target_weights", "cost_estimate", "tradable_universe"}),
            "execution_log": frozenset({"target_weights", "cost_estimate", "tradable_universe"}),
            "attribution": frozenset({"exposure_matrix", "factor_returns", "specific_returns"}),
        },
        forbidden_input_types_by_stage={
            Stage.UNIVERSE: labels | downstream,
            Stage.EXPOSURE: labels | downstream,
            Stage.L2_PREDICTIVITY: downstream | frozenset({"factor_returns", "specific_returns"}),
            Stage.L2_RETURN_DECOMPOSITION: downstream | frozenset({
                "forward_labels", "factor_predictivity", "fm_regression",
                "factor_evaluation_snapshot",
            }),
            Stage.L2_RISK_MODEL: labels | frozenset({
                "factor_predictivity", "fm_regression", "factor_evaluation_snapshot",
                "stock_score", "score_panel_daily", "target_weights",
            }),
            Stage.L2_VALIDATOR: labels | downstream,
            Stage.ALPHA: labels | frozenset({"stock_score", "score_panel_daily", "target_weights"}),
            Stage.COST: labels,
            Stage.SCORING: labels,
            Stage.PORTFOLIO: labels,
            Stage.BACKTEST: labels,
            Stage.EXECUTION: labels,
            Stage.ATTRIBUTION: labels,
            Stage.BENCHMARK: labels,
        },
        adoption_operations=ADOPTION_OPERATIONS,
    )
