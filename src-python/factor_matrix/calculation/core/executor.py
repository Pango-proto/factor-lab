from __future__ import annotations

import json

from .contracts import CalculationRequest, QualityStatus, ScopeMode
from .ports import ArtifactLoader, ArtifactPublisher
from .policy import BoundaryPolicy
from .registry import CalculationRegistry


class CalculationExecutor:
    def __init__(
        self,
        registry: CalculationRegistry,
        loader: ArtifactLoader,
        publisher: ArtifactPublisher,
        policy: BoundaryPolicy,
    ) -> None:
        self._registry = registry
        self._loader = loader
        self._publisher = publisher
        self._policy = policy

    def execute(self, request: CalculationRequest):
        plugin = self._registry.get(request.operation_id, request.operation_version)
        spec = plugin.spec
        self._policy.validate_spec(spec)
        parameters = spec.parameters.validate(request.parameters)
        if spec.scope_mode is ScopeMode.PER_BOARD and not request.scope_key.startswith("board:"):
            raise ValueError("CALCULATION_BOARD_SCOPE_REQUIRED")
        expected = sorted(spec.input_artifact_versions.items())
        observed = sorted(
            (reference.key.artifact_type, reference.key.version)
            for reference in request.inputs
        )
        if observed != expected:
            raise ValueError(f"CALCULATION_INPUT_MISMATCH expected={expected} observed={observed}")
        self._policy.validate_inputs(spec, request.inputs)
        for reference in request.inputs:
            if reference.quality_status is not QualityStatus.PASSED:
                raise RuntimeError(f"CALCULATION_INPUT_QUALITY_FAILED run={reference.key.run_id}")
            if "benchmark_only" in reference.tags and not spec.accepts_benchmark_inputs:
                raise RuntimeError("BENCHMARK_ONLY_INPUT_REJECTED")
            if (
                reference.key.artifact_type == "risk_factor_set"
                and "frozen" not in reference.tags
            ):
                raise RuntimeError("FROZEN_RISK_FACTOR_SET_REQUIRED")
        normalized = CalculationRequest(
            operation_id=request.operation_id,
            operation_version=request.operation_version,
            as_of_date=request.as_of_date,
            scope_key=request.scope_key,
            parameters=parameters,
            inputs=request.inputs,
        )
        result = plugin.calculate(
            normalized,
            tuple(self._loader.load(reference) for reference in request.inputs),
        )
        output_types = tuple(output.artifact_type for output in result.outputs)
        if sorted(output_types) != sorted(spec.output_artifact_types):
            raise RuntimeError("CALCULATION_OUTPUT_CONTRACT_VIOLATION")
        if result.quality.status is not QualityStatus.PASSED:
            failed_checks = [
                {
                    "check_id": check.check_id,
                    "detail": check.detail,
                    "observed": check.observed,
                }
                for check in result.quality.checks
                if not check.passed
            ]
            raise RuntimeError(
                "CALCULATION_QUALITY_GATE_FAILED "
                + json.dumps({
                    "operation_id": request.operation_id,
                    "scope_key": request.scope_key,
                    "as_of_date": request.as_of_date.isoformat(),
                    "failed_checks": failed_checks,
                }, ensure_ascii=False)
            )
        return self._publisher.publish(spec, normalized, result)
