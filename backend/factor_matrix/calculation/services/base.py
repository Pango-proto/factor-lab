from __future__ import annotations

from ..core.contracts import CalculationRequest, Stage
from ..core.executor import CalculationExecutor
from ..core.registry import CalculationRegistry


class StageExecutionService:
    def __init__(
        self,
        stage: Stage,
        registry: CalculationRegistry,
        executor: CalculationExecutor,
    ) -> None:
        self._stage = stage
        self._registry = registry
        self._executor = executor

    def execute(self, request: CalculationRequest):
        spec = self._registry.get(request.operation_id, request.operation_version).spec
        if spec.stage is not self._stage:
            raise ValueError(
                f"CALCULATION_SERVICE_STAGE_MISMATCH expected={self._stage.value} "
                f"observed={spec.stage.value}"
            )
        return self._executor.execute(request)
