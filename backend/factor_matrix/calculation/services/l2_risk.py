from ..core.contracts import Stage
from ..core.executor import CalculationExecutor
from ..core.registry import CalculationRegistry
from .base import StageExecutionService


class RiskModelExecutionService(StageExecutionService):
    def __init__(self, registry: CalculationRegistry, executor: CalculationExecutor) -> None:
        super().__init__(Stage.L2_RISK_MODEL, registry, executor)


class CrossChainValidationService(StageExecutionService):
    def __init__(self, registry: CalculationRegistry, executor: CalculationExecutor) -> None:
        super().__init__(Stage.L2_VALIDATOR, registry, executor)
