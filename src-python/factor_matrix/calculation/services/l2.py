from ..core.contracts import Stage
from ..core.executor import CalculationExecutor
from ..core.registry import CalculationRegistry
from .base import StageExecutionService


class PredictivityExecutionService(StageExecutionService):
    def __init__(self, registry: CalculationRegistry, executor: CalculationExecutor) -> None:
        super().__init__(Stage.L2_PREDICTIVITY, registry, executor)


class ReturnDecompositionExecutionService(StageExecutionService):
    def __init__(self, registry: CalculationRegistry, executor: CalculationExecutor) -> None:
        super().__init__(Stage.L2_RETURN_DECOMPOSITION, registry, executor)
