from ..core.contracts import Stage
from ..core.executor import CalculationExecutor
from ..core.registry import CalculationRegistry
from .base import StageExecutionService


class FactorExecutionService(StageExecutionService):
    def __init__(self, registry: CalculationRegistry, executor: CalculationExecutor) -> None:
        super().__init__(Stage.EXPOSURE, registry, executor)
