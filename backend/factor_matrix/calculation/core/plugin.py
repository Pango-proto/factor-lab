from __future__ import annotations

from typing import Protocol

from .contracts import CalculationRequest, CalculationResult, LoadedArtifact, OperationSpec


class CalculationPlugin(Protocol):
    spec: OperationSpec

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult: ...
