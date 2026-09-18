from __future__ import annotations

from typing import Protocol

from .contracts import (
    ArtifactRef,
    CalculationRequest,
    CalculationResult,
    LoadedArtifact,
    OperationSpec,
)


class ArtifactLoader(Protocol):
    def load(self, reference: ArtifactRef) -> LoadedArtifact: ...


class ArtifactPublisher(Protocol):
    def publish(
        self,
        spec: OperationSpec,
        request: CalculationRequest,
        result: CalculationResult,
    ) -> tuple[ArtifactRef, ...]: ...
