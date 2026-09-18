"""Composable calculation architecture with no built-in quantitative model."""

from .core.contracts import (
    ArtifactKey,
    ArtifactRef,
    CalculationRequest,
    CalculationResult,
    OperationSpec,
    ParameterField,
    ParameterSchema,
    ParameterType,
    QualityCheck,
    QualityReport,
    QualityStatus,
    ScopeMode,
    Stage,
    TableOutput,
)
from .core.executor import CalculationExecutor
from .core.policy import BoundaryPolicy, architecture_boundary_policy
from .core.registry import CalculationRegistry

__all__ = [
    "ArtifactKey",
    "ArtifactRef",
    "CalculationExecutor",
    "BoundaryPolicy",
    "CalculationRegistry",
    "CalculationRequest",
    "CalculationResult",
    "OperationSpec",
    "ParameterField",
    "ParameterSchema",
    "ParameterType",
    "QualityCheck",
    "QualityReport",
    "QualityStatus",
    "ScopeMode",
    "Stage",
    "TableOutput",
    "architecture_boundary_policy",
]
