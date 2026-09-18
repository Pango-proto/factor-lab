from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

import polars as pl


class Stage(StrEnum):
    UNIVERSE = "universe"
    EXPOSURE = "exposure"
    L2_PREDICTIVITY = "l2_predictivity"
    L2_RETURN_DECOMPOSITION = "l2_return_decomposition"
    L2_RISK_MODEL = "l2c_risk_model"
    L2_VALIDATOR = "l2_validator"
    ALPHA = "alpha"
    COST = "cost"
    SCORING = "scoring"
    PORTFOLIO = "portfolio"
    BACKTEST = "backtest"
    EXECUTION = "execution"
    ATTRIBUTION = "attribution"
    BENCHMARK = "benchmark"


class ScopeMode(StrEnum):
    PER_BOARD = "per_board"
    UNIFIED_WITH_BOARD = "unified_with_board"
    CONFIGURABLE = "configurable"


class QualityStatus(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


class ParameterType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING_LIST = "string_list"
    INTEGER_LIST = "integer_list"
    NUMBER_LIST = "number_list"
    MAPPING = "mapping"


def _valid_parameter_type(value: object, type_id: ParameterType) -> bool:
    if type_id is ParameterType.STRING:
        return isinstance(value, str)
    if type_id is ParameterType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if type_id is ParameterType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_id is ParameterType.BOOLEAN:
        return isinstance(value, bool)
    if type_id is ParameterType.STRING_LIST:
        return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)
    if type_id is ParameterType.INTEGER_LIST:
        return isinstance(value, (list, tuple)) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    if type_id is ParameterType.NUMBER_LIST:
        return isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
    return isinstance(value, Mapping)


@dataclass(frozen=True)
class ParameterField:
    name: str
    type_id: ParameterType
    required: bool
    description: str
    allowed_values: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: object) -> None:
        if not _valid_parameter_type(value, self.type_id):
            raise ValueError(f"PARAMETER_TYPE_INVALID name={self.name} expected={self.type_id}")
        if self.allowed_values and value not in self.allowed_values:
            raise ValueError(f"PARAMETER_VALUE_UNSUPPORTED name={self.name}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"PARAMETER_BELOW_MINIMUM name={self.name}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"PARAMETER_ABOVE_MAXIMUM name={self.name}")


@dataclass(frozen=True)
class ParameterSchema:
    fields: tuple[ParameterField, ...]
    allow_extra: bool

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("PARAMETER_SCHEMA_DUPLICATE_FIELD")

    def validate(self, values: Mapping[str, object]) -> Mapping[str, object]:
        definitions = {field.name: field for field in self.fields}
        missing = sorted(
            field.name for field in self.fields if field.required and field.name not in values
        )
        if missing:
            raise ValueError(f"PARAMETER_REQUIRED missing={','.join(missing)}")
        extra = sorted(set(values) - set(definitions))
        if extra and not self.allow_extra:
            raise ValueError(f"PARAMETER_UNKNOWN names={','.join(extra)}")
        for name, value in values.items():
            if name in definitions:
                definitions[name].validate(value)
        return MappingProxyType(dict(values))


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    version: str
    stage: Stage
    input_artifact_types: tuple[str, ...]
    input_artifact_versions: Mapping[str, str]
    output_artifact_types: tuple[str, ...]
    output_artifact_versions: Mapping[str, str]
    parameters: ParameterSchema
    scope_mode: ScopeMode
    accepts_benchmark_inputs: bool
    description: str

    def __post_init__(self) -> None:
        if not self.operation_id or any(char.isspace() for char in self.operation_id):
            raise ValueError("OPERATION_ID_INVALID")
        if not self.version:
            raise ValueError("OPERATION_VERSION_REQUIRED")
        if not self.output_artifact_types:
            raise ValueError("OPERATION_OUTPUT_REQUIRED")
        if len(self.output_artifact_types) != len(set(self.output_artifact_types)):
            raise ValueError("OPERATION_OUTPUT_DUPLICATE")
        if set(self.input_artifact_versions) != set(self.input_artifact_types):
            raise ValueError("OPERATION_INPUT_VERSION_CONTRACT_INVALID")
        if set(self.output_artifact_versions) != set(self.output_artifact_types):
            raise ValueError("OPERATION_OUTPUT_VERSION_CONTRACT_INVALID")
        if not all(self.input_artifact_versions.values()) or not all(
            self.output_artifact_versions.values()
        ):
            raise ValueError("OPERATION_ARTIFACT_VERSION_REQUIRED")


@dataclass(frozen=True)
class ArtifactKey:
    artifact_type: str
    version: str
    run_id: str
    as_of_date: date
    scope_key: str


@dataclass(frozen=True)
class ArtifactRef:
    key: ArtifactKey
    uri: str
    checksum: str
    quality_status: QualityStatus
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LoadedArtifact:
    reference: ArtifactRef
    tables: Mapping[str, pl.DataFrame]


@dataclass(frozen=True)
class CalculationRequest:
    operation_id: str
    operation_version: str
    as_of_date: date
    scope_key: str
    parameters: Mapping[str, object]
    inputs: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class QualityCheck:
    check_id: str
    passed: bool
    detail: str
    observed: object | None = None


@dataclass(frozen=True)
class QualityReport:
    status: QualityStatus
    checks: tuple[QualityCheck, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TableOutput:
    artifact_type: str
    tables: Mapping[str, pl.DataFrame]
    tags: frozenset[str] = frozenset()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CalculationResult:
    outputs: tuple[TableOutput, ...]
    quality: QualityReport
    metrics: Mapping[str, object] = field(default_factory=dict)
