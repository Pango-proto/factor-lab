from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool
    semantic_role: str
    unit: str


@dataclass(frozen=True)
class ArtifactSchema:
    artifact_type: str
    version: str
    primary_key: tuple[str, ...]
    partition_keys: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    dynamic_column_prefixes: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"ARTIFACT_SCHEMA_DUPLICATE_COLUMN type={self.artifact_type}")
        if not set(self.primary_key).issubset(names):
            raise ValueError(f"ARTIFACT_SCHEMA_PRIMARY_KEY_UNKNOWN type={self.artifact_type}")
        if not set(self.partition_keys).issubset(names):
            raise ValueError(f"ARTIFACT_SCHEMA_PARTITION_KEY_UNKNOWN type={self.artifact_type}")


class ArtifactSchemaRegistry:
    def __init__(self, schemas: Iterable[ArtifactSchema] = ()) -> None:
        self._schemas: dict[tuple[str, str], ArtifactSchema] = {}
        for schema in schemas:
            self.register(schema)

    def register(self, schema: ArtifactSchema) -> None:
        key = (schema.artifact_type, schema.version)
        if key in self._schemas:
            raise ValueError(f"ARTIFACT_SCHEMA_DUPLICATE type={key[0]} version={key[1]}")
        self._schemas[key] = schema

    def get(self, artifact_type: str, version: str) -> ArtifactSchema:
        try:
            return self._schemas[(artifact_type, version)]
        except KeyError as exc:
            raise KeyError(f"ARTIFACT_SCHEMA_UNKNOWN type={artifact_type} version={version}") from exc

    def search(self, query: str = "") -> tuple[ArtifactSchema, ...]:
        needle = query.strip().lower()
        return tuple(sorted(
            (
                schema for schema in self._schemas.values()
                if not needle or needle in " ".join(
                    (schema.artifact_type, schema.version, schema.description)
                ).lower()
            ),
            key=lambda schema: (schema.artifact_type, schema.version),
        ))
