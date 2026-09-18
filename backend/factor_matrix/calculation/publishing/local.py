from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now
from ..core.contracts import (
    ArtifactKey,
    ArtifactRef,
    CalculationRequest,
    CalculationResult,
    LoadedArtifact,
    OperationSpec,
)


def _write_immutable_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    if path.exists():
        if file_sha256(path) != file_sha256(temporary):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"IMMUTABLE_ARTIFACT_CONFLICT path={path}")
        temporary.unlink(missing_ok=True)
        return
    os.replace(temporary, path)


class LocalArtifactStore:
    """Filesystem adapter for both artifact loading and immutable publication."""

    def __init__(self, lake: DataLake) -> None:
        self._lake = lake

    def _resolve(self, uri: str) -> Path:
        path = Path(uri)
        return path if path.is_absolute() else self._lake.root / path

    def load(self, reference: ArtifactRef) -> LoadedArtifact:
        metadata_path = self._resolve(reference.uri)
        if file_sha256(metadata_path) != reference.checksum:
            raise RuntimeError(f"ARTIFACT_CHECKSUM_MISMATCH run={reference.key.run_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tables: dict[str, pl.DataFrame] = {}
        for table_id, record in metadata["tables"].items():
            table_path = self._resolve(record["uri"])
            if file_sha256(table_path) != record["checksum"]:
                raise RuntimeError(
                    f"ARTIFACT_TABLE_CHECKSUM_MISMATCH run={reference.key.run_id} table={table_id}"
                )
            tables[table_id] = pl.read_parquet(table_path)
        return LoadedArtifact(reference=reference, tables=tables)

    def publish(
        self,
        spec: OperationSpec,
        request: CalculationRequest,
        result: CalculationResult,
    ) -> tuple[ArtifactRef, ...]:
        definition = {
            "operation_id": spec.operation_id,
            "operation_version": spec.version,
            "as_of_date": request.as_of_date.isoformat(),
            "scope_key": request.scope_key,
            "parameters": dict(request.parameters),
            "inputs": [{"run_id": reference.key.run_id, "checksum": reference.checksum}
                       for reference in request.inputs],
            "code_hash": source_tree_hash(),
        }
        run_id = f"calc_{spec.operation_id}_{json_hash(definition)[:12]}"
        run_dir = (
            self._lake.root / "gold" / "calculations"
            / f"stage={spec.stage.value}" / f"operation={spec.operation_id}" / f"run_id={run_id}"
        )
        manifest_outputs: dict[str, Path] = {}
        references: list[ArtifactRef] = []
        for output in result.outputs:
            artifact_dir = run_dir / f"artifact={output.artifact_type}"
            table_records: dict[str, dict[str, str]] = {}
            for table_id, frame in sorted(output.tables.items()):
                table_path = artifact_dir / f"{table_id}.parquet"
                _write_immutable_parquet(table_path, frame)
                table_records[table_id] = {
                    "uri": str(table_path.relative_to(self._lake.root)),
                    "checksum": file_sha256(table_path),
                }
                manifest_outputs[f"{output.artifact_type}.{table_id}"] = table_path
            metadata = {
                "schema_version": 1,
                "run_id": run_id,
                "artifact_type": output.artifact_type,
                "artifact_version": spec.output_artifact_versions[output.artifact_type],
                "operation_id": spec.operation_id,
                "operation_version": spec.version,
                "stage": spec.stage.value,
                "as_of_date": request.as_of_date.isoformat(),
                "scope_key": request.scope_key,
                "parameters": dict(request.parameters),
                "tags": sorted(output.tags),
                "metadata": dict(output.metadata),
                "quality_status": result.quality.status.value,
                "tables": table_records,
                "parent_artifacts": definition["inputs"],
            }
            metadata_path = self._lake.write_immutable_json(
                artifact_dir / "metadata.json", metadata
            )
            manifest_outputs[f"{output.artifact_type}.metadata"] = metadata_path
            references.append(ArtifactRef(
                key=ArtifactKey(
                    artifact_type=output.artifact_type,
                    version=spec.output_artifact_versions[output.artifact_type],
                    run_id=run_id,
                    as_of_date=request.as_of_date,
                    scope_key=request.scope_key,
                ),
                uri=str(metadata_path.relative_to(self._lake.root)),
                checksum=file_sha256(metadata_path),
                quality_status=result.quality.status,
                tags=output.tags,
            ))
        quality_path = self._lake.write_immutable_json(
            run_dir / "quality.json",
            {
                "status": result.quality.status.value,
                "checks": [
                    {
                        "check_id": check.check_id,
                        "passed": check.passed,
                        "detail": check.detail,
                        "observed": check.observed,
                    }
                    for check in result.quality.checks
                ],
                "metadata": dict(result.quality.metadata),
                "metrics": dict(result.metrics),
            },
        )
        manifest_outputs["quality"] = quality_path
        input_paths = {
            f"input_{index}": self._resolve(reference.uri)
            for index, reference in enumerate(request.inputs)
        }
        self._lake.register_calculation(
            run_id=run_id,
            job=spec.operation_id,
            as_of=request.as_of_date,
            mode="calculation_plugin",
            config={
                "operation_version": spec.version,
                "stage": spec.stage.value,
                "scope_key": request.scope_key,
                "parameters": dict(request.parameters),
            },
            config_hash=json_hash({
                "operation_version": spec.version,
                "scope_key": request.scope_key,
                "parameters": dict(request.parameters),
            }),
            code_hash=definition["code_hash"],
            parent_run_ids=[reference.key.run_id for reference in request.inputs],
            inputs=input_paths,
            outputs=manifest_outputs,
            quality_gate={
                "status": result.quality.status.value,
                "quality_path": str(quality_path.relative_to(self._lake.root)),
            },
            extra={"artifact_types": [reference.key.artifact_type for reference in references]},
        )
        return tuple(references)
