"""Adapters that bring pre-existing gold artifacts into the calculation core.

L1 and the tradable universe were built before anything ran on the executor, so
each grew its own layout, its own manifest shape, and its own idea of what
"current" means. Rewriting those runners now would mean re-materializing 370MB
of exposures for no analytical gain, so these adapters read what is already
published and re-publish it as an addressable artifact instead.

Two layouts exist and both are supported:

  POINTER  gold/<dir>/_CURRENT.json -> manifest -> outputs[table_id].path
           Used by risk_exposure_matrix and l2b_risk_only.

  RUN_DIR  gold/<dir>/artifact_version=1/run_id=<id>/<file>.parquet
           Used by tradable_universe, which publishes no pointer at all.

The RUN_DIR layout carries a real hazard. tradable_universe currently holds
eight runs that split into two scopes, single-date snapshots of 5,543 rows and
full-history builds of 8.35M rows, across three config generations, and nothing
in the directory name distinguishes them. Choosing "the newest" would silently
pick a single-date snapshot and every downstream join would quietly lose 1,729
dates.

So run_id is a required parameter with no default. The caller must say which
run it means. For anything consumed alongside L1 exposures the correct value is
the tradable_universe_run_id recorded on the L1 manifest, which makes exposure
and universe consistent by construction rather than by memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from ...storage import DataLake, file_sha256
from ..core.contracts import (
    CalculationRequest,
    CalculationResult,
    LoadedArtifact,
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


class LegacyLayout(StrEnum):
    POINTER = "pointer"
    RUN_DIR = "run_dir"


# Legacy manifest keys worth preserving verbatim. Anything not listed is still
# recorded under legacy_manifest_keys, so a later reader can tell what was
# available at adoption time even if it was not carried forward.
_LINEAGE_KEYS = (
    "risk_basis_id", "risk_factor_set_id", "risk_factor_set_status",
    "risk_metric_id", "universe_variant", "silver_version_id",
    "tradable_universe_run_id", "board_benchmark_run_id", "registry_snapshot_id",
    "weight_metric_contract", "promotion_gate", "execution_status",
    "data_snapshot_id", "config_hash", "code_hash", "counts", "policy",
    "code_sha", "start", "end",
)

_WINDOW_FIELDS = (
    ParameterField(
        name="start_date", type_id=ParameterType.STRING, required=False,
        description="Optional inclusive lower bound on trade_date, ISO format.",
    ),
    ParameterField(
        name="end_date", type_id=ParameterType.STRING, required=False,
        description="Optional inclusive upper bound on trade_date, ISO format.",
    ),
)

_RUN_ID_FIELD = ParameterField(
    name="legacy_run_id", type_id=ParameterType.STRING, required=True,
    description=(
        "Exact legacy run to adopt. Required, never defaulted: sibling runs "
        "under the same directory may differ in scope or config generation "
        "without saying so in the run identifier."
    ),
)


def _legacy_spec(
    operation_id: str, artifact_type: str, stage: Stage,
    layout: LegacyLayout, description: str,
) -> OperationSpec:
    fields = _WINDOW_FIELDS
    if layout is LegacyLayout.RUN_DIR:
        fields = (_RUN_ID_FIELD,) + fields
    return OperationSpec(
        operation_id=operation_id,
        version="1",
        stage=stage,
        input_artifact_types=(),
        input_artifact_versions={},
        output_artifact_types=(artifact_type,),
        output_artifact_versions={artifact_type: "1"},
        parameters=ParameterSchema(fields=fields, allow_extra=False),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD,
        accepts_benchmark_inputs=False,
        description=description,
    )


@dataclass(frozen=True)
class LegacyArtifactAdapter:
    """Adopt one legacy gold artifact into the calculation core."""

    lake: DataLake
    artifact_type: str
    legacy_directory: str
    table_id: str
    layout: LegacyLayout
    spec: OperationSpec
    legacy_file_stem: str = ""

    def _resolve_pointer(self) -> tuple[Path, dict, str | None]:
        pointer_path = self.lake.root / "gold" / self.legacy_directory / "_CURRENT.json"
        if not pointer_path.exists():
            raise FileNotFoundError(f"LEGACY_CURRENT_POINTER_MISSING path={pointer_path}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = self.lake.root / pointer["manifest"]
        if not manifest_path.exists():
            raise FileNotFoundError(f"LEGACY_MANIFEST_MISSING path={manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", {})
        if self.table_id not in outputs:
            raise ValueError(
                f"LEGACY_TABLE_NOT_IN_MANIFEST table={self.table_id} "
                f"available={sorted(outputs)}"
            )
        record = outputs[self.table_id]
        manifest = dict(manifest)
        manifest["legacy_run_id"] = pointer.get("run_id")
        manifest["legacy_manifest"] = pointer.get("manifest")
        return self.lake.root / record["path"], manifest, record.get("sha256")

    def _resolve_run_dir(self, run_id: str) -> tuple[Path, dict, str | None]:
        run_dir = (
            self.lake.root / "gold" / self.legacy_directory
            / "artifact_version=1" / f"run_id={run_id}"
        )
        if not run_dir.is_dir():
            raise FileNotFoundError(f"LEGACY_RUN_MISSING path={run_dir}")
        metadata_path = run_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists() else {}
        )
        stem = self.legacy_file_stem or self.table_id
        table_path = run_dir / f"{stem}.parquet"
        if not table_path.exists():
            available = sorted(path.name for path in run_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"LEGACY_TABLE_FILE_MISSING expected={table_path.name} "
                f"available={available}"
            )
        metadata = dict(metadata)
        metadata["legacy_run_id"] = run_id
        metadata["legacy_manifest"] = str(metadata_path.relative_to(self.lake.root))
        declared = None
        artifact = metadata.get("artifact")
        if isinstance(artifact, dict):
            declared = artifact.get("sha256")
        return table_path, metadata, declared

    def calculate(
        self,
        request: CalculationRequest,
        inputs: tuple[LoadedArtifact, ...],
    ) -> CalculationResult:
        if self.layout is LegacyLayout.POINTER:
            table_path, manifest, declared_sha = self._resolve_pointer()
        else:
            table_path, manifest, declared_sha = self._resolve_run_dir(
                str(request.parameters["legacy_run_id"])
            )

        observed_sha = file_sha256(table_path)
        if declared_sha and observed_sha != declared_sha:
            raise RuntimeError(
                f"LEGACY_ARTIFACT_CHECKSUM_MISMATCH table={self.table_id} "
                f"run={manifest.get('legacy_run_id')}"
            )

        scan = pl.scan_parquet(table_path)
        start = request.parameters.get("start_date")
        end = request.parameters.get("end_date")
        if start is not None:
            scan = scan.filter(pl.col("trade_date") >= pl.lit(start).str.to_date())
        if end is not None:
            scan = scan.filter(pl.col("trade_date") <= pl.lit(end).str.to_date())
        table = scan.collect()

        lineage = {key: manifest[key] for key in _LINEAGE_KEYS if key in manifest}
        lineage.update({
            "legacy_run_id": manifest.get("legacy_run_id"),
            "legacy_manifest": manifest.get("legacy_manifest"),
            "legacy_layout": self.layout.value,
            "legacy_table_path": str(table_path.relative_to(self.lake.root)),
            "legacy_table_sha256": observed_sha,
            "legacy_checksum_declared": bool(declared_sha),
            "legacy_manifest_keys": sorted(manifest),
            "adoption_window": {"start_date": start, "end_date": end},
            "boundary_operation": True,
        })

        has_trade_date = "trade_date" in table.columns
        checks = (
            QualityCheck(
                check_id="rows_present",
                passed=table.height > 0,
                detail="the adopted window must contain at least one row",
                observed=table.height,
            ),
            QualityCheck(
                check_id="trade_date_present",
                passed=has_trade_date,
                detail="every adopted artifact is keyed on trade_date",
                observed=sorted(table.columns)[:8],
            ),
            QualityCheck(
                check_id="checksum_verified_when_declared",
                passed=True,
                detail=(
                    "the legacy table matched its declared checksum"
                    if declared_sha
                    else "no checksum was declared by the legacy layout; adoption "
                         "recorded the observed digest instead"
                ),
                observed=bool(declared_sha),
            ),
        )
        return CalculationResult(
            outputs=(
                TableOutput(
                    artifact_type=self.artifact_type,
                    tables={self.table_id: table},
                    metadata=lineage,
                ),
            ),
            quality=QualityReport(
                status=(
                    QualityStatus.PASSED
                    if all(check.passed for check in checks)
                    else QualityStatus.FAILED
                ),
                checks=checks,
            ),
            metrics={
                "row_count": table.height,
                "column_count": len(table.columns),
                "date_count": table["trade_date"].n_unique() if has_trade_date else None,
            },
        )


def adopt_risk_exposure_matrix(lake: DataLake) -> LegacyArtifactAdapter:
    return LegacyArtifactAdapter(
        lake=lake,
        artifact_type="risk_exposure_matrix",
        legacy_directory="risk_exposure_matrix",
        table_id="risk_exposure_matrix_v1",
        layout=LegacyLayout.POINTER,
        spec=_legacy_spec(
            operation_id="adopt_risk_exposure_matrix",
            artifact_type="risk_exposure_matrix",
            stage=Stage.EXPOSURE,
            layout=LegacyLayout.POINTER,
            description=(
                "Adopt the published L1 risk-only exposure matrix without "
                "recomputing it. It carries 44 generated risk columns against an "
                "unfrozen candidate risk set, so consumers must record "
                "risk_set_version as a candidate rather than a version number"
            ),
        ),
    )


def adopt_tradable_universe(lake: DataLake) -> LegacyArtifactAdapter:
    return LegacyArtifactAdapter(
        lake=lake,
        artifact_type="tradable_universe",
        legacy_directory="tradable_universe",
        table_id="tradable_universe_v1",
        layout=LegacyLayout.RUN_DIR,
        legacy_file_stem="tradable_universe",
        spec=_legacy_spec(
            operation_id="adopt_tradable_universe",
            artifact_type="tradable_universe",
            stage=Stage.UNIVERSE,
            layout=LegacyLayout.RUN_DIR,
            description=(
                "Adopt one explicitly named tradable universe run. Sibling runs "
                "mix single-date snapshots with full-history builds across "
                "several config generations, so the run must be named by the "
                "caller and should normally come from the L1 manifest's "
                "tradable_universe_run_id"
            ),
        ),
    )
