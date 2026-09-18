"""Versioned contract binding exposure geometry to regression base weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .storage import file_sha256, json_hash


@dataclass(frozen=True)
class WeightMetricContract:
    contract_id: str
    regression_base_scheme_id: str
    orthogonalization_scheme_id: str
    binding: str
    artifact_path: str
    artifact_sha256: str
    missing_date_fallback_scheme_id: str
    missing_asset_fallback_scheme_id: str

    @property
    def is_bound(self) -> bool:
        return self.binding == "same_pit_weight_artifact"

    @property
    def risk_metric_id(self) -> str:
        return f"risk_metric_{json_hash(self.as_identity())[:16]}"

    def as_identity(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "regression_base_scheme_id": self.regression_base_scheme_id,
            "orthogonalization_scheme_id": self.orthogonalization_scheme_id,
            "binding": self.binding,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "missing_date_fallback_scheme_id": self.missing_date_fallback_scheme_id,
            "missing_asset_fallback_scheme_id": self.missing_asset_fallback_scheme_id,
        }

    def resolve_artifact(self, lake_root: Path) -> Path:
        path = lake_root / self.artifact_path
        if not path.is_file():
            raise RuntimeError("WEIGHT_METRIC_ARTIFACT_MISSING")
        if file_sha256(path) != self.artifact_sha256:
            raise RuntimeError("WEIGHT_METRIC_ARTIFACT_HASH_MISMATCH")
        return path


def load_weight_metric_contract(path: Path) -> WeightMetricContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    regression = payload["regression_base_weight"]
    orthogonalization = payload["exposure_orthogonalization_weight"]
    contract = WeightMetricContract(
        contract_id=payload["contract_id"],
        regression_base_scheme_id=regression["scheme_id"],
        orthogonalization_scheme_id=orthogonalization["scheme_id"],
        binding=payload["binding"],
        artifact_path=regression["artifact_path"],
        artifact_sha256=regression["artifact_sha256"],
        missing_date_fallback_scheme_id=regression["missing_date_fallback_scheme_id"],
        missing_asset_fallback_scheme_id=regression["missing_asset_fallback_scheme_id"],
    )
    if not contract.is_bound:
        raise ValueError("WEIGHT_METRIC_FORMAL_BINDING_REQUIRED")
    if contract.regression_base_scheme_id != contract.orthogonalization_scheme_id:
        raise ValueError("WEIGHT_METRIC_BOUND_SCHEME_MISMATCH")
    if orthogonalization.get("source") != "regression_base_weight.artifact":
        raise ValueError("WEIGHT_METRIC_ORTHOGONALIZATION_SOURCE_INVALID")
    return contract
