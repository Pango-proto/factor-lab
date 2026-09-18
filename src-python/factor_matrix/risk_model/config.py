from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

from ..factor_engine import FactorRegistry, FactorRole, FactorStatus
from .contracts import BlockedRiskCandidate, RiskFactorSetSpec, RiskSetMember


def load_risk_factor_set(path: Path, registry: FactorRegistry) -> RiskFactorSetSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expansion_path = payload.get("expansion_manifest")
    expansion_sha = payload.get("expansion_manifest_sha")
    if expansion_path is not None or expansion_sha is not None:
        if not isinstance(expansion_path, str) or not isinstance(expansion_sha, str):
            raise ValueError("RISK_SET_EXPANSION_LINEAGE_INCOMPLETE")
        resolved_expansion = (path.parent / expansion_path).resolve()
        if not resolved_expansion.exists() or len(expansion_sha) != 64:
            raise ValueError("RISK_SET_EXPANSION_MANIFEST_MISSING")
        actual_sha = hashlib.sha256(resolved_expansion.read_bytes()).hexdigest()
        if actual_sha != expansion_sha:
            raise ValueError("RISK_SET_EXPANSION_MANIFEST_HASH_MISMATCH")
        expansion = json.loads(resolved_expansion.read_text(encoding="utf-8"))
        generated = [
            column
            for item in expansion.get("logical_members", ())
            for column in item.get("generated_columns", ())
        ]
        if len(generated) != int(expansion.get("expanded_factor_count", -1)):
            raise ValueError("RISK_SET_EXPANSION_K_MISMATCH")
        if len(generated) != len(set(generated)):
            raise ValueError("RISK_SET_EXPANSION_DUPLICATE_COLUMN")
    result = RiskFactorSetSpec(
        risk_set_version=payload["risk_set_version"],
        status=payload["status"],
        members=tuple(
            RiskSetMember(item["factor_id"], item["factor_version"])
            for item in payload["members"]
        ),
        baseline_members=tuple(payload["baseline_members"]),
        candidate_selection_order=tuple(payload["candidate_selection_order"]),
        blocked_candidates=tuple(
            BlockedRiskCandidate(
                item["factor_id"], item["factor_version"], item["reason"]
            )
            for item in payload.get("blocked_candidates", ())
        ),
        weight_model_id=payload["weight_model_id"],
        selection_test_ids=tuple(payload["selection_test_ids"]),
        selection_parameters=payload["selection_parameters"],
        validation_artifact_run_id=payload.get("validation_artifact_run_id"),
        bias_test_run_id=payload.get("bias_test_run_id"),
        residual_correlation_run_id=payload.get("residual_correlation_run_id"),
        known_residuals=tuple(payload.get("known_residuals", ())),
        frozen_at=(date.fromisoformat(payload["frozen_at"]) if payload.get("frozen_at") else None),
        rationale=payload["rationale"],
        purpose=payload.get("purpose", "production_baseline"),
        parent_version=payload.get("parent_version"),
        expansion_manifest_sha=payload.get("expansion_manifest_sha"),
    )
    for member in result.members:
        spec = registry.get(member.factor_id, member.factor_version).spec
        if spec.role in {FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR}:
            raise ValueError(f"RISK_SET_NON_BASIS_MEMBER_FORBIDDEN factor={member.factor_id}")
        if spec.status is FactorStatus.BLOCKED:
            raise ValueError(f"RISK_SET_BLOCKED_MEMBER_FORBIDDEN factor={member.factor_id}")
    for blocked in result.blocked_candidates:
        spec = registry.get(blocked.factor_id, blocked.factor_version).spec
        if spec.status is not FactorStatus.BLOCKED:
            raise ValueError(
                f"RISK_BLOCKED_CANDIDATE_STATUS_MISMATCH factor={blocked.factor_id}"
            )
    registry.ordered_risk_factors(tuple(member.factor_id for member in result.members))
    return result
