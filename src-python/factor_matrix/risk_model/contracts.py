from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping


@dataclass(frozen=True)
class RiskSetMember:
    factor_id: str
    factor_version: int


@dataclass(frozen=True)
class BlockedRiskCandidate:
    factor_id: str
    factor_version: int
    reason: str

    def __post_init__(self) -> None:
        if not self.factor_id or self.factor_version < 1 or not self.reason.strip():
            raise ValueError("BLOCKED_RISK_CANDIDATE_INVALID")


@dataclass(frozen=True)
class RiskFactorSetSpec:
    risk_set_version: int
    status: str
    members: tuple[RiskSetMember, ...]
    baseline_members: tuple[str, ...]
    candidate_selection_order: tuple[str, ...]
    blocked_candidates: tuple[BlockedRiskCandidate, ...]
    weight_model_id: str
    selection_test_ids: tuple[str, ...]
    selection_parameters: Mapping[str, object]
    validation_artifact_run_id: str | None
    bias_test_run_id: str | None
    residual_correlation_run_id: str | None
    known_residuals: tuple[str, ...]
    frozen_at: date | None
    rationale: str
    purpose: str = "production_baseline"
    parent_version: int | None = None
    expansion_manifest_sha: str | None = None

    def __post_init__(self) -> None:
        if self.risk_set_version < 1:
            raise ValueError("RISK_SET_VERSION_INVALID")
        if self.status not in {"candidate", "frozen", "retired"}:
            raise ValueError("RISK_FACTOR_SET_STATUS_INVALID")
        if self.purpose not in {"production_baseline", "holdout_validation", "research"}:
            raise ValueError("RISK_FACTOR_SET_PURPOSE_INVALID")
        if self.parent_version == self.risk_set_version:
            raise ValueError("RISK_FACTOR_SET_SELF_PARENT_FORBIDDEN")
        keys = [(member.factor_id, member.factor_version) for member in self.members]
        if len(keys) != len(set(keys)):
            raise ValueError("RISK_FACTOR_SET_DUPLICATE_MEMBER")
        member_ids = tuple(member.factor_id for member in self.members)
        if not set(self.baseline_members) <= set(member_ids):
            raise ValueError("RISK_BASELINE_MEMBER_NOT_IN_SET")
        candidates = set(member_ids) - set(self.baseline_members)
        if set(self.candidate_selection_order) != candidates:
            raise ValueError("RISK_CANDIDATE_SELECTION_ORDER_MUST_COVER_ACTIVE_CANDIDATES")
        blocked_ids = {item.factor_id for item in self.blocked_candidates}
        if blocked_ids & set(member_ids):
            raise ValueError("BLOCKED_RISK_CANDIDATE_CANNOT_BE_ACTIVE_MEMBER")
        if self.status == "frozen" and (
            self.frozen_at is None or not self.validation_artifact_run_id
            or not self.bias_test_run_id or not self.residual_correlation_run_id
        ):
            raise ValueError("FROZEN_RISK_FACTOR_SET_REQUIRES_EVIDENCE")
        if self.status == "frozen" and (
            self.purpose in {"production_baseline", "holdout_validation"}
            and (self.expansion_manifest_sha is None or len(self.expansion_manifest_sha) != 64)
        ):
            raise ValueError("FROZEN_RISK_FACTOR_SET_EXPANSION_MANIFEST_REQUIRED")
