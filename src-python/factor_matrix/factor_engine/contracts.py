from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Mapping


class FactorFamily(StrEnum):
    """Legacy in-memory adapter; never persist this as feature identity."""
    RISK = "risk"
    ALPHA = "alpha"


class FactorRole(StrEnum):
    COUNTRY = "country"
    INDUSTRY = "industry"
    BOARD = "board"
    MEMBERSHIP = "membership"
    STYLE_RISK = "style_risk"
    ALPHA_CANDIDATE = "alpha_candidate"
    DESCRIPTOR = "descriptor"
    PROBE = "probe"


class FeatureRole(StrEnum):
    """Explicit assignment of a feature's research/use role."""

    UNASSIGNED = "unassigned"
    BASIS = "basis"
    BETTABLE = "bettable"
    PROBE = "probe"


class FactorStatus(StrEnum):
    DRAFT = "draft"
    TESTING = "testing"
    FROZEN = "frozen"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    INVALIDATED = "invalidated"
    BLOCKED = "blocked"


_ALLOWED_L2A_GROUPS = frozenset({
    "market_cap_quantile", "liquidity_quantile", "board", "market_volatility_regime",
})

REQUIRED_ALPHA_NEUTRALIZATION_DECISIONS = frozenset({
    "beta", "residual_volatility", "liquidity", "listing_age",
    "statistical_factors",
})


def canonical_feature_key(
    *, formula_expr: str | None, source_tables: tuple[str, ...],
    source_fields: tuple[str, ...], params: Mapping[str, object], pit_key: str,
) -> str:
    # Research protocol fields describe how a feature is evaluated, not what
    # the feature is. They are excluded from feature identity so a new
    # risk_set_version or horizon creates a new assertion, not a new feature.
    construction_params = {
        key: value for key, value in params.items()
        if key not in {
            "evaluation_horizon_days", "preregistered_groups",
            "neutralization_decisions", "neutralization_comparison",
        }
    }
    payload = {
        "formula_expr": " ".join((formula_expr or "").split()),
        "source_tables": sorted(source_tables),
        "source_fields": sorted(source_fields),
        "params": construction_params,
        "pit_key": pit_key,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def code_file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    version: int
    family: FactorFamily
    role: FactorRole
    display_name: str
    feature_key: str
    formula_expr: str | None
    source_tables: tuple[str, ...]
    source_fields: tuple[str, ...]
    params: Mapping[str, object]
    lookback_days: int | None
    min_obs: int | None
    pit_key: str
    winsorize_method: str | None
    winsorize_k: float | None
    standardize: str | None
    weight_scheme: str | None
    neutralize_against: str | tuple[str, ...] | None
    orthogonalize_after: tuple[str, ...]
    direction: int | None
    missing_policy: str | None
    coverage_min: float | None
    depends_on: tuple[str, ...]
    status: FactorStatus
    proposed_date: date
    frozen_date: date | None
    status_changed_date: date | None
    status_reason: str
    variant_count: int
    family_root_id: str
    holdout_touched: bool
    research_log_refs: tuple[str, ...]
    hypothesis: str
    reference: str | None
    risk_set_version: int | None
    code_path: str
    code_sha: str
    owner: str | None

    def __post_init__(self) -> None:
        if not self.factor_id or any(char.isspace() for char in self.factor_id):
            raise ValueError("FACTOR_ID_INVALID")
        if self.version < 1:
            raise ValueError("FACTOR_VERSION_INVALID")
        if not self.feature_key:
            raise ValueError("FACTOR_FEATURE_KEY_REQUIRED")
        if (
            (not self.formula_expr and not self.params.get("derived_from_run_id"))
            or not self.source_tables or not self.pit_key
        ):
            raise ValueError("FACTOR_DEFINITION_INCOMPLETE")
        if self.variant_count < 1 or not self.family_root_id:
            raise ValueError("FACTOR_VARIANT_ACCOUNTING_REQUIRED")
        if not self.hypothesis or not self.code_path or not self.code_sha:
            raise ValueError("FACTOR_REPRODUCIBILITY_FIELDS_REQUIRED")
        if self.coverage_min is not None and not 0 <= self.coverage_min <= 1:
            raise ValueError("FACTOR_COVERAGE_MIN_INVALID")
        risk_roles = {
            FactorRole.COUNTRY, FactorRole.INDUSTRY, FactorRole.BOARD,
            FactorRole.MEMBERSHIP, FactorRole.STYLE_RISK,
        }
        if self.role in risk_roles:
            if self.role not in risk_roles or self.direction is not None:
                raise ValueError("RISK_FACTOR_ROLE_OR_DIRECTION_INVALID")
            if self.neutralize_against == "ALL_RISK" or self.risk_set_version is not None:
                raise ValueError("RISK_FACTOR_ALPHA_FIELDS_FORBIDDEN")
        elif self.role in {FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR}:
            if self.role not in {FactorRole.ALPHA_CANDIDATE, FactorRole.DESCRIPTOR}:
                raise ValueError("ALPHA_FACTOR_ROLE_INVALID")
            if self.role is FactorRole.ALPHA_CANDIDATE and self.direction not in (-1, 1):
                raise ValueError("ALPHA_FACTOR_DIRECTION_REQUIRED")
            if self.role is FactorRole.ALPHA_CANDIDATE:
                horizon = self.params.get("evaluation_horizon_days")
                groups = self.params.get("preregistered_groups")
                if not isinstance(horizon, int) or horizon < 1:
                    raise ValueError("ALPHA_EVALUATION_HORIZON_PREREGISTRATION_REQUIRED")
                if not isinstance(groups, (list, tuple)) or not set(groups) <= _ALLOWED_L2A_GROUPS:
                    raise ValueError("ALPHA_EVALUATION_GROUPS_PREREGISTRATION_REQUIRED")
                decisions = self.params.get("neutralization_decisions")
                if not isinstance(decisions, Mapping) or set(decisions) != (
                    REQUIRED_ALPHA_NEUTRALIZATION_DECISIONS
                ) or any(type(value) is not bool for value in decisions.values()):
                    raise ValueError("ALPHA_NEUTRALIZATION_DECISIONS_REQUIRED")
                if self.neutralize_against == "ALL_RISK":
                    raise ValueError("ALPHA_ALL_RISK_FORBIDDEN")
                if isinstance(self.neutralize_against, str):
                    raise ValueError("ALPHA_NEUTRALIZATION_EXPLICIT_LIST_REQUIRED")
                neutralized = set(self.neutralize_against or ())
                if len(neutralized) != len(self.neutralize_against or ()):
                    raise ValueError("ALPHA_NEUTRALIZATION_DUPLICATE_FACTOR")
                named_decisions = REQUIRED_ALPHA_NEUTRALIZATION_DECISIONS - {
                    "statistical_factors"
                }
                if any(
                    decisions[factor_id] != (factor_id in neutralized)
                    for factor_id in named_decisions
                ):
                    raise ValueError("ALPHA_NEUTRALIZATION_DECISION_LIST_MISMATCH")
                if decisions["statistical_factors"]:
                    raise ValueError("ALPHA_STATISTICAL_NEUTRALIZATION_FORBIDDEN_DELTA_ONLY")
                statistical_ids = {
                    factor_id for factor_id in neutralized
                    if factor_id.startswith("statistical_factor_")
                }
                if decisions["statistical_factors"] != bool(statistical_ids):
                    raise ValueError("ALPHA_STATISTICAL_NEUTRALIZATION_DECISION_MISMATCH")
                if neutralized:
                    comparison = self.params.get("neutralization_comparison")
                    if not isinstance(comparison, Mapping) or not (
                        comparison.get("registered_before_labels") is True
                        and comparison.get("metric") == "IC"
                        and comparison.get("arms") == ["raw", "neutralized"]
                    ):
                        raise ValueError("ALPHA_NEUTRALIZATION_IC_COMPARISON_PREREGISTRATION_REQUIRED")
                if neutralized and self.risk_set_version is None:
                    raise ValueError("ALPHA_NEUTRALIZATION_RISK_SET_VERSION_REQUIRED")
        elif self.role is FactorRole.PROBE:
            if self.direction is not None:
                raise ValueError("PROBE_DIRECTION_FORBIDDEN")
            if self.risk_set_version is not None:
                raise ValueError("PROBE_RISK_SET_BINDING_FORBIDDEN")
            if not self.params.get("probe_rationale"):
                raise ValueError("PROBE_RATIONALE_REQUIRED")
            if self.params.get("deployable") is not False:
                raise ValueError("PROBE_MUST_DECLARE_NOT_DEPLOYABLE")
        else:
            raise ValueError("FEATURE_ROLE_UNASSIGNED_REQUIRES_EXPLICIT_ASSIGNMENT")
        if self.status in {FactorStatus.FROZEN, FactorStatus.VALIDATED} and self.frozen_date is None:
            raise ValueError("FROZEN_FACTOR_DATE_REQUIRED")
        expected_key = canonical_feature_key(
            formula_expr=self.formula_expr,
            source_tables=self.source_tables,
            source_fields=self.source_fields,
            params=self.params,
            pit_key=self.pit_key,
        )
        if self.feature_key != expected_key:
            raise ValueError("FACTOR_FEATURE_KEY_MISMATCH")


@dataclass(frozen=True)
class RiskSetMemberRecord:
    """Role-specific membership of a feature in one risk-set version."""

    risk_set_version: int
    feature_id: str
    feature_version: int
    winsorize_method: str | None
    standardize: str | None
    orthogonalize_after: tuple[str, ...]
    admitted_reason: Mapping[str, object]
    status: str

    def __post_init__(self) -> None:
        if self.risk_set_version < 1 or self.feature_version < 1:
            raise ValueError("RISK_SET_MEMBER_VERSION_INVALID")
        if self.status not in {"candidate", "frozen", "retired"}:
            raise ValueError("RISK_SET_MEMBER_STATUS_INVALID")


@dataclass(frozen=True)
class AlphaAssertionRecord:
    """A bettable claim is defined by feature, frozen basis, and horizon."""

    feature_id: str
    feature_version: int
    risk_set_version: int
    horizon_days: int
    hypothesis: str
    direction: int
    variant_count: int
    family_root_id: str
    status: str
    holdout_touched: bool

    def __post_init__(self) -> None:
        if self.feature_version < 1 or self.risk_set_version < 1 or self.horizon_days < 1:
            raise ValueError("ALPHA_ASSERTION_KEY_INVALID")
        if self.direction not in (-1, 1):
            raise ValueError("ALPHA_ASSERTION_DIRECTION_INVALID")
        if self.variant_count < 1 or not self.family_root_id or not self.hypothesis.strip():
            raise ValueError("ALPHA_ASSERTION_PREREGISTRATION_INCOMPLETE")


@dataclass(frozen=True)
class FamilyRootDeclaration:
    """Frozen economic ownership statement, attached to a dimension root."""

    family_root_id: str
    declared_role: FeatureRole
    declaration: str
    declared_at: date
    reason: str
    status: str = "frozen"

    def __post_init__(self) -> None:
        if not self.family_root_id or not self.declaration.strip() or not self.reason.strip():
            raise ValueError("FAMILY_ROOT_DECLARATION_INCOMPLETE")
        if self.declared_role not in {FeatureRole.BASIS, FeatureRole.BETTABLE, FeatureRole.PROBE}:
            raise ValueError("FAMILY_ROOT_DECLARATION_ROLE_INVALID")


