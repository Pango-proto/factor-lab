"""Extensible, portfolio-agnostic factor calculation platform."""

from .contracts import (
    AlphaAssertionRecord, FactorFamily, FactorRole, FactorSpec, FactorStatus,
    FeatureRole, FamilyRootDeclaration, RiskSetMemberRecord,
    canonical_feature_key, code_file_sha,
)
from .plugin import FactorDefinition
from .registry import FactorRegistry
from .store import FactorRegistryStore

__all__ = [
    "FactorDefinition",
    "FactorFamily",
    "FeatureRole",
    "FamilyRootDeclaration",
    "RiskSetMemberRecord",
    "AlphaAssertionRecord",
    "FactorRegistry",
    "FactorRegistryStore",
    "FactorRole",
    "FactorSpec",
    "FactorStatus",
    "canonical_feature_key",
    "code_file_sha",
]
