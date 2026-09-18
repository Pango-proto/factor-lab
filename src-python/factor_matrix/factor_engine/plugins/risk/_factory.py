from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

from ...contracts import (
    FactorFamily, FactorRole, FactorSpec, FactorStatus,
    canonical_feature_key, code_file_sha,
)


@dataclass(frozen=True)
class Definition:
    spec: FactorSpec


def risk_definition(
    *, definition_path: Path, factor_id: str, display_name: str, role: FactorRole,
    formula_expr: str, source_tables: tuple[str, ...], source_fields: tuple[str, ...],
    params: Mapping[str, object], pit_key: str, hypothesis: str,
    lookback_days: int | None = None, min_obs: int | None = None,
    orthogonalize_after: tuple[str, ...] = (), depends_on: tuple[str, ...] = (),
    family_root_id: str, variant_count: int,
    status: FactorStatus = FactorStatus.TESTING, status_reason: str | None = None,
) -> Definition:
    is_style = role is FactorRole.STYLE_RISK
    resolved_params = dict(params)
    if is_style:
        resolved_params.update({
            "winsorize_target_id": (
                "research_protocol.targets.target_two_sided_tail_probability"
            ),
            "coverage_target_id": (
                "research_protocol.targets.maximum_missingness_variance_inflation"
            ),
        })
    path = definition_path.resolve()
    project_root = path.parents[5]
    return Definition(FactorSpec(
        factor_id=factor_id,
        version=1,
        family=FactorFamily.RISK,
        role=role,
        display_name=display_name,
        feature_key=canonical_feature_key(
            formula_expr=formula_expr,
            source_tables=source_tables,
            source_fields=source_fields,
            params=resolved_params,
            pit_key=pit_key,
        ),
        formula_expr=formula_expr,
        source_tables=source_tables,
        source_fields=source_fields,
        params=resolved_params,
        lookback_days=lookback_days,
        min_obs=min_obs,
        pit_key=pit_key,
        winsorize_method="mad" if is_style else "none",
        winsorize_k=None,
        standardize="weighted_z" if is_style else "none",
        weight_scheme="sqrt_mktcap" if is_style else "equal",
        neutralize_against=None,
        orthogonalize_after=orthogonalize_after,
        direction=None,
        missing_policy="industry_median" if is_style else "drop",
        coverage_min=None,
        depends_on=depends_on,
        status=status,
        proposed_date=date(2026, 8, 15),
        frozen_date=None,
        status_changed_date=date(2026, 8, 15),
        status_reason=(status_reason or "Candidate risk definition; variance diagnostics pending"),
        variant_count=variant_count,
        family_root_id=family_root_id,
        holdout_touched=False,
        research_log_refs=(),
        hypothesis=hypothesis,
        reference=None,
        risk_set_version=None,
        code_path=str(path.relative_to(project_root)),
        code_sha=code_file_sha(path),
        owner=None,
    ))
