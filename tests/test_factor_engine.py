from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
import sqlite3

import pytest

from factor_matrix.factor_engine import (
    FactorFamily, FactorRegistry, FactorRegistryStore, FactorRole, FactorSpec,
    FactorStatus, canonical_feature_key,
)
from factor_matrix.risk_model import load_risk_factor_set


@dataclass(frozen=True)
class Definition:
    spec: FactorSpec


def definition(
    factor_id: str,
    family: FactorFamily,
    *,
    formula: str | None = None,
    orthogonalize_after: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
) -> Definition:
    formula_expr = formula or factor_id
    tables = ("valuation_daily",)
    fields = ("float_mkt_cap",)
    params = (
        {} if family is FactorFamily.RISK
        else {
            "evaluation_horizon_days": 20,
            "preregistered_groups": ["board"],
            "neutralization_decisions": {
                "beta": False,
                "residual_volatility": False,
                "liquidity": False,
                "listing_age": False,
                "statistical_factors": False,
            },
        }
    )
    return Definition(FactorSpec(
        factor_id=factor_id,
        version=1,
        family=family,
        role=(FactorRole.STYLE_RISK if family is FactorFamily.RISK else FactorRole.ALPHA_CANDIDATE),
        display_name=factor_id,
        feature_key=canonical_feature_key(
            formula_expr=formula_expr, source_tables=tables,
            source_fields=fields, params=params, pit_key="trade_date",
        ),
        formula_expr=formula_expr,
        source_tables=tables,
        source_fields=fields,
        params=params,
        lookback_days=None,
        min_obs=None,
        pit_key="trade_date",
        winsorize_method="mad",
        winsorize_k=5.0,
        standardize="weighted_z",
        weight_scheme="sqrt_mktcap",
        neutralize_against=None,
        orthogonalize_after=orthogonalize_after,
        direction=(None if family is FactorFamily.RISK else 1),
        missing_policy="industry_median",
        coverage_min=0.95,
        depends_on=depends_on,
        status=FactorStatus.TESTING,
        proposed_date=date(2026, 8, 14),
        frozen_date=None,
        status_changed_date=date(2026, 8, 14),
        status_reason="test",
        variant_count=1,
        family_root_id=factor_id,
        holdout_touched=False,
        research_log_refs=(),
        hypothesis="test hypothesis",
        reference=None,
        risk_set_version=None,
        code_path="tests/test_factor_engine.py",
        code_sha="fixture",
        owner=None,
    ))


def test_one_registry_contains_risk_and_alpha_views() -> None:
    registry = FactorRegistry.discover()
    assert registry.factor_ids(FactorFamily.ALPHA) == ()
    assert {"country", "size", "beta"} <= set(registry.factor_ids(FactorFamily.RISK))
    assert registry.get("ownership").spec.status is FactorStatus.BLOCKED
    assert registry.get("board").spec.source_tables == ("tradable_universe_v1",)
    assert registry.get("listing_age").spec.formula_expr == (
        "log1p((trade_date - exchange_list_date).calendar_days)"
    )


def test_registry_supports_versioned_add_remove() -> None:
    registry = FactorRegistry([definition("candidate", FactorFamily.ALPHA)])
    assert registry.get("candidate", 1).spec.family is FactorFamily.ALPHA
    registry.remove("candidate", 1)
    with pytest.raises(KeyError, match="FACTOR_UNKNOWN"):
        registry.get("candidate", 1)


def test_alpha_registration_requires_all_five_neutralization_decisions() -> None:
    valid = definition("candidate", FactorFamily.ALPHA).spec
    params = dict(valid.params)
    params.pop("neutralization_decisions")
    with pytest.raises(ValueError, match="ALPHA_NEUTRALIZATION_DECISIONS_REQUIRED"):
        replace(valid, params=params)


def test_alpha_registration_forbids_all_risk_even_with_version() -> None:
    valid = definition("candidate", FactorFamily.ALPHA).spec
    with pytest.raises(ValueError, match="ALPHA_ALL_RISK_FORBIDDEN"):
        replace(valid, neutralize_against="ALL_RISK", risk_set_version=1)


def test_alpha_neutralization_decisions_must_match_explicit_factor_list() -> None:
    valid = definition("candidate", FactorFamily.ALPHA).spec
    with pytest.raises(ValueError, match="ALPHA_NEUTRALIZATION_DECISION_LIST_MISMATCH"):
        replace(valid, neutralize_against=("beta",), risk_set_version=1)
    params = dict(valid.params)
    params["neutralization_decisions"] = {
        **params["neutralization_decisions"], "beta": True,
    }
    params["neutralization_comparison"] = {
        "registered_before_labels": True,
        "arms": ["raw", "neutralized"],
        "metric": "IC",
    }
    accepted = replace(
        valid, params=params, neutralize_against=("beta",), risk_set_version=1,
    )
    assert accepted.neutralize_against == ("beta",)


def test_alpha_neutralization_requires_raw_vs_neutralized_ic_preregistration() -> None:
    valid = definition("candidate", FactorFamily.ALPHA).spec
    params = dict(valid.params)
    params["neutralization_decisions"] = {
        **params["neutralization_decisions"], "liquidity": True,
    }
    with pytest.raises(
        ValueError, match="ALPHA_NEUTRALIZATION_IC_COMPARISON_PREREGISTRATION_REQUIRED"
    ):
        replace(
            valid, params=params, neutralize_against=("liquidity",),
            risk_set_version=1,
        )


def test_feature_key_blocks_cross_family_duplicate() -> None:
    shared_formula = "log(float_mkt_cap)"
    with pytest.raises(ValueError, match="FACTOR_FEATURE_KEY_DUPLICATE"):
        FactorRegistry([
            definition("size_risk", FactorFamily.RISK, formula=shared_formula),
            definition("size_alpha", FactorFamily.ALPHA, formula=shared_formula),
        ])


def test_dependency_and_orthogonalization_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="FACTOR_ORTHOGONALIZATION_CYCLE"):
        FactorRegistry([
            definition("a", FactorFamily.RISK, orthogonalize_after=("b",)),
            definition("b", FactorFamily.RISK, orthogonalize_after=("a",)),
        ])


def test_candidate_risk_set_uses_registry_versions_and_topology() -> None:
    registry = FactorRegistry.discover()
    factor_set = load_risk_factor_set(
        Path("config/risk_factor_set_candidate_v1.json"), registry
    )
    ordered = registry.ordered_risk_factors(
        tuple(member.factor_id for member in factor_set.members)
    )
    assert factor_set.status == "candidate"
    assert "ownership" not in {member.factor_id for member in factor_set.members}
    assert factor_set.blocked_candidates[0].factor_id == "ownership"
    assert factor_set.candidate_selection_order[:3] == (
        "size", "beta", "residual_volatility",
    )
    assert ordered.index("size") < ordered.index("beta")
    assert ordered.index("beta") < ordered.index("residual_volatility")


def test_registry_store_materializes_one_physical_table(tmp_path) -> None:
    store = FactorRegistryStore(tmp_path / "factor_registry.sqlite")
    registry = FactorRegistry.discover()
    store.sync_definitions(registry)
    assert store.row_count() == len(registry.search())


def test_registry_store_materializes_risk_set_members(tmp_path: Path) -> None:
    registry = FactorRegistry.discover()
    factor_set = load_risk_factor_set(
        Path("config/risk_factor_set_candidate_v1.json"), registry
    )
    store = FactorRegistryStore(tmp_path / "factor_registry.sqlite")
    store.sync_definitions(registry)
    store.sync_risk_factor_set(factor_set)

    with sqlite3.connect(store.path) as connection:
        count = connection.execute(
            "SELECT count(*) FROM risk_set_member "
            "WHERE risk_set_version = ? AND status != 'retired'",
            (factor_set.risk_set_version,),
        ).fetchone()[0]
        assert count == len(factor_set.members)
        


def test_alpha_neutralization_can_reference_only_a_frozen_risk_set(tmp_path: Path) -> None:
    risk_registry = FactorRegistry.discover()
    factor_set = load_risk_factor_set(
        Path("config/risk_factor_set_candidate_v1.json"), risk_registry
    )
    store = FactorRegistryStore(tmp_path / "factor_registry.sqlite")
    store.sync_definitions(risk_registry)
    store.sync_risk_factor_set(factor_set)

    base = definition("future_candidate", FactorFamily.ALPHA).spec
    params = dict(base.params)
    params["neutralization_decisions"] = {
        **params["neutralization_decisions"], "beta": True,
    }
    params["neutralization_comparison"] = {
        "registered_before_labels": True,
        "arms": ["raw", "neutralized"],
        "metric": "IC",
    }
    candidate = replace(
        base, params=params, neutralize_against=("beta",),
        risk_set_version=factor_set.risk_set_version,
    )
    alpha_registry = FactorRegistry([Definition(candidate)])

    # A candidate risk set must not be referenceable. The positive case
    # (acceptance after freeze) needs a submitted research_attempt and is
    # covered when the first real alpha is registered after G6a.
    with pytest.raises(ValueError, match="ALPHA_ASSERTION_RISK_SET_NOT_FROZEN"):
        store.sync_definitions(alpha_registry)



def test_candidate_risk_set_sync_retires_blocked_stale_member(tmp_path: Path) -> None:
    registry = FactorRegistry.discover()
    factor_set = load_risk_factor_set(
        Path("config/risk_factor_set_candidate_v1.json"), registry
    )
    store = FactorRegistryStore(tmp_path / "factor_registry.sqlite")
    store.sync_definitions(registry)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO risk_set "
            "(risk_set_version, status, purpose, rationale, created_at) "
            "VALUES (1, 'candidate', 'research', 'stale fixture', '2026-08-18')"
        )
        connection.execute(
            "INSERT INTO risk_set_member "
            "(risk_set_version, feature_id, feature_version, winsorize_method, "
            " standardize, orthogonalize_after_json, admitted_reason_json, "
            " status, frozen_date) "
            "VALUES (1, 'ownership', 1, NULL, NULL, '[]', '{}', 'candidate', NULL)"
        )
    store.sync_risk_factor_set(factor_set)
