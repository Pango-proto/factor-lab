from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from factor_matrix.factor_engine import FactorRegistry, FactorRegistryStore


PROJECT = Path(__file__).resolve().parents[1]


def test_v2_registry_splits_definition_membership_assertion_and_probe(tmp_path: Path) -> None:
    store = FactorRegistryStore(tmp_path / "registry.sqlite")
    store.sync_definitions(FactorRegistry.discover())
    with sqlite3.connect(store.path) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"feature_registry", "risk_set_member", "alpha_assertion", "probe_registry"} <= tables
        columns = {row[1] for row in connection.execute("PRAGMA table_info(feature_registry)")}
        assert {"family", "direction", "variant_count", "risk_set_version"}.isdisjoint(columns)
        assert connection.execute("SELECT count(*) FROM alpha_assertion").fetchone()[0] == 0


def test_same_feature_is_allowed_to_migrate_across_basis_versions_but_not_self_neutralize(
    tmp_path: Path,
) -> None:
    store = FactorRegistryStore(tmp_path / "registry.sqlite")
    store.sync_definitions(FactorRegistry.discover())

    with sqlite3.connect(store.path) as connection:
        # risk-set headers (DB trigger: member/assertion require a known header)
        for version in (1, 2):
            connection.execute(
                "INSERT INTO risk_set "
                "(risk_set_version, status, purpose, rationale, created_at) "
                "VALUES (?, 'candidate', 'research', 'test fixture', '2026-08-18')",
                (version,),
            )

        # size is a basis member of version 1 only
        connection.execute(
            "INSERT INTO risk_set_member "
            "(risk_set_version, feature_id, feature_version, winsorize_method, "
            " standardize, orthogonalize_after_json, admitted_reason_json, "
            " status, frozen_date) "
            "VALUES (1, 'size', 1, 'mad', 'weighted_z', '[]', '{}', 'candidate', NULL)"
        )

        # submitted attempts for both basis versions, so the only remaining
        # reason an insert can abort is self-neutralization
        for version in (1, 2):
            connection.execute(
                "INSERT INTO research_attempt "
                "(attempt_id, family_root_id, feature_id, feature_version, "
                " feature_spec_hash, risk_set_version, horizon_days, submitted, "
                " attempted_at, code_sha, config_sha) "
                "VALUES (?, 'size', 'size', 1, 'spec', ?, 1, 1, "
                "        '2026-08-17', 'code', 'config')",
                (f"attempt_basis_{version}", version),
            )

        # migration across basis versions is allowed
        connection.execute(
            "INSERT INTO alpha_assertion "
            "(feature_id, feature_version, risk_set_version, horizon_days, "
            " hypothesis, direction, variant_count, family_root_id, status, "
            " holdout_touched, neutralize_against_json, registered_at, attempt_id) "
            "VALUES ('size', 1, 2, 1, 'size predicts returns', 1, 1, 'size', "
            "        'testing', 0, '[]', '2026-08-17', 'attempt_basis_2')"
        )

        # betting on a feature inside its own basis is not
        with pytest.raises(sqlite3.IntegrityError, match="SELF_NEUTRALIZATION"):
            connection.execute(
                "INSERT INTO alpha_assertion "
                "(feature_id, feature_version, risk_set_version, horizon_days, "
                " hypothesis, direction, variant_count, family_root_id, status, "
                " holdout_touched, neutralize_against_json, registered_at, attempt_id) "
                "VALUES ('size', 1, 1, 1, 'invalid', 1, 1, 'size', "
                "        'testing', 0, '[]', '2026-08-17', 'attempt_basis_1')"
            )


def test_family_root_declarations_are_frozen_before_alpha_labels(tmp_path: Path) -> None:
    config = json.loads(
        (PROJECT / "config" / "family_root_declarations_v1.json").read_text()
    )
    assert config["freeze_before_any_alpha_labels"] is True
    assert {item["family_root_id"] for item in config["declarations"]} == {
        "valuation", "momentum", "growth", "leverage", "earnings_yield",
    }
    assert all(item["declared_role"] == "bettable" for item in config["declarations"])


def test_alpha_risk_migration_is_one_way_and_recharges_fdr() -> None:
    config = json.loads(
        (PROJECT / "config" / "alpha_risk_migration_protocol_v1.json").read_text()
    )
    assert config["alpha_to_risk"]["allowed"] is True
    assert config["risk_to_alpha"]["allowed"] is False
    assert "increase the current FDR denominator" in " ".join(
        config["re_capturing_cost"]["on_risk_set_version_change"]
    )
    assert config["statistical_factors"] == {
        "role": "basis", "bettable": False, "formula_expr": None,
        "lineage": "derived_from_run_id_required",
    }

def test_feature_key_identifies_definition(tmp_path: Path) -> None:
    """v2 protects definitions via feature_key UNIQUE, not via a status column."""
    store = FactorRegistryStore(tmp_path / "registry.sqlite")
    store.sync_definitions(FactorRegistry.discover())
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT feature_key, count(*) FROM feature_registry "
            "GROUP BY feature_key HAVING count(*) > 1"
        ).fetchall()
    assert rows == []