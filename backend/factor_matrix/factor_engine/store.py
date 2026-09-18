from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .registry import FactorRegistry
from .contracts import FactorRole, FactorSpec
from ..risk_model.acceptance import ensure_acceptance_schema, registry_acceptance, acceptance_from_mapping

if TYPE_CHECKING:
    from ..risk_model import RiskFactorSetSpec


SCHEMA = """
CREATE TABLE IF NOT EXISTS legacy_factor_registry (
  factor_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  family TEXT NOT NULL CHECK (family IN ('risk','alpha')),
  role TEXT NOT NULL,
  display_name TEXT NOT NULL,
  feature_key TEXT NOT NULL UNIQUE,
  formula_expr TEXT NOT NULL,
  source_tables_json TEXT NOT NULL,
  source_fields_json TEXT NOT NULL,
  params_json TEXT NOT NULL,
  lookback_days INTEGER,
  min_obs INTEGER,
  pit_key TEXT NOT NULL,
  winsorize_method TEXT,
  winsorize_k REAL,
  standardize TEXT,
  weight_scheme TEXT,
  neutralize_against_json TEXT,
  orthogonalize_after_json TEXT NOT NULL,
  direction INTEGER CHECK (direction IS NULL OR direction IN (-1,1)),
  missing_policy TEXT,
  coverage_min REAL,
  depends_on_json TEXT NOT NULL,
  status TEXT NOT NULL,
  proposed_date TEXT NOT NULL,
  frozen_date TEXT,
  status_changed_date TEXT,
  status_reason TEXT NOT NULL,
  variant_count INTEGER NOT NULL CHECK (variant_count >= 1),
  family_root_id TEXT NOT NULL,
  holdout_touched INTEGER NOT NULL CHECK (holdout_touched IN (0,1)),
  research_log_refs_json TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  reference_text TEXT,
  risk_set_version INTEGER,
  code_path TEXT NOT NULL,
  code_sha TEXT NOT NULL,
  owner TEXT,
  PRIMARY KEY (factor_id, version)
);
CREATE TABLE IF NOT EXISTS factor_exposure (
  trade_date TEXT NOT NULL,
  stock_id TEXT NOT NULL,
  factor_id TEXT NOT NULL,
  factor_version INTEGER NOT NULL,
  calculation_run_id TEXT NOT NULL,
  exposure REAL,
  PRIMARY KEY (trade_date, stock_id, factor_id, factor_version, calculation_run_id)
);
CREATE TABLE IF NOT EXISTS factor_diagnostics (
  factor_id TEXT NOT NULL,
  factor_version INTEGER NOT NULL,
  eval_run_id TEXT NOT NULL,
  metric_id TEXT NOT NULL,
  metric_value REAL,
  PRIMARY KEY (factor_id, factor_version, eval_run_id, metric_id)
);
CREATE TABLE IF NOT EXISTS research_events (
  event_id TEXT PRIMARY KEY,
  attempt_id TEXT,
  run_at TEXT NOT NULL,
  factor_ids_json TEXT NOT NULL,
  config_sha TEXT NOT NULL,
  data_end_date TEXT NOT NULL,
  touched_holdout INTEGER NOT NULL CHECK (touched_holdout IN (0,1)),
  headline_ic REAL,
  decision TEXT NOT NULL,
  operator TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_attempt (
  attempt_id TEXT PRIMARY KEY,
  family_root_id TEXT NOT NULL,
  feature_id TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  feature_spec_hash TEXT NOT NULL,
  risk_set_version INTEGER NOT NULL,
  horizon_days INTEGER NOT NULL CHECK (horizon_days >= 1),
  submitted INTEGER NOT NULL CHECK (submitted IN (0,1)),
  attempted_at TEXT NOT NULL,
  code_sha TEXT NOT NULL,
  config_sha TEXT NOT NULL,
  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version)
);
CREATE INDEX IF NOT EXISTS idx_research_attempt_family_root
  ON research_attempt(family_root_id, attempted_at);
CREATE TABLE IF NOT EXISTS research_attempt_resolution (
  attempt_id TEXT PRIMARY KEY,
  submitted INTEGER NOT NULL CHECK (submitted IN (0,1)),
  resolved_at TEXT NOT NULL,
  decision TEXT NOT NULL,
  FOREIGN KEY (attempt_id) REFERENCES research_attempt(attempt_id)
);
CREATE TABLE IF NOT EXISTS risk_set (
  risk_set_version INTEGER PRIMARY KEY,
  status TEXT NOT NULL CHECK (status IN ('candidate','frozen','retired')),
  purpose TEXT NOT NULL CHECK (purpose IN ('production_baseline','holdout_validation','research')),
  parent_version INTEGER,
  risk_basis_id TEXT,
  estimation_end_date TEXT,
  frozen_date TEXT,
  validation_artifact_run_id TEXT,
  bias_test_run_id TEXT,
  residual_correlation_run_id TEXT,
  known_residuals_json TEXT NOT NULL DEFAULT '[]',
  weight_model_id TEXT,
  selection_test_ids_json TEXT NOT NULL DEFAULT '[]',
  selection_parameters_json TEXT NOT NULL DEFAULT '{}',
  members_manifest_sha TEXT,
  expansion_manifest_sha TEXT,
  rationale TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (parent_version) REFERENCES risk_set(risk_set_version)
);
CREATE TABLE IF NOT EXISTS risk_set_version (
  risk_set_version INTEGER NOT NULL,
  factor_id TEXT NOT NULL,
  factor_version INTEGER NOT NULL,
  status TEXT NOT NULL,
  frozen_date TEXT,
  validation_artifact_run_id TEXT,
  bias_test_run_id TEXT,
  residual_correlation_run_id TEXT,
  known_residuals_json TEXT NOT NULL DEFAULT '[]',
  rationale TEXT NOT NULL,
  PRIMARY KEY (risk_set_version, factor_id, factor_version),
  FOREIGN KEY (factor_id, factor_version) REFERENCES legacy_factor_registry(factor_id, version)
);
CREATE TRIGGER IF NOT EXISTS legacy_factor_registry_frozen_definition_immutable
BEFORE UPDATE ON legacy_factor_registry
WHEN OLD.status IN ('frozen','validated','deprecated','invalidated') AND (
  NEW.feature_key IS NOT OLD.feature_key OR
  NEW.formula_expr IS NOT OLD.formula_expr OR
  NEW.source_tables_json IS NOT OLD.source_tables_json OR
  NEW.source_fields_json IS NOT OLD.source_fields_json OR
  NEW.params_json IS NOT OLD.params_json OR
  NEW.pit_key IS NOT OLD.pit_key OR
  NEW.winsorize_method IS NOT OLD.winsorize_method OR
  NEW.winsorize_k IS NOT OLD.winsorize_k OR
  NEW.standardize IS NOT OLD.standardize OR
  NEW.weight_scheme IS NOT OLD.weight_scheme OR
  NEW.neutralize_against_json IS NOT OLD.neutralize_against_json OR
  NEW.orthogonalize_after_json IS NOT OLD.orthogonalize_after_json OR
  NEW.direction IS NOT OLD.direction OR
  NEW.missing_policy IS NOT OLD.missing_policy OR
  NEW.coverage_min IS NOT OLD.coverage_min OR
  NEW.depends_on_json IS NOT OLD.depends_on_json OR
  NEW.risk_set_version IS NOT OLD.risk_set_version OR
  NEW.code_sha IS NOT OLD.code_sha
)
BEGIN
  SELECT RAISE(ABORT, 'FROZEN_FACTOR_DEFINITION_IMMUTABLE');
END;
CREATE TRIGGER IF NOT EXISTS legacy_factor_registry_frozen_delete_forbidden
BEFORE DELETE ON legacy_factor_registry
WHEN OLD.status IN ('frozen','validated','deprecated','invalidated')
BEGIN
  SELECT RAISE(ABORT, 'FROZEN_FACTOR_DELETE_FORBIDDEN');
END;

CREATE TABLE IF NOT EXISTS feature_registry (
  feature_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version >= 1),
  feature_key TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  formula_expr TEXT,
  source_tables_json TEXT NOT NULL,
  source_fields_json TEXT NOT NULL,
  params_json TEXT NOT NULL,
  lookback_days INTEGER,
  min_obs INTEGER,
  pit_key TEXT,
  missing_policy TEXT,
  coverage_min REAL,
  family_root_id TEXT NOT NULL,
  derived_from_run_id TEXT,
  proposed_date TEXT NOT NULL,
  code_path TEXT,
  code_sha TEXT,
  owner TEXT,
  PRIMARY KEY (feature_id, version)
);
CREATE TABLE IF NOT EXISTS feature_assignment (
  feature_id TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  risk_set_version INTEGER,
  assigned_role TEXT NOT NULL CHECK (assigned_role IN ('unassigned','basis','bettable','probe')),
  status TEXT NOT NULL CHECK (status IN ('unassigned','active','blocked','retired')),
  reason TEXT NOT NULL,
  assigned_at TEXT NOT NULL,
  PRIMARY KEY (feature_id, feature_version, assigned_role, risk_set_version),
  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version),
  FOREIGN KEY (risk_set_version) REFERENCES risk_set(risk_set_version),
  CHECK (assigned_role IN ('basis','bettable') AND risk_set_version IS NOT NULL
         OR assigned_role IN ('unassigned','probe'))
);
CREATE TABLE IF NOT EXISTS risk_set_member (
  risk_set_version INTEGER NOT NULL,
  feature_id TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  winsorize_method TEXT,
  standardize TEXT,
  orthogonalize_after_json TEXT NOT NULL,
  admitted_reason_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('candidate','frozen','retired')),
  frozen_date TEXT,
  PRIMARY KEY (risk_set_version, feature_id, feature_version),
  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version)
);
CREATE TABLE IF NOT EXISTS risk_set_expansion (
  risk_set_version INTEGER NOT NULL,
  base_feature_id TEXT NOT NULL,
  base_feature_version INTEGER NOT NULL,
  expansion_group_id TEXT NOT NULL,
  expansion_type TEXT NOT NULL,
  generated_columns_json TEXT NOT NULL,
  k_definition TEXT NOT NULL,
  manifest_sha TEXT NOT NULL,
  PRIMARY KEY (risk_set_version, base_feature_id, base_feature_version, expansion_group_id),
  FOREIGN KEY (risk_set_version) REFERENCES risk_set(risk_set_version),
  FOREIGN KEY (base_feature_id, base_feature_version)
    REFERENCES feature_registry(feature_id, version)
);
CREATE TABLE IF NOT EXISTS alpha_assertion (
  feature_id TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  risk_set_version INTEGER NOT NULL,
  horizon_days INTEGER NOT NULL CHECK (horizon_days >= 1),
  hypothesis TEXT NOT NULL,
  direction INTEGER NOT NULL CHECK (direction IN (-1,1)),
  variant_count INTEGER NOT NULL CHECK (variant_count >= 1),
  family_root_id TEXT NOT NULL,
  status TEXT NOT NULL,
  holdout_touched INTEGER NOT NULL CHECK (holdout_touched IN (0,1)),
  neutralize_against_json TEXT NOT NULL DEFAULT '[]',
  attempt_id TEXT,
  registered_at TEXT NOT NULL,
  PRIMARY KEY (feature_id, feature_version, risk_set_version, horizon_days),
  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version),
  FOREIGN KEY (attempt_id) REFERENCES research_attempt(attempt_id)
);
CREATE TABLE IF NOT EXISTS probe_registry (
  feature_id TEXT NOT NULL,
  feature_version INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role = 'probe'),
  status TEXT NOT NULL,
  rationale TEXT NOT NULL,
  registered_at TEXT NOT NULL,
  PRIMARY KEY (feature_id, feature_version),
  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version)
);
CREATE TABLE IF NOT EXISTS family_root_declaration (
  family_root_id TEXT PRIMARY KEY,
  declaration_version INTEGER NOT NULL DEFAULT 1,
  declared_role TEXT NOT NULL CHECK (declared_role IN ('basis','bettable','probe')),
  declaration TEXT NOT NULL,
  declared_at TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status = 'frozen')
);
CREATE TRIGGER IF NOT EXISTS family_root_declaration_immutable
BEFORE UPDATE ON family_root_declaration
WHEN OLD.status = 'frozen' AND (
  NEW.declaration_version IS NOT OLD.declaration_version OR
  NEW.declared_role IS NOT OLD.declared_role OR
  NEW.declaration IS NOT OLD.declaration OR
  NEW.declared_at IS NOT OLD.declared_at OR
  NEW.reason IS NOT OLD.reason
)
BEGIN
  SELECT RAISE(ABORT, 'FAMILY_ROOT_DECLARATION_IMMUTABLE');
END;
CREATE TRIGGER IF NOT EXISTS family_root_declaration_delete_forbidden
BEFORE DELETE ON family_root_declaration
WHEN OLD.status = 'frozen'
BEGIN
  SELECT RAISE(ABORT, 'FAMILY_ROOT_DECLARATION_DELETE_FORBIDDEN');
END;
"""


class FactorRegistryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(SCHEMA)
            self._archive_legacy_factor_table(connection)
            self._migrate_legacy_feature_definitions(connection)
            self._migrate_legacy_research_events(connection)
            self._ensure_risk_set_columns(connection)
            self._ensure_family_declaration_columns(connection)
            self._ensure_structural_schema(connection)
            ensure_acceptance_schema(connection)

    @staticmethod
    def _archive_legacy_factor_table(connection: sqlite3.Connection) -> None:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "factor_registry" in tables and "legacy_factor_registry_archive" not in tables:
            connection.execute(
                "ALTER TABLE factor_registry RENAME TO legacy_factor_registry_archive"
            )

    @staticmethod
    def _migrate_legacy_feature_definitions(connection: sqlite3.Connection) -> None:
        """Copy the old family-coupled table into the role-neutral definition table."""
        old = {row[1] for row in connection.execute("PRAGMA table_info(legacy_factor_registry_archive)")}
        if old:
            source = "legacy_factor_registry_archive"
        else:
            old = {row[1] for row in connection.execute("PRAGMA table_info(legacy_factor_registry)")}
            source = "legacy_factor_registry"
        if not old:
            return
        connection.execute(
            f"""
            INSERT OR IGNORE INTO feature_registry (
              feature_id, version, feature_key, display_name, formula_expr,
              source_tables_json, source_fields_json, params_json, lookback_days,
              min_obs, pit_key, missing_policy, coverage_min, family_root_id,
              derived_from_run_id, proposed_date, code_path, code_sha, owner
            )
            SELECT factor_id, version, feature_key, display_name, formula_expr,
              source_tables_json, source_fields_json, params_json, lookback_days,
              min_obs, pit_key, missing_policy, coverage_min, family_root_id,
              json_extract(params_json, '$.derived_from_run_id'), proposed_date,
              code_path, code_sha, owner
            FROM {source}
            """
        )

    @staticmethod
    def _ensure_risk_set_columns(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(risk_set_version)")}
        additions = {
            "bias_test_run_id": "TEXT",
            "residual_correlation_run_id": "TEXT",
            "known_residuals_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE risk_set_version ADD COLUMN {name} {definition}"
                )

    @staticmethod
    def _ensure_family_declaration_columns(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(family_root_declaration)")
        }
        if columns and "declaration_version" not in columns:
            connection.execute(
                "ALTER TABLE family_root_declaration ADD COLUMN declaration_version INTEGER NOT NULL DEFAULT 1"
            )
        if columns:
            # Recreate the trigger so pre-existing metadata databases also
            # treat the declaration version as immutable.
            connection.execute("DROP TRIGGER IF EXISTS family_root_declaration_immutable")
            connection.execute(
                """
                CREATE TRIGGER family_root_declaration_immutable
                BEFORE UPDATE ON family_root_declaration
                WHEN OLD.status = 'frozen' AND (
                  NEW.declaration_version IS NOT OLD.declaration_version OR
                  NEW.declared_role IS NOT OLD.declared_role OR
                  NEW.declaration IS NOT OLD.declaration OR
                  NEW.declared_at IS NOT OLD.declared_at OR
                  NEW.reason IS NOT OLD.reason
                )
                BEGIN
                  SELECT RAISE(ABORT, 'FAMILY_ROOT_DECLARATION_IMMUTABLE');
                END
                """
            )

    @staticmethod
    def _ensure_structural_schema(connection: sqlite3.Connection) -> None:
        """Upgrade existing metadata DBs without treating legacy tables as state."""
        columns = {row[1] for row in connection.execute("PRAGMA table_info(alpha_assertion)")}
        if "attempt_id" not in columns:
            connection.execute("ALTER TABLE alpha_assertion ADD COLUMN attempt_id TEXT")
        risk_columns = {row[1] for row in connection.execute("PRAGMA table_info(risk_set)")}
        if "expansion_manifest_sha" not in risk_columns:
            connection.execute("ALTER TABLE risk_set ADD COLUMN expansion_manifest_sha TEXT")

        # Existing databases were created before the structural tables existed.
        # Preserve their rows, but make the new risk_set header authoritative.
        connection.execute(
            """
            INSERT OR IGNORE INTO risk_set(
              risk_set_version, status, purpose, parent_version, risk_basis_id,
              frozen_date, validation_artifact_run_id, bias_test_run_id,
              residual_correlation_run_id, known_residuals_json, rationale, created_at
            )
            SELECT risk_set_version,
              CASE WHEN max(status) = 'frozen' THEN 'frozen' ELSE 'candidate' END,
              'production_baseline', NULL, NULL, max(frozen_date),
              max(validation_artifact_run_id), max(bias_test_run_id),
              max(residual_correlation_run_id), max(known_residuals_json),
              max(rationale), COALESCE(max(frozen_date), '1970-01-01')
            FROM risk_set_version
            GROUP BY risk_set_version
            """
        )
        assignment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(feature_assignment)")
        }
        legacy_assignment_table = None
        if "risk_set_version" not in assignment_columns:
            legacy_assignment_table = "feature_assignment_legacy"
            connection.execute(
                "ALTER TABLE feature_assignment RENAME TO feature_assignment_legacy"
            )
            connection.execute(
                """
                CREATE TABLE feature_assignment (
                  feature_id TEXT NOT NULL,
                  feature_version INTEGER NOT NULL,
                  risk_set_version INTEGER,
                  assigned_role TEXT NOT NULL CHECK (assigned_role IN ('unassigned','basis','bettable','probe')),
                  status TEXT NOT NULL CHECK (status IN ('unassigned','active','blocked','retired')),
                  reason TEXT NOT NULL,
                  assigned_at TEXT NOT NULL,
                  PRIMARY KEY (feature_id, feature_version, assigned_role, risk_set_version),
                  FOREIGN KEY (feature_id, feature_version) REFERENCES feature_registry(feature_id, version),
                  FOREIGN KEY (risk_set_version) REFERENCES risk_set(risk_set_version),
                  CHECK (assigned_role IN ('basis','bettable') AND risk_set_version IS NOT NULL
                         OR assigned_role IN ('unassigned','probe'))
                )
                """
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO feature_assignment(
              feature_id, feature_version, risk_set_version,
              assigned_role, status, reason, assigned_at
            )
            SELECT feature_id, version, NULL, 'unassigned',
              'unassigned', 'No explicit role assignment has been recorded.', proposed_date
            FROM feature_registry
            """
        )
        legacy_table = None
        for candidate in ("legacy_factor_registry_archive", "legacy_factor_registry"):
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (candidate,)
            ).fetchone():
                legacy_table = candidate
                break
        if legacy_table:
            connection.execute(
                f"""INSERT OR IGNORE INTO feature_assignment(
                       feature_id,feature_version,risk_set_version,assigned_role,
                       status,reason,assigned_at
                     )
                     SELECT legacy.factor_id,legacy.version,NULL,'unassigned','blocked',
                       legacy.status_reason,legacy.proposed_date
                     FROM {legacy_table} legacy
                     WHERE legacy.status='blocked'"""
            )
        connection.execute(
            """INSERT OR IGNORE INTO feature_assignment(
                 feature_id,feature_version,risk_set_version,assigned_role,status,reason,assigned_at
               )
               SELECT m.feature_id,m.feature_version,m.risk_set_version,'basis','active',
                 'Explicitly materialized in risk_set_member.',COALESCE(m.frozen_date,'1970-01-01')
               FROM risk_set_member m
               WHERE m.status != 'retired'"""
        )
        connection.execute(
            """INSERT OR IGNORE INTO feature_assignment(
                 feature_id,feature_version,risk_set_version,assigned_role,status,reason,assigned_at
               )
               SELECT a.feature_id,a.feature_version,a.risk_set_version,'bettable','active',
                 'Explicitly materialized in alpha_assertion.',a.registered_at
               FROM alpha_assertion a"""
        )
        connection.execute(
            """INSERT OR IGNORE INTO feature_assignment(
                 feature_id,feature_version,risk_set_version,assigned_role,status,reason,assigned_at
               )
               SELECT p.feature_id,p.feature_version,NULL,'probe','active',
                 'Explicitly materialized in probe_registry.',p.registered_at
               FROM probe_registry p"""
        )
        connection.execute(
            """DELETE FROM feature_assignment
               WHERE assigned_role='unassigned'
                 AND EXISTS (
                   SELECT 1 FROM feature_assignment active
                   WHERE active.feature_id=feature_assignment.feature_id
                     AND active.feature_version=feature_assignment.feature_version
                     AND active.assigned_role != 'unassigned'
               )"""
        )
        connection.execute(
            """DELETE FROM feature_assignment
               WHERE rowid NOT IN (
                 SELECT min(rowid) FROM feature_assignment
                 GROUP BY feature_id,feature_version,assigned_role,risk_set_version
               )"""
        )
        connection.execute(
            """DELETE FROM feature_assignment
               WHERE assigned_role='unassigned' AND status='unassigned'
                 AND EXISTS (
                   SELECT 1 FROM feature_assignment blocked
                   WHERE blocked.feature_id=feature_assignment.feature_id
                     AND blocked.feature_version=feature_assignment.feature_version
                     AND blocked.status='blocked'
                 )"""
        )
        connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_feature_assignment_global_role
               ON feature_assignment(feature_id, feature_version, assigned_role)
               WHERE risk_set_version IS NULL"""
        )

        connection.execute("DROP TRIGGER IF EXISTS alpha_assertion_cannot_self_neutralize")
        connection.execute("DROP TRIGGER IF EXISTS risk_set_member_cannot_absorb_alpha_assertion")
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS alpha_assertion_requires_submitted_attempt
            BEFORE INSERT ON alpha_assertion
            WHEN NEW.attempt_id IS NULL OR NOT EXISTS (
              SELECT 1 FROM research_attempt a
              WHERE a.attempt_id = NEW.attempt_id
                AND a.feature_id = NEW.feature_id
                AND a.feature_version = NEW.feature_version
                AND a.risk_set_version = NEW.risk_set_version
                AND a.horizon_days = NEW.horizon_days
                AND (a.submitted = 1 OR EXISTS (
                  SELECT 1 FROM research_attempt_resolution r
                  WHERE r.attempt_id = a.attempt_id AND r.submitted = 1
                ))
            )
            BEGIN
              SELECT RAISE(ABORT, 'ALPHA_ASSERTION_SUBMITTED_ATTEMPT_REQUIRED');
            END;

            CREATE TRIGGER IF NOT EXISTS alpha_assertion_cannot_self_neutralize
            BEFORE INSERT ON alpha_assertion
            WHEN EXISTS (
              SELECT 1 FROM risk_set_member m
              WHERE m.risk_set_version = NEW.risk_set_version
                AND m.feature_id = NEW.feature_id
            )
            BEGIN
              SELECT RAISE(ABORT, 'ALPHA_ASSERTION_SELF_NEUTRALIZATION_FORBIDDEN');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_member_cannot_absorb_alpha_assertion
            BEFORE INSERT ON risk_set_member
            WHEN EXISTS (
              SELECT 1 FROM alpha_assertion a
              WHERE a.risk_set_version = NEW.risk_set_version
                AND a.feature_id = NEW.feature_id
            )
            BEGIN
              SELECT RAISE(ABORT, 'RISK_MEMBER_ALPHA_ASSERTION_COLLISION');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_member_requires_known_risk_set
            BEFORE INSERT ON risk_set_member
            WHEN NOT EXISTS (
              SELECT 1 FROM risk_set r
              WHERE r.risk_set_version = NEW.risk_set_version
            )
            BEGIN
              SELECT RAISE(ABORT, 'RISK_SET_HEADER_REQUIRED');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_member_frozen_immutable
            BEFORE UPDATE ON risk_set_member
            WHEN EXISTS (
              SELECT 1 FROM risk_set r
              WHERE r.risk_set_version = OLD.risk_set_version
                AND r.status = 'frozen'
            )
            BEGIN
              SELECT RAISE(ABORT, 'FROZEN_RISK_SET_MEMBER_IMMUTABLE');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_member_frozen_delete_forbidden
            BEFORE DELETE ON risk_set_member
            WHEN EXISTS (
              SELECT 1 FROM risk_set r
              WHERE r.risk_set_version = OLD.risk_set_version
                AND r.status = 'frozen'
            )
            BEGIN
              SELECT RAISE(ABORT, 'FROZEN_RISK_SET_MEMBER_DELETE_FORBIDDEN');
            END;

            CREATE TRIGGER IF NOT EXISTS alpha_assertion_immutable
            BEFORE UPDATE ON alpha_assertion
            BEGIN
              SELECT RAISE(ABORT, 'ALPHA_ASSERTION_APPEND_ONLY');
            END;

            CREATE TRIGGER IF NOT EXISTS alpha_assertion_delete_forbidden
            BEFORE DELETE ON alpha_assertion
            BEGIN
              SELECT RAISE(ABORT, 'ALPHA_ASSERTION_DELETE_FORBIDDEN');
            END;

            CREATE TRIGGER IF NOT EXISTS research_attempt_immutable
            BEFORE UPDATE ON research_attempt
            BEGIN
              SELECT RAISE(ABORT, 'RESEARCH_ATTEMPT_APPEND_ONLY');
            END;

            CREATE TRIGGER IF NOT EXISTS research_attempt_delete_forbidden
            BEFORE DELETE ON research_attempt
            BEGIN
              SELECT RAISE(ABORT, 'RESEARCH_ATTEMPT_DELETE_FORBIDDEN');
            END;

            CREATE TRIGGER IF NOT EXISTS research_attempt_resolution_immutable
            BEFORE UPDATE ON research_attempt_resolution
            BEGIN
              SELECT RAISE(ABORT, 'RESEARCH_ATTEMPT_RESOLUTION_APPEND_ONLY');
            END;

            CREATE TRIGGER IF NOT EXISTS research_attempt_resolution_delete_forbidden
            BEFORE DELETE ON research_attempt_resolution
            BEGIN
              SELECT RAISE(ABORT, 'RESEARCH_ATTEMPT_RESOLUTION_DELETE_FORBIDDEN');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_frozen_immutable
            BEFORE UPDATE ON risk_set
            WHEN OLD.status = 'frozen' AND (
              NEW.status IS NOT OLD.status OR NEW.purpose IS NOT OLD.purpose OR
              NEW.parent_version IS NOT OLD.parent_version OR
              NEW.risk_basis_id IS NOT OLD.risk_basis_id OR
              NEW.estimation_end_date IS NOT OLD.estimation_end_date OR
              NEW.frozen_date IS NOT OLD.frozen_date OR
              NEW.validation_artifact_run_id IS NOT OLD.validation_artifact_run_id OR
              NEW.bias_test_run_id IS NOT OLD.bias_test_run_id OR
              NEW.residual_correlation_run_id IS NOT OLD.residual_correlation_run_id OR
              NEW.known_residuals_json IS NOT OLD.known_residuals_json OR
              NEW.weight_model_id IS NOT OLD.weight_model_id OR
              NEW.selection_test_ids_json IS NOT OLD.selection_test_ids_json OR
              NEW.selection_parameters_json IS NOT OLD.selection_parameters_json OR
              NEW.members_manifest_sha IS NOT OLD.members_manifest_sha OR
              NEW.expansion_manifest_sha IS NOT OLD.expansion_manifest_sha OR
              NEW.rationale IS NOT OLD.rationale
            )
            BEGIN
              SELECT RAISE(ABORT, 'FROZEN_RISK_SET_IMMUTABLE');
            END;

            CREATE TRIGGER IF NOT EXISTS risk_set_delete_forbidden
            BEFORE DELETE ON risk_set
            WHEN OLD.status = 'frozen'
            BEGIN
              SELECT RAISE(ABORT, 'FROZEN_RISK_SET_DELETE_FORBIDDEN');
            END;

            """
        )

    def issue_research_attempt(
        self,
        *,
        family_root_id: str,
        feature_id: str,
        feature_version: int,
        feature_spec_hash: str,
        risk_set_version: int,
        horizon_days: int,
        code_sha: str,
        config_sha: str,
        submitted: bool = False,
    ) -> str:
        """Issue the only identifier that permits an L2a label read."""
        if not family_root_id or not feature_id or len(feature_spec_hash) != 64:
            raise ValueError("RESEARCH_ATTEMPT_IDENTITY_INVALID")
        if len(code_sha) != 64 or len(config_sha) != 64:
            raise ValueError("RESEARCH_ATTEMPT_SHA_INVALID")
        if feature_version < 1 or risk_set_version < 1 or horizon_days < 1:
            raise ValueError("RESEARCH_ATTEMPT_DIMENSION_INVALID")
        self.initialize()
        attempt_id = f"attempt_{uuid.uuid4().hex}"
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            if not connection.execute(
                """SELECT EXISTS(
                     SELECT 1 FROM feature_registry
                     WHERE feature_id=? AND version=?
                   )""",
                (feature_id, feature_version),
            ).fetchone()[0]:
                raise ValueError("RESEARCH_ATTEMPT_FEATURE_NOT_REGISTERED")
            if not registry_acceptance(connection, risk_set_version).basis_consumption_allowed:
                raise ValueError("RESEARCH_ATTEMPT_RISK_SET_NOT_FROZEN")
            connection.execute(
                """INSERT INTO research_attempt(
                     attempt_id, family_root_id, feature_id, feature_version,
                     feature_spec_hash, risk_set_version, horizon_days, submitted,
                     attempted_at, code_sha, config_sha
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    attempt_id, family_root_id, feature_id, feature_version,
                    feature_spec_hash, risk_set_version, horizon_days, int(submitted),
                    datetime.now(timezone.utc).isoformat(), code_sha, config_sha,
                ),
            )
        return attempt_id

    def validate_research_attempt(
        self, attempt_id: str, *, feature_id: str, feature_version: int,
        risk_set_version: int, horizon_days: int,
    ) -> None:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """SELECT feature_id, feature_version, risk_set_version, horizon_days
                   FROM research_attempt WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()
        if row != (feature_id, feature_version, risk_set_version, horizon_days):
            raise ValueError("L2A_ATTEMPT_SCOPE_MISMATCH")

    def resolve_research_attempt(
        self, attempt_id: str, *, submitted: bool, decision: str,
    ) -> None:
        if not decision.strip():
            raise ValueError("RESEARCH_ATTEMPT_DECISION_REQUIRED")
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            try:
                connection.execute(
                    """INSERT INTO research_attempt_resolution(
                         attempt_id, submitted, resolved_at, decision
                       ) VALUES (?,?,?,?)""",
                    (
                        attempt_id, int(submitted),
                        datetime.now(timezone.utc).isoformat(), decision,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("RESEARCH_ATTEMPT_RESOLUTION_APPEND_ONLY_CONFLICT") from exc

    def research_attempt_count(self, family_root_id: str | None = None) -> int:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            if family_root_id is None:
                return int(connection.execute("SELECT count(*) FROM research_attempt").fetchone()[0])
            return int(connection.execute(
                "SELECT count(*) FROM research_attempt WHERE family_root_id=?",
                (family_root_id,),
            ).fetchone()[0])

    @staticmethod
    def _migrate_legacy_research_events(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(research_events)")
        }
        required = {
            "event_id", "attempt_id", "run_at", "factor_ids_json", "config_sha", "data_end_date",
            "touched_holdout", "headline_ic", "decision", "operator", "payload_json",
        }
        if columns == required:
            return
        legacy_rows = connection.execute("SELECT * FROM research_events").fetchall()
        legacy_names = [
            row[1] for row in connection.execute("PRAGMA table_info(research_events)")
        ]
        connection.execute("ALTER TABLE research_events RENAME TO research_events_legacy")
        connection.execute("""
            CREATE TABLE research_events (
              event_id TEXT PRIMARY KEY, attempt_id TEXT, run_at TEXT NOT NULL,
              factor_ids_json TEXT NOT NULL, config_sha TEXT NOT NULL,
              data_end_date TEXT NOT NULL, touched_holdout INTEGER NOT NULL
                CHECK (touched_holdout IN (0,1)),
              headline_ic REAL, decision TEXT NOT NULL, operator TEXT NOT NULL,
              payload_json TEXT NOT NULL
            )
        """)
        for values in legacy_rows:
            row = dict(zip(legacy_names, values))
            payload = row.get("payload_json") or "{}"
            run_at = row.get("occurred_at") or "1970-01-01T00:00:00"
            factor_ids = [row["factor_id"]] if row.get("factor_id") else []
            connection.execute(
                "INSERT INTO research_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["event_id"], None, run_at, json.dumps(factor_ids),
                    hashlib.sha256(payload.encode()).hexdigest(), run_at[:10], 1, None,
                    "Migrated legacy event; holdout state unknown", "legacy", payload,
                ),
            )
        connection.execute("DROP TABLE research_events_legacy")

    @staticmethod
    def _validate_alpha_risk_set_reference(
        connection: sqlite3.Connection, spec: FactorSpec,
    ) -> None:
        if spec.role is not FactorRole.ALPHA_CANDIDATE:
            return
        if spec.risk_set_version is None:
            raise ValueError("ALPHA_ASSERTION_RISK_SET_VERSION_REQUIRED")
        header = connection.execute(
            "SELECT status FROM risk_set WHERE risk_set_version=?",
            (spec.risk_set_version,),
        ).fetchone()
        if header is None:
            raise ValueError("ALPHA_ASSERTION_RISK_SET_UNKNOWN")
        if not registry_acceptance(connection, spec.risk_set_version).basis_consumption_allowed:
            raise ValueError("ALPHA_ASSERTION_RISK_SET_NOT_FROZEN")
        neutralized = set(spec.neutralize_against or ())
        rows = connection.execute(
            "SELECT feature_id,status FROM risk_set_member WHERE risk_set_version=?",
            (spec.risk_set_version,),
        ).fetchall()
        if not rows:
            raise ValueError("ALPHA_ASSERTION_RISK_SET_UNKNOWN")
        if any(status not in ("frozen", "candidate") for _, status in rows):
            raise ValueError("ALPHA_ASSERTION_RISK_SET_NOT_FROZEN")
        risk_factor_ids = {feature_id for feature_id, _ in rows}
        if not neutralized <= risk_factor_ids:
            raise ValueError("ALPHA_NEUTRALIZATION_FACTOR_NOT_IN_RISK_SET")
        expected_statistical = {
            factor_id for factor_id in risk_factor_ids
            if factor_id.startswith("statistical_factor_")
        }
        selected_statistical = neutralized & expected_statistical
        decision = spec.params["neutralization_decisions"]["statistical_factors"]
        if decision or selected_statistical:
            raise ValueError("ALPHA_STATISTICAL_NEUTRALIZATION_FORBIDDEN_DELTA_ONLY")

    def sync_definitions(self, registry: FactorRegistry) -> None:
        self.initialize()
        declaration_path = Path(__file__).resolve().parents[3] / "config/family_root_declarations_v1.json"
        if declaration_path.exists():
            self.sync_family_root_declarations(declaration_path)
        statement = """
        INSERT INTO feature_registry (
          feature_id, version, feature_key, display_name, formula_expr,
          source_tables_json, source_fields_json, params_json, lookback_days,
          min_obs, pit_key, missing_policy, coverage_min, family_root_id,
          derived_from_run_id, proposed_date, code_path, code_sha, owner
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(feature_id, version) DO UPDATE SET
          feature_key=excluded.feature_key, display_name=excluded.display_name,
          formula_expr=excluded.formula_expr, source_tables_json=excluded.source_tables_json,
          source_fields_json=excluded.source_fields_json, params_json=excluded.params_json,
          lookback_days=excluded.lookback_days, min_obs=excluded.min_obs,
          pit_key=excluded.pit_key, missing_policy=excluded.missing_policy,
          coverage_min=excluded.coverage_min, family_root_id=excluded.family_root_id,
          derived_from_run_id=excluded.derived_from_run_id, proposed_date=excluded.proposed_date,
          code_path=excluded.code_path, code_sha=excluded.code_sha, owner=excluded.owner
        """
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            for plugin in registry.search():
                spec = plugin.spec
                self._validate_alpha_risk_set_reference(connection, spec)
                connection.execute(statement, (
                    spec.factor_id, spec.version, spec.feature_key, spec.display_name,
                    spec.formula_expr,
                    json.dumps(spec.source_tables), json.dumps(spec.source_fields),
                    json.dumps(dict(spec.params), sort_keys=True), spec.lookback_days,
                    spec.min_obs, spec.pit_key, spec.missing_policy, spec.coverage_min,
                    spec.family_root_id, spec.params.get("derived_from_run_id"),
                    spec.proposed_date.isoformat(), spec.code_path, spec.code_sha, spec.owner,
                ))
                connection.execute(
                    """INSERT OR IGNORE INTO feature_assignment(
                         feature_id,feature_version,risk_set_version,assigned_role,
                         status,reason,assigned_at
                       ) VALUES (?, ?, NULL, 'unassigned', 'unassigned', ?, ?)""",
                    (
                        spec.factor_id, spec.version,
                        "No explicit role assignment has been recorded.",
                        spec.proposed_date.isoformat(),
                    ),
                )
                if spec.role is FactorRole.ALPHA_CANDIDATE:
                    declaration = connection.execute(
                        "SELECT declared_role,status FROM family_root_declaration WHERE family_root_id=?",
                        (spec.family_root_id,),
                    ).fetchone()
                    if declaration != ("bettable", "frozen"):
                        raise ValueError("ALPHA_FAMILY_ROOT_BETTABLE_DECLARATION_REQUIRED")
                    horizon = spec.params["evaluation_horizon_days"]
                    attempt_id = spec.params.get("attempt_id")
                    if not isinstance(attempt_id, str) or not attempt_id:
                        raise ValueError("ALPHA_ASSERTION_ATTEMPT_ID_REQUIRED")
                    attempt_count = connection.execute(
                        "SELECT count(*) FROM research_attempt WHERE family_root_id=?",
                        (spec.family_root_id,),
                    ).fetchone()[0]
                    if not attempt_count:
                        raise ValueError("ALPHA_ASSERTION_RESEARCH_ATTEMPT_REQUIRED")
                    connection.execute(
                        """
                        INSERT INTO alpha_assertion (
                          feature_id, feature_version, risk_set_version, horizon_days,
                          hypothesis, direction, variant_count, family_root_id,
                          status, holdout_touched, neutralize_against_json, attempt_id,
                          registered_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            spec.factor_id, spec.version, spec.risk_set_version, horizon,
                            spec.hypothesis, spec.direction, attempt_count,
                            spec.family_root_id, spec.status.value, int(spec.holdout_touched),
                            json.dumps(spec.neutralize_against or ()), attempt_id,
                            spec.proposed_date.isoformat(),
                        ),
                    )
                    connection.execute(
                        """DELETE FROM feature_assignment
                           WHERE feature_id=? AND feature_version=?
                             AND assigned_role='unassigned'""",
                        (spec.factor_id, spec.version),
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO feature_assignment(
                             feature_id,feature_version,risk_set_version,assigned_role,
                             status,reason,assigned_at
                           ) VALUES (?,?,?,'bettable','active',?,?)""",
                        (spec.factor_id, spec.version, spec.risk_set_version,
                         "Explicit Alpha assertion.", spec.proposed_date.isoformat()),
                    )
                elif spec.role is FactorRole.DESCRIPTOR:
                    connection.execute(
                        """
                        INSERT INTO probe_registry(feature_id,feature_version,role,status,rationale,registered_at)
                        VALUES (?,?, 'probe', ?, ?, ?)
                        ON CONFLICT(feature_id,feature_version) DO UPDATE SET status=excluded.status,
                          rationale=excluded.rationale
                        """,
                        (spec.factor_id, spec.version, spec.status.value, spec.hypothesis,
                         spec.proposed_date.isoformat()),
                    )
                    connection.execute(
                        """DELETE FROM feature_assignment
                           WHERE feature_id=? AND feature_version=?
                             AND assigned_role='unassigned'""",
                        (spec.factor_id, spec.version),
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO feature_assignment(
                             feature_id,feature_version,risk_set_version,assigned_role,
                             status,reason,assigned_at
                           ) VALUES (?, ?, NULL, 'probe', 'active', ?, ?)""",
                        (spec.factor_id, spec.version,
                         "Explicit probe registration.", spec.proposed_date.isoformat()),
                    )

    def row_count(self) -> int:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            return int(connection.execute("SELECT count(*) FROM feature_registry").fetchone()[0])

    def sync_family_root_declarations(self, config_path: Path) -> None:
        """Persist frozen economic ownership declarations separately from features."""
        self.initialize()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        with sqlite3.connect(self.path) as connection:
            for item in payload.get("declarations", ()):
                connection.execute(
                    """
                    INSERT INTO family_root_declaration(
                      family_root_id, declaration_version, declared_role, declaration,
                      declared_at, reason, status
                    ) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(family_root_id) DO UPDATE SET
                      declaration_version=excluded.declaration_version,
                      declared_role=excluded.declared_role,
                      declaration=excluded.declaration,
                      declared_at=excluded.declared_at,
                      reason=excluded.reason,
                      status=excluded.status
                    """,
                    (
                        item["family_root_id"], int(item.get("declaration_version", 1)),
                        item["declared_role"], item["declaration"],
                        item["declared_at"], item["reason"], item["status"],
                    ),
                )

    def sync_risk_set_expansion(self, config_path: Path) -> None:
        """Backfill the logical-to-physical K mapping from an immutable manifest."""
        self.initialize()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        expected_sha = payload.get("source_manifest_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("RISK_SET_EXPANSION_SOURCE_HASH_REQUIRED")
        source_manifest = payload.get("source_manifest")
        if not isinstance(source_manifest, str):
            raise ValueError("RISK_SET_EXPANSION_SOURCE_MANIFEST_REQUIRED")
        source_path = (config_path.parents[1] / source_manifest).resolve()
        if not source_path.exists() or hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("RISK_SET_EXPANSION_SOURCE_HASH_MISMATCH")
        generated = [
            column
            for item in payload.get("logical_members", ())
            for column in item.get("generated_columns", ())
        ]
        if len(generated) != payload.get("expanded_factor_count") or len(generated) != len(set(generated)):
            raise ValueError("RISK_SET_EXPANSION_K_INVALID")
        manifest_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        with sqlite3.connect(self.path) as connection:
            for item in payload["logical_members"]:
                connection.execute(
                    """
                    INSERT INTO risk_set_expansion(
                      risk_set_version, base_feature_id, base_feature_version,
                      expansion_group_id, expansion_type, generated_columns_json,
                      k_definition, manifest_sha
                    ) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(risk_set_version, base_feature_id, base_feature_version, expansion_group_id)
                    DO UPDATE SET expansion_type=excluded.expansion_type,
                      generated_columns_json=excluded.generated_columns_json,
                      k_definition=excluded.k_definition,
                      manifest_sha=excluded.manifest_sha
                    """,
                    (
                        payload["risk_set_version"], item["base_feature_id"],
                        item["base_feature_version"], item["expansion_group_id"],
                        item["expansion_type"], json.dumps(item["generated_columns"], sort_keys=True),
                        payload["k_definition"], manifest_sha,
                    ),
                )

    def sync_risk_factor_set(
        self, factor_set: "RiskFactorSetSpec", registry: FactorRegistry | None = None,
    ) -> None:
        """Materialize a versioned risk set without deleting membership history."""
        acceptance = acceptance_from_mapping(vars(factor_set))
        self.initialize()
        if registry is None:
            registry = FactorRegistry.discover()
        statement = """
        INSERT INTO risk_set_member (
          risk_set_version, feature_id, feature_version, winsorize_method,
          standardize, orthogonalize_after_json, admitted_reason_json, status, frozen_date
        ) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(risk_set_version, feature_id, feature_version) DO UPDATE SET
          winsorize_method=excluded.winsorize_method,
          standardize=excluded.standardize,
          orthogonalize_after_json=excluded.orthogonalize_after_json,
          admitted_reason_json=excluded.admitted_reason_json,
          status=excluded.status, frozen_date=excluded.frozen_date
        """
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            existing = connection.execute(
                "SELECT status,purpose,parent_version FROM risk_set WHERE risk_set_version=?",
                (factor_set.risk_set_version,),
            ).fetchone()
            purpose = getattr(factor_set, "purpose", "production_baseline")
            parent_version = getattr(factor_set, "parent_version", None)
            if purpose not in {"production_baseline", "holdout_validation", "research"}:
                raise ValueError("RISK_SET_PURPOSE_INVALID")
            if existing and (existing[0] == "frozen" or registry_acceptance(connection, factor_set.risk_set_version).basis_consumption_allowed):
                current = {
                    (row[0], row[1]) for row in connection.execute(
                        """SELECT feature_id,feature_version FROM risk_set_member
                           WHERE risk_set_version=? AND status != 'retired'""",
                        (factor_set.risk_set_version,),
                    )
                }
                requested = {
                    (member.factor_id, member.factor_version) for member in factor_set.members
                }
                if current != requested:
                    raise ValueError("FROZEN_RISK_SET_MEMBERSHIP_IMMUTABLE")
                old_acceptance = registry_acceptance(connection, factor_set.risk_set_version)
                if acceptance != old_acceptance:
                    raise ValueError('RISK_ACCEPTANCE_UPDATE_REQUIRES_EXPLICIT_EVIDENCE_API')
                return
            connection.execute(
                """INSERT INTO risk_set(
                     risk_set_version,status,purpose,parent_version,frozen_date,
                     validation_artifact_run_id,bias_test_run_id,
                     residual_correlation_run_id,known_residuals_json,weight_model_id,
                     selection_test_ids_json,selection_parameters_json,expansion_manifest_sha,
                     rationale,created_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(risk_set_version) DO UPDATE SET
                     status=excluded.status, purpose=excluded.purpose,
                     parent_version=excluded.parent_version, frozen_date=excluded.frozen_date,
                     validation_artifact_run_id=excluded.validation_artifact_run_id,
                     bias_test_run_id=excluded.bias_test_run_id,
                     residual_correlation_run_id=excluded.residual_correlation_run_id,
                     known_residuals_json=excluded.known_residuals_json,
                     weight_model_id=excluded.weight_model_id,
                     selection_test_ids_json=excluded.selection_test_ids_json,
                     selection_parameters_json=excluded.selection_parameters_json,
                     expansion_manifest_sha=excluded.expansion_manifest_sha,
                     rationale=excluded.rationale""",
                (
                    factor_set.risk_set_version, factor_set.status, purpose, parent_version,
                    factor_set.frozen_at.isoformat() if factor_set.frozen_at else None,
                    factor_set.validation_artifact_run_id, factor_set.bias_test_run_id,
                    factor_set.residual_correlation_run_id,
                    json.dumps(factor_set.known_residuals, ensure_ascii=False, sort_keys=True),
                    factor_set.weight_model_id,
                    json.dumps(factor_set.selection_test_ids, ensure_ascii=False),
                    json.dumps(dict(factor_set.selection_parameters), ensure_ascii=False, sort_keys=True),
                    factor_set.expansion_manifest_sha,
                    factor_set.rationale, datetime.now(timezone.utc).isoformat(),
                ),
            )
            active_keys = {
                (member.factor_id, member.factor_version) for member in factor_set.members
            }
            stale = connection.execute(
                """SELECT feature_id,feature_version,status FROM risk_set_member
                   WHERE risk_set_version=?""",
                (factor_set.risk_set_version,),
            ).fetchall()
            for factor_id, factor_version, status in stale:
                if (factor_id, factor_version) in active_keys:
                    continue
                if status != "candidate":
                    raise ValueError(
                        "NON_CANDIDATE_RISK_SET_MEMBER_REMOVAL_FORBIDDEN "
                        f"factor={factor_id} status={status}"
                    )
                connection.execute(
                    """UPDATE risk_set_member SET status='retired'
                       WHERE risk_set_version=? AND feature_id=? AND feature_version=?""",
                    (factor_set.risk_set_version, factor_id, factor_version),
                )
            for member in factor_set.members:
                if registry is None:
                    raise ValueError("RISK_SET_REGISTRY_REQUIRED")
                spec = registry.get(member.factor_id, member.factor_version).spec
                admitted_reason = {
                    "rationale": factor_set.rationale,
                    "selection_parameters": factor_set.selection_parameters,
                }
                connection.execute(statement, (
                    factor_set.risk_set_version,
                    member.factor_id,
                    member.factor_version,
                    spec.winsorize_method,
                    spec.standardize,
                    json.dumps(spec.orthogonalize_after),
                    json.dumps(admitted_reason, ensure_ascii=False, sort_keys=True),
                    factor_set.status,
                    factor_set.frozen_at.isoformat() if factor_set.frozen_at else None,
                ))
                connection.execute(
                    """DELETE FROM feature_assignment
                       WHERE feature_id=? AND feature_version=?
                         AND assigned_role='unassigned'""",
                    (member.factor_id, member.factor_version),
                )
                connection.execute(
                    """INSERT OR IGNORE INTO feature_assignment(
                         feature_id,feature_version,risk_set_version,assigned_role,
                         status,reason,assigned_at
                       ) VALUES (?,?,?,'basis','active',?,?)""",
                    (member.factor_id, member.factor_version, factor_set.risk_set_version,
                     "Explicit risk_set_member assignment.",
                    factor_set.frozen_at.isoformat() if factor_set.frozen_at else "1970-01-01"),
                )
            connection.execute("""UPDATE risk_set SET risk_basis_id=?,
              basis_acceptance_status=?,covariance_acceptance_status=?,pit_acceptance_status=?,
              basis_evidence_json=?,covariance_evidence_json=?,pit_evidence_json=? WHERE risk_set_version=?""",
              (acceptance.risk_basis_id,acceptance.basis_acceptance_status,acceptance.covariance_acceptance_status,
               acceptance.pit_acceptance_status,json.dumps(acceptance.basis_evidence),json.dumps(acceptance.covariance_evidence),
               json.dumps(acceptance.pit_evidence),factor_set.risk_set_version))

    def record_risk_acceptance(self, state) -> None:
        """Certify stages against a registered, matching basis; never change membership."""
        state = acceptance_from_mapping(vars(state))
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute('SELECT risk_basis_id,status FROM risk_set WHERE risk_set_version=?',
                                     (state.risk_set_version,)).fetchone()
            if row is None or row[0] != state.risk_basis_id or row[1]=='retired':
                raise ValueError('RISK_ACCEPTANCE_REGISTRY_IDENTITY')
            if not connection.execute('SELECT 1 FROM risk_set_member WHERE risk_set_version=?',
                                      (state.risk_set_version,)).fetchone():
                raise ValueError('RISK_ACCEPTANCE_MEMBERS_REQUIRED')
            old = registry_acceptance(connection,state.risk_set_version)
            if old.basis_consumption_allowed and (state.basis_acceptance_status != 'passed' or state.basis_evidence != old.basis_evidence):
                raise ValueError('RISK_ACCEPTED_BASIS_IMMUTABLE')
            connection.execute("""UPDATE risk_set SET basis_acceptance_status=?,covariance_acceptance_status=?,
              pit_acceptance_status=?,basis_evidence_json=?,covariance_evidence_json=?,pit_evidence_json=?
              WHERE risk_set_version=?""",(state.basis_acceptance_status,state.covariance_acceptance_status,
              state.pit_acceptance_status,json.dumps(state.basis_evidence),json.dumps(state.covariance_evidence),
              json.dumps(state.pit_evidence),state.risk_set_version))
