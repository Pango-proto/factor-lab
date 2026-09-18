"""Separate G6a/G6b/PIT certification; legacy frozen never implies certification."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

STAGES = ('basis', 'covariance', 'pit')
REQUIRED_CHECKS = {
    'basis': ('l1_lineage', 'g2_independent', 'g3_identity', 'semantic_costs', 'role_migration'),
    'covariance': ('generic_bias', 'stratified_bias', 'residual_correlation', 'estimator_lineage'),
    'pit': ('observed_before_decision', 'immutable_source_chain', 'real_daily_snapshots'),
}


def validate_evidence(ref, stage, version, basis_id):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('RISK_ACCEPTANCE_EVIDENCE_REFERENCE')
    path = Path(ref['path'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError('RISK_ACCEPTANCE_EVIDENCE_PATH')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('RISK_ACCEPTANCE_EVIDENCE_HASH')
    evidence = json.loads(path.read_text())
    if (evidence.get('schema') != 'risk_acceptance_evidence_v2' or
        evidence.get('stage') != stage or evidence.get('status') != 'passed' or
        evidence.get('risk_set_version') != version or evidence.get('risk_basis_id') != basis_id or
        not evidence.get('run_id') or evidence.get('holdout_evaluated') is not False):
        raise ValueError('RISK_ACCEPTANCE_EVIDENCE_SCOPE_OR_STATUS')
    if any(evidence.get('checks', {}).get(k) is not True for k in REQUIRED_CHECKS[stage]):
        raise ValueError('RISK_ACCEPTANCE_CHECKS_INCOMPLETE')
    if stage == 'pit' and evidence.get('historical_pit_verified') is not True:
        raise ValueError('RISK_ACCEPTANCE_REAL_PIT_REQUIRED')
    return evidence


@dataclass(frozen=True)
class RiskAcceptanceState:
    risk_set_version: int
    risk_basis_id: str | None = None
    basis_acceptance_status: str = 'unknown'
    covariance_acceptance_status: str = 'unknown'
    pit_acceptance_status: str = 'unknown'
    basis_evidence: dict | None = None
    covariance_evidence: dict | None = None
    pit_evidence: dict | None = None

    def __post_init__(self):
        for stage in STAGES:
            status = getattr(self, stage+'_acceptance_status')
            if status not in ('unknown', 'unavailable', 'failed', 'passed'):
                raise ValueError('RISK_ACCEPTANCE_STATUS')
            if status == 'passed':
                if not self.risk_basis_id or self.risk_set_version < 1:
                    raise ValueError('RISK_ACCEPTANCE_IDENTITY')
                validate_evidence(getattr(self, stage+'_evidence'), stage,
                                  self.risk_set_version, self.risk_basis_id)
        if self.covariance_acceptance_status == 'passed' and self.basis_acceptance_status != 'passed':
            raise ValueError('RISK_COVARIANCE_REQUIRES_BASIS')

    @property
    def basis_consumption_allowed(self):
        return self.basis_acceptance_status == 'passed'

    @property
    def risk_optimization_eligible(self):
        return all(getattr(self, s+'_acceptance_status') == 'passed' for s in STAGES)


def acceptance_from_mapping(row):
    return RiskAcceptanceState(**{k: row[k] for k in RiskAcceptanceState.__dataclass_fields__ if k in row})


def registry_acceptance(connection, version):
    cursor = connection.execute('SELECT * FROM risk_set WHERE risk_set_version=?', (version,))
    record = cursor.fetchone()
    if record is None:
        raise ValueError('RISK_ACCEPTANCE_UNKNOWN_SET')
    row = dict(zip([c[0] for c in cursor.description], record))
    for s in STAGES:
        row[s+'_evidence'] = json.loads(row.get(s+'_evidence_json') or 'null')
    if row['status'] == 'retired':
        return RiskAcceptanceState(version)
    return acceptance_from_mapping(row)


def ensure_acceptance_schema(connection):
    """Additive migration: preserve old rows and frozen semantics, default unknown."""
    columns = {r[1] for r in connection.execute('PRAGMA table_info(risk_set)')}
    connection.execute('''CREATE TABLE IF NOT EXISTS risk_acceptance_event(
        event_id TEXT PRIMARY KEY,risk_set_version INTEGER NOT NULL,state_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,FOREIGN KEY(risk_set_version) REFERENCES risk_set(risk_set_version))''')
    for event in ('UPDATE','DELETE'):
        connection.execute(f'''CREATE TRIGGER IF NOT EXISTS risk_acceptance_event_no_{event.lower()}
            BEFORE {event} ON risk_acceptance_event
            BEGIN SELECT RAISE(ABORT,'RISK_ACCEPTANCE_EVENT_APPEND_ONLY'); END''')
    for stage in STAGES:
        name = stage+'_acceptance_status'
        if name not in columns:
            connection.execute(f"ALTER TABLE risk_set ADD COLUMN {name} TEXT NOT NULL DEFAULT 'unknown' CHECK ({name} IN ('unknown','unavailable','failed','passed'))")
        name = stage+'_evidence_json'
        if name not in columns:
            connection.execute(f'ALTER TABLE risk_set ADD COLUMN {name} TEXT')
    # These supplement legacy frozen triggers. G6a also locks candidate membership.
    for event in ('INSERT', 'UPDATE', 'DELETE'):
        key = 'NEW' if event == 'INSERT' else 'OLD'
        versions=f'{key}.risk_set_version' if event!='UPDATE' else 'OLD.risk_set_version,NEW.risk_set_version'
        connection.execute(f'DROP TRIGGER IF EXISTS risk_member_basis_lock_{event.lower()}')
        connection.execute(f"""CREATE TRIGGER IF NOT EXISTS risk_member_basis_lock_{event.lower()}
          BEFORE {event} ON risk_set_member
          WHEN EXISTS(SELECT 1 FROM risk_set WHERE risk_set_version IN ({versions})
                      AND basis_acceptance_status='passed')
          BEGIN SELECT RAISE(ABORT,'ACCEPTED_RISK_BASIS_MEMBER_IMMUTABLE'); END""")
    connection.execute("""CREATE TRIGGER IF NOT EXISTS risk_basis_identity_lock
      BEFORE UPDATE ON risk_set WHEN OLD.basis_acceptance_status='passed' AND (
        NEW.risk_basis_id IS NOT OLD.risk_basis_id OR NEW.risk_set_version IS NOT OLD.risk_set_version OR
        NEW.weight_model_id IS NOT OLD.weight_model_id OR NEW.members_manifest_sha IS NOT OLD.members_manifest_sha OR
        NEW.expansion_manifest_sha IS NOT OLD.expansion_manifest_sha OR
        NEW.selection_parameters_json IS NOT OLD.selection_parameters_json OR
        NEW.basis_acceptance_status IS NOT OLD.basis_acceptance_status OR
        NEW.basis_evidence_json IS NOT OLD.basis_evidence_json)
      BEGIN SELECT RAISE(ABORT,'ACCEPTED_RISK_BASIS_IMMUTABLE'); END""")
    connection.execute("""CREATE TRIGGER IF NOT EXISTS risk_basis_delete_lock
      BEFORE DELETE ON risk_set WHEN OLD.basis_acceptance_status='passed'
      BEGIN SELECT RAISE(ABORT,'ACCEPTED_RISK_BASIS_DELETE_FORBIDDEN'); END""")
