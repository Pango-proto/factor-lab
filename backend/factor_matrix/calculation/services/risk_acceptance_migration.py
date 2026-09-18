"""Audited additive metadata migration with a pre-migration SQLite backup."""
from pathlib import Path
import json
import sqlite3
from ...storage import file_sha256,json_hash,utc_now,DataLake
from ...risk_model.acceptance import ensure_acceptance_schema


def migrate_acceptance(*,registry_path,project_root,approval_path,approval_sha256,output_root):
    if file_sha256(approval_path)!=approval_sha256:raise ValueError('ACCEPTANCE_MIGRATION_APPROVAL_HASH')
    approval=json.loads(approval_path.read_text())
    if approval.get('status')!='approved' or 'G6a_G6b_state_split' not in approval.get('scope',[]):
        raise ValueError('ACCEPTANCE_MIGRATION_SCOPE')
    if file_sha256(project_root/approval['proposal_path'])!=approval['proposal_sha256']:
        raise ValueError('ACCEPTANCE_MIGRATION_PROPOSAL_CHANGED')
    if not registry_path.is_file():raise ValueError('ACCEPTANCE_MIGRATION_REGISTRY_MISSING')
    identity={'registry_before_sha256':file_sha256(registry_path),'approval_sha256':approval_sha256,
              'service_sha256':file_sha256(Path(__file__)),
              'schema_code_sha256':file_sha256(Path(__file__).parents[2]/'risk_model'/'acceptance.py')}
    out=output_root/('run_id=acceptance_migration_'+json_hash(identity)[:16]);out.mkdir(parents=True,exist_ok=False)
    before={};columns={}
    with sqlite3.connect(f'file:{registry_path.resolve()}?mode=ro',uri=True) as source:
        source.execute('BEGIN')
        with sqlite3.connect(out/'registry_before.sqlite') as backup:source.backup(backup)
        for table in ['risk_set','risk_set_member','family_root_declaration','research_attempt','alpha_assertion']:
            columns[table]=[r[1] for r in source.execute(f'PRAGMA table_info({table})')]
            before[table]=source.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
    with sqlite3.connect(registry_path) as c:
        c.execute('BEGIN IMMEDIATE')
        ensure_acceptance_schema(c)
        for table,names in columns.items():
            selected=','.join('"'+name+'"' for name in names)
            if c.execute(f'SELECT {selected} FROM {table} ORDER BY rowid').fetchall()!=before[table]:
                raise ValueError('ACCEPTANCE_MIGRATION_EXISTING_VALUES_CHANGED')
        states=c.execute('SELECT risk_set_version,basis_acceptance_status,covariance_acceptance_status,pit_acceptance_status FROM risk_set').fetchall()
    manifest={'schema_version':2,'status':'additive_migration_passed','identity':identity,
              'created_at':utc_now().isoformat(),'old_table_rows_unchanged':{k:len(v) for k,v in before.items()},
              'states':states,'backup_sha256':file_sha256(out/'registry_before.sqlite'),
              'registry_after_sha256':file_sha256(registry_path),'current_changed':False,'production_promoted':False}
    return DataLake(output_root).write_immutable_json(out/'_MANIFEST.json',manifest)
