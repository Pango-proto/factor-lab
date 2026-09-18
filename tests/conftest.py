"""Portable synthetic test projects; never substitute production audit evidence."""
import json
from pathlib import Path
import shutil

import pytest

from factor_matrix.storage import file_sha256


@pytest.fixture(scope='session')
def core6_design_project(tmp_path_factory):
    """Exercise the real loader with explicit artificial evidence in a temp project.

    The repository's approved design and hashes are not changed. This fixture
    validates mechanics, not historical approval or real-data admission.
    """
    source = Path(__file__).resolve().parents[1]
    project = tmp_path_factory.mktemp('synthetic_core6_project')
    shutil.copytree(source / 'config', project / 'config')
    path = project / 'config/risk_core6_design_v2.json'
    spec = json.loads(path.read_text())
    inherited = {}
    for index, (relative, expected) in enumerate(spec['inherited_file_hashes'].items()):
        if relative.startswith('config/'):
            inherited[relative] = expected
        else:
            relative = f'fixture-evidence/evidence-{index}.txt'
            evidence = project / relative
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(f'SYNTHETIC TEST EVIDENCE {index}; no real approval.\n')
            inherited[relative] = file_sha256(evidence)
    spec['inherited_file_hashes'] = inherited
    path.write_text(json.dumps(spec, indent=2) + '\n')
    return project
