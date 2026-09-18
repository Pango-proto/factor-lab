"""A passed G3 cannot remain current against a different L1 publication."""

import json
from pathlib import Path

import pytest

from factor_matrix.storage import file_sha256
from scripts.check_g3_release_gate import check


@pytest.fixture
def published_project(tmp_path):
    project = Path(__file__).resolve().parents[1]

    def write(relative, payload):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    for name in (
        "g3_numerical_invariance_protocol_v1.json", "g3_gate_versioning_v1.json",
        "g4_termination_rule_v1.json", "g6_freeze_protocol_v1.json",
    ):
        write(f"config/{name}", json.loads((project / "config" / name).read_text()))
    l1 = write("data/gold/risk_exposure_matrix/run_id=l1/_MANIFEST.json", {
        "run_id": "l1", "risk_basis_id": "basis",
    })
    write("data/gold/risk_exposure_matrix/_CURRENT.json", {
        "run_id": "l1", "risk_basis_id": "basis",
        "manifest": str(l1.relative_to(tmp_path / "data")),
    })
    g3 = write("data/gold/l2b_risk_only/run_id=g3/_MANIFEST.json", {
        "run_id": "g3", "l1_run_id": "l1", "risk_basis_id": "basis", "status": "passed",
        "input_hashes": {"l1_history_manifest": file_sha256(l1)},
    })
    attestation = write("data/diagnostics/attestation/_MANIFEST.json", {
        "status": "passed", "gate_version": "g3_condition_gate_v2", "g3_run_id": "g3",
        "identity": {"g3_manifest_sha": file_sha256(g3)},
    })
    write("data/gold/l2b_risk_only/_CURRENT.json", {
        "run_id": "g3", "l1_run_id": "l1", "risk_basis_id": "basis", "status": "active",
        "gate_version": "g3_condition_gate_v2",
        "gate_config_sha": file_sha256(tmp_path / "config/g3_numerical_invariance_protocol_v1.json"),
        "manifest": str(g3.relative_to(tmp_path / "data")),
        "gate_attestation_manifest": str(attestation.relative_to(tmp_path / "data")),
        "gate_attestation_sha256": file_sha256(attestation),
    })
    assert check(tmp_path) == []
    return tmp_path


@pytest.mark.parametrize(("field", "failure"), [
    ("run_id", "G3_CURRENT_L1_RUN_MISMATCH"),
    ("risk_basis_id", "G3_CURRENT_L1_RISK_BASIS_MISMATCH"),
])
def test_release_gate_rejects_stale_l1_binding(published_project, field, failure):
    pointer = published_project / "data/gold/risk_exposure_matrix/_CURRENT.json"
    payload = json.loads(pointer.read_text())
    payload[field] = "changed"
    pointer.write_text(json.dumps(payload))
    assert failure in check(published_project)


def test_release_gate_rejects_l1_manifest_mutation(published_project):
    manifest = published_project / "data/gold/risk_exposure_matrix/run_id=l1/_MANIFEST.json"
    payload = json.loads(manifest.read_text())
    payload["changed_input"] = True
    manifest.write_text(json.dumps(payload))
    assert "G3_CURRENT_L1_MANIFEST_HASH_MISMATCH" in check(published_project)
