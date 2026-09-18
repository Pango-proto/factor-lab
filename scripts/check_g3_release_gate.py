#!/usr/bin/env python3
"""Fail CI unless the active G3 pointer is v2-attested and termination rules exist."""

from __future__ import annotations

import json
from pathlib import Path

from factor_matrix.storage import file_sha256


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
CURRENT_PATH = DATA / "gold" / "l2b_risk_only" / "_CURRENT.json"
GATE_CONFIG = PROJECT / "config" / "g3_numerical_invariance_protocol_v1.json"
VERSIONING = PROJECT / "config" / "g3_gate_versioning_v1.json"


def check(project: Path = PROJECT) -> list[str]:
    data = project / "data"
    current_path = data / "gold" / "l2b_risk_only" / "_CURRENT.json"
    failures: list[str] = []
    if not current_path.exists():
        return ["G3_CURRENT_POINTER_MISSING"]
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("status") != "active":
        failures.append("G3_CURRENT_NOT_ACTIVE")
    if current.get("gate_version") != "g3_condition_gate_v2":
        failures.append("G3_CURRENT_GATE_VERSION_NOT_V2")
    manifest_path = data / current["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != current.get("run_id") or manifest.get("status") != "passed":
        failures.append("G3_CURRENT_MANIFEST_NOT_MATCHING_PASSED")
    if manifest.get("risk_basis_id") != current.get("risk_basis_id"):
        failures.append("G3_CURRENT_RISK_BASIS_MISMATCH")
    l1_pointer = data / "gold/risk_exposure_matrix/_CURRENT.json"
    if not l1_pointer.exists():
        failures.append("L1_CURRENT_POINTER_MISSING")
    else:
        l1_current = json.loads(l1_pointer.read_text(encoding="utf-8"))
        if current.get("l1_run_id") != l1_current.get("run_id") or manifest.get("l1_run_id") != l1_current.get("run_id"):
            failures.append("G3_CURRENT_L1_RUN_MISMATCH")
        if current.get("risk_basis_id") != l1_current.get("risk_basis_id"):
            failures.append("G3_CURRENT_L1_RISK_BASIS_MISMATCH")
        l1_manifest_path = data / l1_current["manifest"]
        if not l1_manifest_path.exists():
            failures.append("L1_CURRENT_MANIFEST_MISSING")
        else:
            l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
            if l1_manifest.get("run_id") != l1_current.get("run_id") or l1_manifest.get("risk_basis_id") != l1_current.get("risk_basis_id"):
                failures.append("L1_CURRENT_MANIFEST_MISMATCH")
            if manifest.get("input_hashes", {}).get("l1_history_manifest") != file_sha256(l1_manifest_path):
                failures.append("G3_CURRENT_L1_MANIFEST_HASH_MISMATCH")
    if current.get("gate_config_sha") != file_sha256(project / GATE_CONFIG.relative_to(PROJECT)):
        failures.append("G3_CURRENT_GATE_CONFIG_HASH_MISMATCH")

    attestation_rel = current.get("gate_attestation_manifest")
    if not attestation_rel:
        failures.append("G3_CURRENT_ATTESTATION_MISSING")
    else:
        attestation_path = data / attestation_rel
        if not attestation_path.exists():
            failures.append("G3_CURRENT_ATTESTATION_FILE_MISSING")
        else:
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            if attestation.get("status") != "passed":
                failures.append("G3_ATTESTATION_NOT_PASSED")
            if attestation.get("gate_version") != "g3_condition_gate_v2":
                failures.append("G3_ATTESTATION_GATE_VERSION_NOT_V2")
            if attestation.get("g3_run_id") != current.get("run_id"):
                failures.append("G3_ATTESTATION_RUN_MISMATCH")
            if attestation.get("identity", {}).get("g3_manifest_sha") != file_sha256(manifest_path):
                failures.append("G3_ATTESTATION_MANIFEST_HASH_MISMATCH")
            if current.get("gate_attestation_sha256") != file_sha256(attestation_path):
                failures.append("G3_CURRENT_ATTESTATION_HASH_MISMATCH")

    versioning = json.loads((project / VERSIONING.relative_to(PROJECT)).read_text(encoding="utf-8"))
    if versioning["current_gate_version"] != "g3_condition_gate_v2":
        failures.append("G3_VERSIONING_CURRENT_NOT_V2")
    if versioning["versions"]["g3_condition_gate_v1"]["status"] != "superseded":
        failures.append("G3_V1_NOT_SUPERSEDED")

    termination = json.loads(
        (project / "config" / "g4_termination_rule_v1.json").read_text(encoding="utf-8")
    )
    required_clauses = {"G4.1_weight", "G4.1b_liquidity", "condition_number", "G4.2_membership", "G4.3_residual_correlation", "G4.4_listing_age", "statistical_factors"}
    if required_clauses - set(termination.get("termination_clauses", {})):
        failures.append("G4_TERMINATION_CLAUSES_INCOMPLETE")

    g6 = json.loads((project / "config" / "g6_freeze_protocol_v1.json").read_text(encoding="utf-8"))
    if g6.get("publication_semantics", {}).get("g6_action") != "publish risk_set_version=1":
        failures.append("G6_PUBLICATION_SEMANTICS_NOT_VERSIONED")

    # Any explicit v1 reference in a downstream Gold manifest is invalid unless
    # the manifest records that it has been superseded.
    for path in sorted((data / "gold").rglob("_MANIFEST.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("gate_version") == "g3_condition_gate_v1" and payload.get("status") != "superseded":
            failures.append(f"G3_V1_DOWNSTREAM_NOT_SUPERSEDED:{path.relative_to(project)}")
    return failures


def main() -> None:
    failures = check()
    if failures:
        raise SystemExit("\n".join(failures))
    print("G3 release gate passed")


if __name__ == "__main__":
    main()
