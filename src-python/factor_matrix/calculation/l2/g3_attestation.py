"""Append-only G3 canonicalization attestation and current-pointer binding."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ...storage import DataLake, file_sha256, json_hash, utc_now
from .g3_gate import (
    G3_ATTESTATION_CONFIG_PATH, G3_GATE_CONFIG_PATH, G3_GATE_VERSION,
)
from .g3_runner import load_current_g3_manifest


def run_g3_canonicalization_attestation(
    lake: DataLake,
    *,
    config_path: Path = G3_ATTESTATION_CONFIG_PATH,
    gate_config_path: Path = G3_GATE_CONFIG_PATH,
) -> Path:
    """Create the immutable attestation for the currently active G3 run."""
    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    if current.get("run_id") != g3.get("run_id"):
        raise RuntimeError("G3_ATTESTATION_CURRENT_MANIFEST_MISMATCH")

    l1_current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    l1_current = json.loads(l1_current_path.read_text(encoding="utf-8"))
    l1_manifest_path = lake.root / l1_current["manifest"]
    l1 = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    if g3.get("l1_run_id") != l1.get("run_id") or g3.get("risk_basis_id") != l1.get("risk_basis_id"):
        raise RuntimeError("G3_ATTESTATION_LINEAGE_MISMATCH")
    if g3.get("risk_basis_id") != "risk_basis_5649d34a30c03cef":
        raise RuntimeError("G3_ATTESTATION_UNEXPECTED_RISK_BASIS")

    evidence = json.loads(config_path.read_text(encoding="utf-8"))
    gate_config_sha = file_sha256(gate_config_path)
    identity: dict[str, Any] = {
        "gate_version": G3_GATE_VERSION,
        "gate_config_sha": gate_config_sha,
        "attestation_config_sha": file_sha256(config_path),
        "g3_manifest_sha": file_sha256(g3_manifest_path),
        "l1_manifest_sha": file_sha256(l1_manifest_path),
        "risk_basis_id": g3["risk_basis_id"],
        "g3_run_id": g3["run_id"],
        "l1_run_id": l1["run_id"],
        "weight_metric_contract_sha": file_sha256(Path("config/weight_metric_contract_v1.json")),
    }
    run_id = f"g3_canonicalization_attestation_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g3_numerical_invariance_attestation" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "g3_canonicalization_attestation",
        "status": "passed",
        "created_at": utc_now().isoformat(),
        "gate_version": G3_GATE_VERSION,
        "gate_config_sha": gate_config_sha,
        "g3_run_id": g3["run_id"],
        "l1_run_id": l1["run_id"],
        "risk_basis_id": g3["risk_basis_id"],
        "identity": identity,
        "input_hashes": {
            "g3_manifest": identity["g3_manifest_sha"],
            "l1_manifest": identity["l1_manifest_sha"],
            "weight_metric_contract_v1": identity["weight_metric_contract_sha"],
            "gate_config": gate_config_sha,
            "attestation_config": identity["attestation_config_sha"],
        },
        "evidence": evidence,
        "decision": "passed_clean_reparameterization",
    }
    return lake.write_immutable_json(manifest_path, manifest)


def bind_g3_attestation_to_current(lake: DataLake, attestation_path: Path) -> Path:
    """Bind a passed immutable attestation to the mutable G3 current pointer."""
    current_path = lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("status") != "passed" or attestation.get("gate_version") != G3_GATE_VERSION:
        raise RuntimeError("G3_ATTESTATION_NOT_PASSED")
    if attestation.get("g3_run_id") != g3.get("run_id"):
        raise RuntimeError("G3_ATTESTATION_RUN_MISMATCH")
    if attestation.get("identity", {}).get("g3_manifest_sha") != file_sha256(g3_manifest_path):
        raise RuntimeError("G3_ATTESTATION_MANIFEST_HASH_MISMATCH")

    updated = {
        **current,
        "status": "active",
        "gate_version": G3_GATE_VERSION,
        "gate_config_sha": attestation["gate_config_sha"],
        "gate_attestation_manifest": str(attestation_path.relative_to(lake.root)),
        "gate_attestation_sha256": file_sha256(attestation_path),
        "gate_status": "passed",
        "risk_basis_id": g3["risk_basis_id"],
        "l1_run_id": g3["l1_run_id"],
        "updated_at": utc_now().isoformat(),
    }
    temporary = current_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, current_path)
    return current_path
