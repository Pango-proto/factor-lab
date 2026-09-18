"""Single-source G3 gate identity shared by builders, attestation, and CI."""

from __future__ import annotations

from pathlib import Path


G3_GATE_VERSION = "g3_condition_gate_v2"
G3_GATE_CONFIG_PATH = Path("config/g3_numerical_invariance_protocol_v1.json")
G3_ATTESTATION_CONFIG_PATH = Path("config/g3_canonicalization_attestation_v1.json")
