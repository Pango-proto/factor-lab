"""Single loader for frozen risk-descriptor construction parameters."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "risk_descriptor_parameters_v1.json"


@lru_cache(maxsize=1)
def load_risk_descriptor_parameters(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("RISK_DESCRIPTOR_PARAMETER_SCHEMA_UNSUPPORTED")
    definitions = payload.get("definitions")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("RISK_DESCRIPTOR_DEFINITIONS_REQUIRED")
    return payload


def descriptor_parameters(factor_id: str) -> dict[str, Any]:
    payload = load_risk_descriptor_parameters()
    try:
        values = payload["definitions"][factor_id]
    except KeyError as exc:
        raise KeyError(f"RISK_DESCRIPTOR_PARAMETERS_MISSING factor={factor_id}") from exc
    return {"parameter_config_id": payload["config_id"], **values}
