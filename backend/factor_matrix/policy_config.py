from __future__ import annotations

import json
from pathlib import Path

from .research import MatrixConfig


def load_market_domain_policy(path: Path) -> MatrixConfig:
    """Load the versioned market-domain policy without framework defaults."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {field.name for field in MatrixConfig.__dataclass_fields__.values()}
    observed = set(payload)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            f"MARKET_DOMAIN_POLICY_FIELDS_MISMATCH missing={missing} extra={extra}"
        )
    return MatrixConfig(**payload)
