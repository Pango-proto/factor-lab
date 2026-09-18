"""Static guard for frozen configuration surfaces and declared consumers."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "config/frozen_config_consumption_v1.json"


def main() -> int:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    failures: list[str] = []
    for binding in policy["bindings"]:
        config_path = PROJECT / binding["config"]
        consumer_path = PROJECT / binding["consumer"]
        if not config_path.exists():
            failures.append(f"CONFIG_MISSING {binding['config']}")
            continue
        if not consumer_path.exists():
            failures.append(f"CONSUMER_MISSING {binding['consumer']}")
            continue
        text = consumer_path.read_text(encoding="utf-8")
        for marker in binding["required_markers"]:
            if marker not in text:
                failures.append(
                    f"FROZEN_CONFIG_CONSUMER_MARKER_MISSING "
                    f"{binding['config']}{binding['json_pointer']} "
                    f"consumer={binding['consumer']} marker={marker}"
                )
    if failures:
        print("\n".join(failures))
        return 1
    print(f"frozen config consumption passed: {len(policy['bindings'])} bindings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
