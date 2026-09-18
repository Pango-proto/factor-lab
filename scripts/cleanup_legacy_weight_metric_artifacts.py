"""Remove pre-contract L1/G2/G3 artifacts after the weight-metric migration.

The cleanup is deliberately manifest-driven: only run directories lacking the
new risk metric/basis identity are eligible.  Current pointers are checked
before deletion so a stale pointer cannot be created accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RETENTION_POLICY = PROJECT / "config" / "diagnostic_retention_policy_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_run_dirs(base: Path, *, require_basis: bool) -> list[Path]:
    selected: list[Path] = []
    for manifest_path in sorted(base.glob("run_id=*/_MANIFEST.json")):
        manifest = _load(manifest_path)
        missing_metric = not manifest.get("risk_metric_id")
        missing_basis = require_basis and not manifest.get("risk_basis_id")
        if missing_metric or missing_basis:
            selected.append(manifest_path.parent.resolve())
    return selected


def _current_manifest_paths() -> set[Path]:
    paths: set[Path] = set()
    for current_path in (
        DATA / "gold/risk_exposure_matrix/_CURRENT.json",
        DATA / "gold/l2b_risk_only/_CURRENT.json",
    ):
        current = _load(current_path)
        for field in ("manifest", "diagnostics_manifest"):
            value = current.get(field)
            if value:
                paths.add((DATA / str(value)).resolve())
    return paths


def _metadata_mirrors(run_ids: set[str]) -> list[Path]:
    selected: list[Path] = []
    base = DATA / "metadata/manifests"
    for path in sorted(base.glob("*.json")):
        try:
            payload = _load(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if payload.get("run_id") in run_ids:
            selected.append(path.resolve())
    return selected


def _assert_selection_evidence_is_not_a_cleanup_target(targets: list[Path]) -> None:
    """Selection/freeze evidence is permanently retained by policy."""
    protected_tokens = ("selection", "freeze", "experiment_registry")
    for target in targets:
        manifest_path = target / "_MANIFEST.json"
        if not manifest_path.exists():
            continue
        payload = _load(manifest_path)
        job = str(payload.get("job", "")).lower()
        if any(token in job for token in protected_tokens):
            raise RuntimeError(
                "REFUSE_DELETE_SELECTION_FREEZE_EVIDENCE "
                f"{target} policy={RETENTION_POLICY}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    l1_dirs = _legacy_run_dirs(
        DATA / "gold/risk_exposure_matrix", require_basis=False
    )
    g2_dirs = _legacy_run_dirs(DATA / "diagnostics/l1_history", require_basis=True)
    g3_dirs = _legacy_run_dirs(DATA / "gold/l2b_risk_only", require_basis=True)
    current_manifests = _current_manifest_paths()
    targets = l1_dirs + g2_dirs + g3_dirs
    if not RETENTION_POLICY.exists():
        raise RuntimeError(f"RETENTION_POLICY_MISSING {RETENTION_POLICY}")
    _assert_selection_evidence_is_not_a_cleanup_target(targets)
    target_manifests = {path / "_MANIFEST.json" for path in targets}
    overlap = current_manifests & target_manifests
    if overlap:
        raise RuntimeError(
            "REFUSE_DELETE_CURRENT_MANIFESTS "
            + ",".join(str(path) for path in sorted(overlap))
        )

    for target in targets:
        if DATA.resolve() not in target.parents or not target.name.startswith("run_id="):
            raise RuntimeError(f"UNSAFE_CLEANUP_TARGET {target}")

    l1_run_ids = {path.name.removeprefix("run_id=") for path in l1_dirs}
    metadata_files = _metadata_mirrors(l1_run_ids)
    target_strings = sorted(str(path.relative_to(PROJECT)) for path in targets + metadata_files)
    digest = hashlib.sha256("\n".join(target_strings).encode("utf-8")).hexdigest()
    summary = {
        "schema_version": "legacy_weight_metric_cleanup_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "selection_rule": {
            "l1": "missing risk_metric_id",
            "g2_g3": "missing risk_metric_id or risk_basis_id",
        },
        "deleted_or_selected_counts": {
            "l1_run_directories": len(l1_dirs),
            "g2_run_directories": len(g2_dirs),
            "g3_run_directories": len(g3_dirs),
            "l1_metadata_mirrors": len(metadata_files),
        },
        "target_list_sha256": digest,
        "current_manifest_overlap": 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute:
        return

    for path in targets:
        shutil.rmtree(path)
    for path in metadata_files:
        path.unlink()
    audit_path = DATA / "diagnostics/weight_metric_migration/legacy_cleanup_v1.json"
    audit_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
