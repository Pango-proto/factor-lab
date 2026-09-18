"""Current-pointer resolution for calculation artifacts.

LocalArtifactStore publishes immutable runs but writes no pointer, so a
downstream operation has no way to discover the run it should consume. Without
this, every consumer would end up scanning directories and guessing at the
newest one, which the orchestration contract forbids by name.

Two properties matter and are enforced here rather than by convention:

  - the pointer is only written after a run has already published, so a failed
    run cannot become current. The executor raises before publish on a quality
    failure, so a failed run leaves no artifact at all, and the pointer keeps
    addressing the last good one.
  - the pointer records the checksum it was written against. LocalArtifactStore
    re-verifies that checksum on load, so a silently rewritten artifact fails
    loudly instead of being consumed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ...storage import DataLake, file_sha256, utc_now
from ..core.contracts import ArtifactKey, ArtifactRef, QualityStatus


class CurrentPointer:
    def __init__(self, lake: DataLake) -> None:
        self._lake = lake

    def _path(self, artifact_type: str, version: str, scope_key: str) -> Path:
        return (
            self._lake.root / "gold" / "calculations" / "_current"
            / f"artifact={artifact_type}" / f"version={version}"
            / f"scope={scope_key.replace(':', '_')}.json"
        )

    def publish(self, reference: ArtifactRef) -> Path:
        if reference.quality_status is not QualityStatus.PASSED:
            raise RuntimeError(
                f"CURRENT_POINTER_REQUIRES_PASSED_QUALITY run={reference.key.run_id}"
            )
        path = self._path(
            reference.key.artifact_type, reference.key.version, reference.key.scope_key
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "artifact_type": reference.key.artifact_type,
            "artifact_version": reference.key.version,
            "run_id": reference.key.run_id,
            "as_of_date": reference.key.as_of_date.isoformat(),
            "scope_key": reference.key.scope_key,
            "uri": reference.uri,
            "checksum": reference.checksum,
            "quality_status": reference.quality_status.value,
            "tags": sorted(reference.tags),
            "published_at": utc_now().isoformat(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def resolve(self, artifact_type: str, version: str, scope_key: str) -> ArtifactRef:
        path = self._path(artifact_type, version, scope_key)
        if not path.exists():
            raise FileNotFoundError(
                f"CURRENT_POINTER_MISSING artifact={artifact_type} "
                f"version={version} scope={scope_key}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata_path = self._lake.root / payload["uri"]
        observed = file_sha256(metadata_path)
        if observed != payload["checksum"]:
            raise RuntimeError(
                f"CURRENT_POINTER_CHECKSUM_DRIFT artifact={artifact_type} "
                f"run={payload['run_id']}"
            )
        return ArtifactRef(
            key=ArtifactKey(
                artifact_type=payload["artifact_type"],
                version=payload["artifact_version"],
                run_id=payload["run_id"],
                as_of_date=date.fromisoformat(payload["as_of_date"]),
                scope_key=payload["scope_key"],
            ),
            uri=payload["uri"],
            checksum=payload["checksum"],
            quality_status=QualityStatus(payload["quality_status"]),
            tags=frozenset(payload["tags"]),
        )
