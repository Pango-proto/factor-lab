"""Immutable membership-only comparison of listing-age universe variants."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..calculation.l1 import summarize_variant_sensitivity
from ..storage import DataLake, json_hash, source_tree_hash, utc_now


def run_listing_variant_sensitivity(
    lake: DataLake,
    *,
    universe_metadata_path: Path,
    listing_window_manifest_path: Path,
    policy_path: Path = Path("config/new_listing_policy_v1.json"),
) -> Path:
    universe_metadata = json.loads(universe_metadata_path.read_text(encoding="utf-8"))
    window_manifest = json.loads(listing_window_manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if universe_metadata.get("job") != "tradable_universe_v1":
        raise ValueError("SENSITIVITY_REQUIRES_TRADABLE_UNIVERSE_METADATA")
    if window_manifest.get("job") != "new_listing_window_diagnostics":
        raise ValueError("SENSITIVITY_REQUIRES_LISTING_WINDOW_MANIFEST")
    identity = {
        "universe_run_id": universe_metadata["run_id"],
        "listing_window_run_id": window_manifest["run_id"],
        "code_hash": source_tree_hash(),
        "new_listing_policy_sha": json_hash(policy),
    }
    run_id = f"listing_variant_sensitivity_{json_hash(identity)[:16]}"
    manifest_path = lake.manifests / f"{run_id}.json"
    if manifest_path.exists():
        return manifest_path

    universe_path = lake.root / universe_metadata["artifact"]
    window_path = (
        lake.root
        / window_manifest["outputs"]["new_listing_window_v1"]["path"]
    )
    result = summarize_variant_sensitivity(
        pl.scan_parquet(universe_path),
        derived_listing_window=pl.read_parquet(window_path),
    )
    output_root = (
        lake.root / "diagnostics" / "listing_variant_sensitivity" / f"run_id={run_id}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    output_path = output_root / "listing_variant_sensitivity_v1.parquet"
    result.write_parquet(output_path)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "listing_variant_sensitivity_diagnostics",
        "execution_status": "passed",
        "created_at": utc_now().isoformat(),
        "identity": identity,
        "inputs": {
            "tradable_universe": lake.artifact_record(universe_path),
            "tradable_universe_metadata": lake.artifact_record(universe_metadata_path),
            "new_listing_window": lake.artifact_record(window_path),
            "new_listing_window_manifest": lake.artifact_record(
                listing_window_manifest_path
            ),
        },
        "outputs": {
            "listing_variant_sensitivity_v1": lake.artifact_record(output_path)
        },
        "scope": "membership_only_no_exposure_formula_changes",
        "formal_exposure_publish_gate_status": (
            "open_for_frozen_d0"
            if policy["d0_model_exclusion"]["status"] == "frozen"
            else "blocked_d0_not_frozen"
        ),
        "formal_universe_variant": policy["formal_exposure_publish_gate"][
            "formal_universe_variant"
        ],
        "diagnostic_is_publish_authorization": False,
        "summary": result.to_dicts(),
    }
    return lake.write_immutable_json(manifest_path, manifest)
