from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now


L2B_PRODUCTS = (
    "factor_returns_v1", "specific_returns_v1",
    "factor_regression_quality_v1", "cross_section_stats_v1",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _validate_unique(frame: pl.DataFrame, keys: tuple[str, ...], product: str) -> None:
    if frame.height != frame.unique(subset=list(keys)).height:
        raise ValueError(f"L2_PRODUCT_PRIMARY_KEY_DUPLICATE product={product}")


def validate_l2b_products(
    products: Mapping[str, pl.DataFrame], *, maximum_invalid_date_ratio: float,
    maximum_condition_number: float,
) -> dict[str, object]:
    if set(products) != set(L2B_PRODUCTS):
        raise ValueError("L2B_PRODUCT_SET_INCOMPLETE")
    factor = products["factor_returns_v1"]
    specific = products["specific_returns_v1"]
    quality = products["factor_regression_quality_v1"]
    stats = products["cross_section_stats_v1"]
    _validate_unique(factor, ("trade_date", "factor_id", "regression_mode"), "factor_returns_v1")
    _validate_unique(specific, ("trade_date", "asset_id", "regression_mode"), "specific_returns_v1")
    _validate_unique(quality, ("trade_date", "regression_mode"), "factor_regression_quality_v1")
    _validate_unique(stats, ("trade_date", "board_id", "regression_mode"), "cross_section_stats_v1")
    dates = set(quality.get_column("trade_date").to_list())
    for product, frame in (("factor_returns_v1", factor), ("specific_returns_v1", specific),
                           ("cross_section_stats_v1", stats)):
        if set(frame.get_column("trade_date").to_list()) != dates:
            raise ValueError(f"L2B_PRODUCT_DATE_COVERAGE_MISMATCH product={product}")
    if factor.height != sum(quality.get_column("factor_count")):
        raise ValueError("L2B_FACTOR_RETURN_ROW_COUNT_MISMATCH")
    if specific.height != sum(quality.get_column("sample_count")):
        raise ValueError("L2B_SPECIFIC_RETURN_ROW_COUNT_MISMATCH")
    if quality.filter(
        (pl.col("matrix_rank") != pl.col("expected_matrix_rank"))
        | ~pl.col("constraint_identification_passed")
    ).filter(pl.col("status") != "invalid").height:
        raise ValueError("L2B_RANK_FAILURE_NOT_MARKED_INVALID")
    invalid_count = quality.filter(pl.col("status") == "invalid").height
    invalid_ratio = invalid_count / quality.height if quality.height else 1.0
    if invalid_ratio > maximum_invalid_date_ratio:
        raise ValueError(f"L2B_INVALID_DATE_RATIO_EXCEEDED ratio={invalid_ratio:.6f}")
    passed_quality = quality.filter(pl.col("status") == "passed")
    maximum_constraint_error = passed_quality.get_column("constraint_error").max()
    maximum_identity_error = passed_quality.get_column("regression_identity_error").max()
    if maximum_constraint_error is None or maximum_constraint_error >= 1e-10:
        raise ValueError("L2B_CONSTRAINT_IDENTITY_FAILED")
    if maximum_identity_error is None or maximum_identity_error >= 1e-10:
        raise ValueError("L2B_REGRESSION_IDENTITY_FAILED")
    if quality.filter(
        (pl.col("condition_number") > maximum_condition_number) & (pl.col("status") != "invalid")
    ).height:
        raise ValueError("L2B_HIGH_CONDITION_DATE_NOT_INVALID")
    return {
        "date_count": len(dates),
        "invalid_date_count": invalid_count,
        "invalid_date_ratio": invalid_ratio,
        "maximum_constraint_error": maximum_constraint_error,
        "maximum_regression_identity_error": maximum_identity_error,
        "row_counts": {product: frame.height for product, frame in products.items()},
    }


class L2AtomicPublisher:
    """Content-addressed, all-products-before-pointer publication for L2 runs."""

    def __init__(self, lake: DataLake) -> None:
        self._lake = lake

    def publish_l2b(
        self,
        *,
        products: Mapping[str, pl.DataFrame],
        mode: str,
        configuration: Mapping[str, object],
        risk_set_version: int,
        input_paths: Mapping[str, Path],
        date_range: tuple[date, date],
        derived_params: Mapping[str, object],
        code_sha: str | None = None,
    ) -> str:
        unresolved = tuple(derived_params.get("unresolved_required", ()))
        if unresolved:
            raise ValueError(
                f"L2_REQUIRED_DERIVED_PARAMETERS_UNRESOLVED keys={','.join(unresolved)}"
            )
        quality_summary = validate_l2b_products(
            products,
            maximum_invalid_date_ratio=float(derived_params["maximum_invalid_date_ratio"]),
            maximum_condition_number=float(derived_params["maximum_condition_number"]),
        )
        config_sha = json_hash(dict(configuration))
        resolved_code_sha = code_sha or source_tree_hash()
        input_hashes = {
            name: file_sha256(path.resolve()) for name, path in sorted(input_paths.items())
        }
        identity = {
            "mode": mode,
            "config_sha": config_sha,
            "code_sha": resolved_code_sha,
            "risk_set_version": risk_set_version,
            "input_hashes": input_hashes,
            "date_range": [item.isoformat() for item in date_range],
            "derived_params": dict(derived_params),
        }
        run_id = f"l2b_{json_hash(identity)[:20]}"
        created_at = utc_now().isoformat()
        final_directories: dict[str, Path] = {}
        for product, frame in sorted(products.items()):
            root = self._lake.root / "gold" / product
            final = root / f"run_id={run_id}"
            final_directories[product] = final
            if final.exists():
                existing = json.loads((final / "_MANIFEST.json").read_text(encoding="utf-8"))
                if existing.get("identity") != identity:
                    raise RuntimeError(f"L2_IDEMPOTENCY_CONFLICT product={product}")
                continue
            staging = root / f".run_id={run_id}.tmp-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            table_records = []
            for key, partition in frame.partition_by("trade_date", as_dict=True).items():
                trade_date = key[0] if isinstance(key, tuple) else key
                part_path = staging / f"date={trade_date.isoformat()}" / "part-0.parquet"
                part_path.parent.mkdir(parents=True)
                partition.write_parquet(part_path, compression="zstd", statistics=True)
                table_records.append({
                    "path": str(part_path.relative_to(staging)),
                    "sha256": file_sha256(part_path),
                    "rows": partition.height,
                })
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "product": product,
                **identity,
                "identity": identity,
                "row_count": frame.height,
                "partitions": sorted(table_records, key=lambda item: item["path"]),
                "quality_summary": quality_summary,
                "created_at": created_at,
            }
            (staging / "_MANIFEST.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            root.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
        global_manifest = {
            "run_id": run_id,
            "identity": identity,
            "products": {
                product: str(path.relative_to(self._lake.root))
                for product, path in final_directories.items()
            },
            "quality_summary": quality_summary,
        }
        global_path = self._lake.root / "gold" / "l2" / f"run_id={run_id}" / "_MANIFEST.json"
        self._lake.write_immutable_json(global_path, global_manifest)
        _atomic_text(self._lake.root / "gold" / "l2" / "_CURRENT", run_id + "\n")
        return run_id


def load_current_l2_product(lake: DataLake, product: str) -> tuple[dict[str, object], pl.DataFrame]:
    """Resolve only the globally committed run; orphaned partial runs stay invisible."""
    current_path = lake.root / "gold" / "l2" / "_CURRENT"
    if not current_path.exists():
        raise FileNotFoundError("L2_CURRENT_POINTER_MISSING")
    run_id = current_path.read_text(encoding="utf-8").strip()
    global_manifest_path = lake.root / "gold" / "l2" / f"run_id={run_id}" / "_MANIFEST.json"
    global_manifest = json.loads(global_manifest_path.read_text(encoding="utf-8"))
    try:
        product_path = lake.root / global_manifest["products"][product]
    except KeyError as exc:
        raise KeyError(f"L2_CURRENT_PRODUCT_MISSING product={product}") from exc
    manifest = json.loads((product_path / "_MANIFEST.json").read_text(encoding="utf-8"))
    frames = []
    for partition in manifest["partitions"]:
        path = product_path / partition["path"]
        if file_sha256(path) != partition["sha256"]:
            raise RuntimeError(f"L2_PRODUCT_CHECKSUM_MISMATCH product={product}")
        frames.append(pl.read_parquet(path))
    return manifest, pl.concat(frames, how="vertical")
