#!/usr/bin/env python3
"""Run label-free design/weight diagnostics for the structural G3 candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src-python"))

from factor_matrix.calculation.l2.design_matrix import (  # noqa: E402
    build_daily_categorical_constrained_design,
)
from factor_matrix.calculation.l2.g3_runner import _active_factor_columns  # noqa: E402
from factor_matrix.calculation.l2.linear_algebra import null_space_basis  # noqa: E402
from factor_matrix.storage import file_sha256, json_hash  # noqa: E402


DATA = PROJECT / "data"
CONFIG_PATH = PROJECT / "config/g3_conditioning_diagnostics_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _block(factor_id: str) -> str:
    if factor_id == "risk_country":
        return "country"
    if factor_id.startswith("risk_industry_"):
        return "industry"
    if factor_id.startswith("risk_board_"):
        return "board"
    if factor_id.startswith("risk_index_"):
        return "index_membership"
    return "style"


def _weight_stats(values: np.ndarray) -> dict[str, float]:
    ordered = np.sort(values)
    total = float(values.sum())
    top_count = max(1, math.ceil(values.size * 0.01))
    return {
        "effective_sample_size": total * total / float(np.square(values).sum()),
        "max_to_median_ratio": float(values.max() / np.median(values)),
        "top_1pct_weight_share": float(ordered[-top_count:].sum() / total),
    }


def _effective_members(values: np.ndarray) -> float:
    total = float(values.sum())
    return total * total / float(np.square(values).sum())


def _matrix_diagnostic(design: Any) -> dict[str, Any]:
    inputs = design.regression_input
    estimation = np.asarray(inputs.is_tradable, dtype=bool)
    x = np.asarray(inputs.exposures, dtype=float)[estimation]
    weights = np.asarray(inputs.base_weights, dtype=float)[estimation]
    constraints = [list(row) for row in inputs.equality_constraints]
    basis = np.asarray(
        null_space_basis(constraints, len(inputs.factor_ids)), dtype=float
    )
    reduced_x = x @ basis
    gram = reduced_x.T @ (weights[:, None] * reduced_x)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    direction = basis @ eigenvectors[:, 0]
    direction /= np.linalg.norm(direction)
    largest = int(np.argmax(np.abs(direction)))
    if direction[largest] < 0:
        direction = -direction
    return {
        "condition_number": float(np.linalg.cond(gram)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "direction": direction,
        "weights": weights,
        "x": x,
        "factor_ids": tuple(inputs.factor_ids),
    }


def _l1_partition_index(history: dict[str, Any]) -> list[tuple[date, date, Path]]:
    rows: list[tuple[date, date, Path]] = []
    base = DATA / "gold/risk_exposure_matrix"
    for run_id in history["partition_run_ids"]:
        manifest = _load(base / f"run_id={run_id}/_MANIFEST.json")
        path = DATA / manifest["outputs"]["risk_exposure_matrix_v1"]["path"]
        rows.append((date.fromisoformat(manifest["start"]), date.fromisoformat(manifest["end"]), path))
    return rows


def _read_exposures(
    exposure_dates: list[date], partition_index: list[tuple[date, date, Path]],
    risk_columns: list[str],
) -> pl.DataFrame:
    wanted = set(exposure_dates)
    frames: list[pl.DataFrame] = []
    metadata = ["trade_date", "asset_id", "board_id", "is_valid"]
    for start, end, path in partition_index:
        dates = [value for value in wanted if start <= value <= end]
        if not dates:
            continue
        schema = pl.read_parquet_schema(path)
        available = metadata + [column for column in risk_columns if column in schema]
        frame = (
            pl.scan_parquet(path).filter(pl.col("trade_date").is_in(dates))
            .select(available).collect()
        )
        missing = [column for column in risk_columns if column not in frame.columns]
        if missing:
            frame = frame.with_columns(*(pl.lit(0.0).alias(column) for column in missing))
        frames.append(frame.select(metadata + risk_columns))
    if not frames:
        raise RuntimeError("CONDITION_DIAGNOSTIC_L1_DATES_NOT_FOUND")
    return pl.concat(frames, how="vertical")


def main() -> None:
    protocol = _load(CONFIG_PATH)
    current = _load(DATA / "gold/risk_exposure_matrix/_CURRENT.json")
    history_path = DATA / current["manifest"]
    history = _load(history_path)
    if history.get("risk_basis_id") != protocol["risk_basis_id"]:
        raise RuntimeError("CONDITION_DIAGNOSTIC_RISK_BASIS_MISMATCH")
    if history.get("risk_metric_id") != protocol["risk_metric_id"]:
        raise RuntimeError("CONDITION_DIAGNOSTIC_RISK_METRIC_MISMATCH")

    exposure_history_path = DATA / history["outputs"]["risk_exposure_matrix_v1"]["path"]
    history_schema = pl.read_parquet_schema(exposure_history_path)
    risk_columns = [
        column for column, dtype in history_schema.items()
        if column.startswith("risk_") and dtype.is_float()
    ]
    partition_index = _l1_partition_index(history)
    weight_path = DATA / history["weight_metric_contract"]["artifact_path"]
    valuation_path = DATA / "silver/valuation_daily/data.parquet"
    selected_dates = set(protocol["preselected_gate_failure_trade_dates"])

    g3_manifests: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in (DATA / "gold/l2b_risk_only").glob("run_id=*/_MANIFEST.json"):
        manifest = _load(manifest_path)
        if manifest.get("risk_basis_id") == history["risk_basis_id"]:
            g3_manifests.append((manifest_path, manifest))
    g3_manifests.sort(key=lambda item: item[1]["start"])
    if len(g3_manifests) != 84:
        raise RuntimeError(
            f"CONDITION_DIAGNOSTIC_EXPECTS_84_G3_PARTITIONS observed={len(g3_manifests)}"
        )

    daily_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    minimum_members = int(history["derived_params"]["minimum_category_members"])

    for partition_number, (manifest_path, g3_manifest) in enumerate(g3_manifests, start=1):
        quality_path = DATA / g3_manifest["outputs"]["factor_regression_quality_v1"]["path"]
        specific_path = DATA / g3_manifest["outputs"]["specific_returns_v1"]["path"]
        quality = pl.read_parquet(quality_path).select(
            "trade_date", "exposure_date", "status", "condition_number"
        )
        exposure_dates = quality["exposure_date"].to_list()
        domain = pl.read_parquet(specific_path).select(
            "trade_date", "exposure_date", "asset_id", "in_estimation_domain"
        )
        exposures = _read_exposures(exposure_dates, partition_index, risk_columns).rename(
            {"trade_date": "exposure_date"}
        )
        weights = (
            pl.scan_parquet(weight_path)
            .filter(pl.col("exposure_date").is_in(exposure_dates))
            .select("exposure_date", "asset_id", "candidate_weight").collect()
        )
        valuation = (
            pl.scan_parquet(valuation_path)
            .filter(pl.col("trade_date").is_in(exposure_dates))
            .select("trade_date", "asset_id", "float_mkt_cap").collect()
            .rename({"trade_date": "exposure_date"})
        )
        joined = (
            domain.join(exposures, on=("exposure_date", "asset_id"), how="inner", validate="m:1")
            .join(weights, on=("exposure_date", "asset_id"), how="left", validate="m:1")
            .join(valuation, on=("exposure_date", "asset_id"), how="left", validate="m:1")
            .with_columns(
                pl.lit(True).alias("model_eligible"),
                pl.lit(0.0).alias("realized_return"),
            )
        )
        if joined.select(
            pl.any_horizontal(
                pl.col("candidate_weight").is_null(), pl.col("float_mkt_cap").is_null()
            ).any()
        ).item():
            raise RuntimeError("CONDITION_DIAGNOSTIC_WEIGHT_OR_CAP_COVERAGE_INCOMPLETE")

        quality_by_exposure = {
            row["exposure_date"]: row for row in quality.iter_rows(named=True)
        }
        for key, daily in joined.partition_by("exposure_date", as_dict=True).items():
            exposure_date = key[0] if isinstance(key, tuple) else key
            observed = quality_by_exposure[exposure_date]
            trade_date = observed["trade_date"]
            factor_columns, blocks = _active_factor_columns(daily, risk_columns)
            common = {
                "exposure_columns": factor_columns,
                "family_by_column": {column: "risk" for column in factor_columns},
                "categorical_blocks": blocks,
                "minimum_category_members": minimum_members,
                "estimation_column": "in_estimation_domain",
            }
            structural_design = build_daily_categorical_constrained_design(
                daily, base_weight_column="candidate_weight", **common
            )
            sqrt_design = build_daily_categorical_constrained_design(daily, **common)
            diagnostics = {
                "structural_pit_v1": _matrix_diagnostic(structural_design),
                "sqrt_cap_counterfactual": _matrix_diagnostic(sqrt_design),
            }
            row: dict[str, Any] = {
                "trade_date": trade_date,
                "exposure_date": exposure_date,
                "factor_count": len(factor_columns),
                "observed_effective_condition_number": float(observed["condition_number"]),
                "observed_effective_status": observed["status"],
            }
            for scheme_id, diagnostic in diagnostics.items():
                stats = _weight_stats(diagnostic["weights"])
                prefix = "structural" if scheme_id == "structural_pit_v1" else "sqrt_cap"
                row[f"{prefix}_base_condition_number"] = diagnostic["condition_number"]
                row[f"{prefix}_minimum_eigenvalue"] = diagnostic["minimum_eigenvalue"]
                row[f"{prefix}_effective_sample_size"] = stats["effective_sample_size"]
                row[f"{prefix}_max_to_median_ratio"] = stats["max_to_median_ratio"]
                row[f"{prefix}_top_1pct_weight_share"] = stats["top_1pct_weight_share"]

                if trade_date.isoformat() in selected_dates:
                    direction = diagnostic["direction"]
                    factor_ids = diagnostic["factor_ids"]
                    order = np.argsort(np.abs(direction))[::-1]
                    for rank, position in enumerate(order[:12], start=1):
                        loading_rows.append({
                            "trade_date": trade_date, "exposure_date": exposure_date,
                            "weight_scheme_id": scheme_id, "rank": rank,
                            "factor_id": factor_ids[position],
                            "factor_block": _block(factor_ids[position]),
                            "direction_loading": float(direction[position]),
                            "squared_loading_share": float(direction[position] ** 2),
                        })
                    for block_id in ("country", "industry", "board", "index_membership", "style"):
                        contribution = sum(
                            float(value * value) for factor_id, value in zip(factor_ids, direction)
                            if _block(factor_id) == block_id
                        )
                        block_rows.append({
                            "trade_date": trade_date, "exposure_date": exposure_date,
                            "weight_scheme_id": scheme_id, "factor_block": block_id,
                            "squared_loading_share": contribution,
                        })
                    x = diagnostic["x"]
                    values = diagnostic["weights"]
                    for position, factor_id in enumerate(factor_ids):
                        if _block(factor_id) not in {"industry", "board", "index_membership"}:
                            continue
                        support = np.abs(x[:, position]) > 0.5
                        if not np.any(support):
                            continue
                        category_rows.append({
                            "trade_date": trade_date, "exposure_date": exposure_date,
                            "weight_scheme_id": scheme_id, "factor_id": factor_id,
                            "factor_block": _block(factor_id),
                            "member_count": int(support.sum()),
                            "weighted_effective_member_count": _effective_members(values[support]),
                            "weight_share": float(values[support].sum() / values.sum()),
                        })
            daily_rows.append(row)
        input_records.append({
            "run_id": g3_manifest["run_id"],
            "quality_sha256": g3_manifest["outputs"]["factor_regression_quality_v1"]["sha256"],
            "specific_domain_sha256": g3_manifest["outputs"]["specific_returns_v1"]["sha256"],
        })
        print(
            f"completed partition {partition_number}/{len(g3_manifests)} "
            f"run_id={g3_manifest['run_id']}",
            flush=True,
        )

    daily = pl.DataFrame(daily_rows).sort("trade_date")
    loadings = pl.DataFrame(loading_rows).sort("trade_date", "weight_scheme_id", "rank")
    blocks = pl.DataFrame(block_rows).sort("trade_date", "weight_scheme_id", "factor_block")
    categories = pl.DataFrame(category_rows).sort(
        "trade_date", "weight_scheme_id", "factor_block", "factor_id"
    )
    quantiles = []
    for scheme_id, column in (
        ("structural_pit_v1", "structural_base_condition_number"),
        ("sqrt_cap_counterfactual", "sqrt_cap_base_condition_number"),
        ("observed_structural_effective_huber", "observed_effective_condition_number"),
    ):
        values = daily[column].to_numpy()
        quantiles.append({
            "series_id": scheme_id,
            "p50": float(np.quantile(values, 0.5)),
            "p99": float(np.quantile(values, 0.99)),
            "p999": float(np.quantile(values, 0.999)),
            "maximum": float(values.max()),
        })
    tail = pl.DataFrame(quantiles)

    identity = {
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "l1_history_run_id": history["run_id"],
        "risk_basis_id": history["risk_basis_id"],
        "risk_metric_id": history["risk_metric_id"],
        "weight_artifact_sha256": file_sha256(weight_path),
        "g3_partition_inputs": input_records,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    run_id = f"g3_conditioning_diagnostic_{json_hash(identity)[:16]}"
    run_dir = DATA / "diagnostics/g3_conditioning" / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    products = {
        "daily_condition_and_weight_concentration_v1": daily,
        "minimum_eigen_direction_factor_loadings_v1": loadings,
        "minimum_eigen_direction_block_contributions_v1": blocks,
        "invalid_date_category_effective_members_v1": categories,
        "condition_number_tail_comparison_v1": tail,
    }
    outputs: dict[str, Any] = {}
    for name, frame in products.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        outputs[name] = {
            "path": str(path.relative_to(DATA)),
            "sha256": file_sha256(path),
            "rows": frame.height,
        }

    frozen_protocol = _load(PROJECT / "config/research_protocol_v1.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "g3_conditioning_diagnostics",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_class": protocol["diagnostic_class"],
        "return_value_columns_read": [],
        "domain_columns_read_from_g3": [
            "trade_date", "exposure_date", "asset_id", "in_estimation_domain"
        ],
        "previously_materialized_quality_fields_read": [
            "status", "condition_number"
        ],
        "identity": identity,
        "counts": {
            "dates": daily.height,
            "preselected_gate_failure_dates": len(selected_dates),
            "g3_partitions": len(g3_manifests),
        },
        "frozen_invalid_date_authority": {
            "research_protocol_target": frozen_protocol["targets"][
                "maximum_post_burn_invalid_date_ratio"
            ],
            "g3_history_implementation": "G3_HISTORY_REQUIRES_ZERO_INVALID_DATES",
            "five_percent_rule_found_in_executable_config": False,
        },
        "outputs": outputs,
    }
    manifest_path = run_dir / "_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
