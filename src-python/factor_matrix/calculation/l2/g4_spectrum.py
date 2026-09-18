"""Pre-registered G4.1 residual eigenspectrum diagnostic."""

from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, open_duckdb, source_tree_hash, utc_now
from .g3_runner import load_current_g3_manifest


DEFAULT_CONFIG = Path("config/g4_validation_protocol_v1.json")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _principal_angle_degrees(
    previous_assets: list[str], previous_loadings: np.ndarray,
    current_assets: list[str], current_loadings: np.ndarray,
    *, dimension: int | None = None,
) -> tuple[float | None, int]:
    common = sorted(set(previous_assets).intersection(current_assets))
    available = min(previous_loadings.shape[1], current_loadings.shape[1])
    dimension = available if dimension is None else min(dimension, available)
    if not common or dimension == 0 or len(common) <= dimension:
        return None, len(common)
    previous_index = {asset: idx for idx, asset in enumerate(previous_assets)}
    current_index = {asset: idx for idx, asset in enumerate(current_assets)}
    left = previous_loadings[[previous_index[asset] for asset in common], :dimension]
    right = current_loadings[[current_index[asset] for asset in common], :dimension]
    left_q, _ = np.linalg.qr(left)
    right_q, _ = np.linalg.qr(right)
    singular = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    cosine = float(np.clip(np.min(singular), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine))), len(common)


def run_g4_statistical_factor_spectrum(
    lake: DataLake, *, config_path: Path = DEFAULT_CONFIG,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["g4_1_statistical_factor_protocol"]
    if not protocol.get("preregistered"):
        raise RuntimeError("G4_STATISTICAL_FACTOR_PROTOCOL_NOT_PREREGISTERED")
    if protocol.get("per_board_estimation") != "forbidden":
        raise RuntimeError("G4_STATISTICAL_FACTORS_MUST_USE_ALL_A_SHARE_UNIVERSE")
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    specific_path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    identity = {
        "g3_run_id": g3["run_id"],
        "input_hashes": {
            "g3_manifest": file_sha256(g3_manifest_path),
            "specific_returns": file_sha256(specific_path),
            "protocol": file_sha256(config_path),
        },
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_spectrum_{g3['end'].replace('-', '')}_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_statistical_spectrum" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise RuntimeError(f"G4_PARTIAL_RUN_CONFLICT:{run_dir}")

    window = int(protocol["rolling_window_days"])
    stride = int(protocol["evaluation_stride_trading_days"])
    coverage = float(protocol["minimum_asset_observation_ratio"])
    maximum_factors = int(protocol["maximum_factor_count"])
    stability_threshold = float(
        protocol["stable_factor_count_rule"][
            "maximum_largest_principal_angle_degrees"
        ]
    )
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        # The rolling windows overlap heavily.  Read the residual long table
        # once and reuse the in-memory frame; querying/parsing the same parquet
        # file once per endpoint was the dominant avoidable cost in G4.1.
        raw_all = connection.execute(
            f"SELECT trade_date,asset_id,specific_return FROM read_parquet('{_sql_path(specific_path)}') "
            "WHERE specific_return IS NOT NULL ORDER BY trade_date, asset_id"
        ).pl()
        dates = raw_all.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
        endpoint_indices = list(range(window - 1, len(dates), stride))
        if endpoint_indices[-1] != len(dates) - 1:
            endpoint_indices.append(len(dates) - 1)
        spectrum_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        previous_assets: list[str] | None = None
        previous_loadings: np.ndarray | None = None
        previous_endpoint = None
        loading_snapshots: list[tuple[date, list[str], np.ndarray]] = []
        for endpoint_index in endpoint_indices:
            window_dates = dates[endpoint_index - window + 1:endpoint_index + 1]
            start, end = window_dates[0], window_dates[-1]
            raw = raw_all.filter(pl.col("trade_date").is_between(start, end))
            wide = raw.pivot(on="asset_id", index="trade_date", values="specific_return").sort("trade_date")
            asset_columns = [column for column in wide.columns if column != "trade_date"]
            minimum_observations = math.ceil(coverage * window)
            assets = [
                column for column in asset_columns
                if wide[column].len() - wide[column].null_count() >= minimum_observations
            ]
            values = wide.select(assets).to_numpy().astype(np.float64, copy=False)
            observed = ~np.isnan(values)
            counts = observed.sum(axis=0)
            means = np.nansum(values, axis=0) / counts
            centered = np.where(observed, values - means, 0.0)
            rms = np.sqrt(np.sum(centered * centered, axis=0) / window)
            valid = np.isfinite(rms) & (rms > 0)
            centered = centered[:, valid]
            assets = [asset for asset, keep in zip(assets, valid, strict=True) if keep]
            standardized = centered / rms[valid]
            gram = standardized @ standardized.T / window
            eigenvalues, temporal_vectors = np.linalg.eigh(gram)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            temporal_vectors = temporal_vectors[:, order]
            n_effective = len(assets)
            q_ratio = n_effective / window
            mp_upper = (1.0 + math.sqrt(q_ratio)) ** 2
            raw_factor_count = int(np.sum(eigenvalues > mp_upper))
            selected_count = min(raw_factor_count, maximum_factors)
            total_variance = float(np.sum(eigenvalues))
            explained = (
                float(np.sum(eigenvalues[:selected_count]) / total_variance)
                if selected_count and total_variance > 0 else 0.0
            )
            if selected_count:
                denominator = np.sqrt(window * eigenvalues[:selected_count])
                loadings = standardized.T @ temporal_vectors[:, :selected_count] / denominator
            else:
                loadings = np.empty((n_effective, 0), dtype=np.float64)
            angle = None
            common_assets = 0
            if previous_assets is not None and previous_loadings is not None:
                angle, common_assets = _principal_angle_degrees(
                    previous_assets, previous_loadings, assets, loadings,
                )
                for dimension in range(1, maximum_factors + 1):
                    dimension_angle, dimension_common = _principal_angle_degrees(
                        previous_assets, previous_loadings, assets, loadings,
                        dimension=dimension,
                    )
                    stability_rows.append({
                        "previous_window_end": previous_endpoint,
                        "window_end": end,
                        "subspace_dimension": min(
                            dimension, previous_loadings.shape[1], loadings.shape[1]
                        ),
                        "largest_principal_angle_degrees": dimension_angle,
                        "common_assets": dimension_common,
                    })
            summary_rows.append({
                "window_start": start, "window_end": end,
                "n_trading_days": window, "n_effective_assets": n_effective,
                "q_ratio": q_ratio, "mp_upper_edge": mp_upper,
                "raw_factor_count": raw_factor_count,
                "selected_factor_count": selected_count,
                "selected_variance_share": explained,
                "largest_principal_angle_degrees": angle,
                "stability_common_assets": common_assets,
                "previous_window_end": previous_endpoint,
            })
            loading_snapshots.append((end, list(assets), loadings.copy()))
            for rank, value in enumerate(eigenvalues[:maximum_factors + 10], start=1):
                spectrum_rows.append({
                    "window_start": start, "window_end": end, "eigenvalue_rank": rank,
                    "eigenvalue": float(value), "mp_upper_edge": mp_upper,
                    "above_mp_upper_edge": bool(value > mp_upper),
                })
            previous_assets, previous_loadings, previous_endpoint = assets, loadings, end
    finally:
        connection.close()

    summary = pl.DataFrame(summary_rows)
    spectrum = pl.DataFrame(spectrum_rows)
    stability = pl.DataFrame(stability_rows)
    dimension_summary = stability.group_by("subspace_dimension").agg(
        pl.len().alias("adjacent_window_pairs"),
        pl.col("largest_principal_angle_degrees").median().alias(
            "median_largest_principal_angle_degrees"
        ),
    ).sort("subspace_dimension")
    outputs = {}
    for name, frame in {
        "statistical_factor_spectrum_summary_v1": summary,
        "statistical_factor_eigenvalues_v1": spectrum,
        "statistical_factor_subspace_stability_v1": stability,
        "statistical_factor_stable_dimension_summary_v1": dimension_summary,
    }.items():
        path = run_dir / f"{name}.parquet"
        _write_parquet(path, frame)
        outputs[name] = lake.artifact_record(path)
    counts = summary["selected_factor_count"]
    angles = summary["largest_principal_angle_degrees"].drop_nulls()
    reported_dimensions = tuple(
        dimension for dimension in (1, 5, 10, 20) if dimension <= maximum_factors
    )
    stability_medians = {
        f"median_largest_principal_angle_{dimension}d_degrees": float(
            stability.filter(pl.col("subspace_dimension") == dimension)[
                "largest_principal_angle_degrees"
            ].median()
        )
        for dimension in reported_dimensions
    }
    stable_dimensions = dimension_summary.filter(
        pl.col("median_largest_principal_angle_degrees") <= stability_threshold
    )["subspace_dimension"]
    j_stable = int(stable_dimensions.max()) if len(stable_dimensions) else 0
    decision_end = date.fromisoformat(protocol["decision_sample_end_inclusive"])
    loading_frames: list[pl.DataFrame] = []
    for endpoint, assets, loadings in loading_snapshots:
        values: dict[str, Any] = {
            "window_end": [endpoint] * len(assets),
            "asset_id": assets,
            "within_decision_sample": [endpoint <= decision_end] * len(assets),
        }
        for dimension in range(j_stable):
            values[f"statistical_loading_{dimension + 1}"] = loadings[:, dimension]
        loading_frames.append(pl.DataFrame(values))
    stable_loadings = pl.concat(loading_frames) if loading_frames else pl.DataFrame()
    loadings_path = run_dir / "statistical_factor_stable_loadings_v1.parquet"
    _write_parquet(loadings_path, stable_loadings)
    outputs["statistical_factor_stable_loadings_v1"] = lake.artifact_record(loadings_path)
    all_windows_hit_cap = bool((summary["selected_factor_count"] == maximum_factors).all())
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "job": "g4_1_statistical_factor_spectrum", "status": "passed",
        "created_at": utc_now().isoformat(), "g3_run_id": g3["run_id"],
        "identity": identity, "protocol": protocol,
        "headline": {
            "evaluation_windows": summary.height,
            "median_selected_factor_count": float(counts.median()),
            "minimum_selected_factor_count": int(counts.min()),
            "maximum_selected_factor_count": int(counts.max()),
            "latest_selected_factor_count": int(counts[-1]),
            "J_MP": f">={maximum_factors}" if all_windows_hit_cap else int(counts[-1]),
            "J_MP_censored": all_windows_hit_cap,
            "raw_mp_factor_count_minimum_diagnostic": int(summary["raw_factor_count"].min()),
            "raw_mp_factor_count_median_diagnostic": float(summary["raw_factor_count"].median()),
            "raw_mp_factor_count_maximum_diagnostic": int(summary["raw_factor_count"].max()),
            "J_stable": j_stable,
            "J_stable_angle_threshold_degrees": stability_threshold,
            "decision_factor_count": min(maximum_factors, j_stable),
            "stable_loading_basis": "eigenvalue_ordered_sign_indeterminate",
            "stable_loading_decision_sample_end": decision_end.isoformat(),
            "median_selected_variance_share": float(summary["selected_variance_share"].median()),
            "median_largest_principal_angle_degrees": float(angles.median()) if len(angles) else None,
            "statistical_factor_iteration_required": bool(counts[-1] > 0),
            **stability_medians,
        },
        "outputs": outputs,
    }
    return lake.write_immutable_json(manifest_path, manifest)
