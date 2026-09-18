"""Formal read-only I/O runner for G4.3 residual permutation v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now
from .g3_runner import load_current_g3_manifest
from .g4_3_residual_permutation import _load_l1_exposure_frame
from .g4_3_residual_permutation_v2 import (
    negative_control_gate_v2,
    permutation_test_v2,
)


IMPLEMENTATION_ID = "g4_3_stratified_residual_permutation_v2"


@dataclass(frozen=True)
class G43V2Config:
    input_run_id: str
    strata_definition: str
    groupings: tuple[str, ...]
    negative_controls: tuple[str, ...] = ("industry", "board")
    n_permutations: int = 1000
    seed: int = 20260817
    winsor_mad_k: float = 5.0
    min_group_size: int = 20
    min_obs_per_day: int = 5
    min_effective_days: int = 20
    fdr_q: float = 0.05
    control_band: tuple[float, float] = (0.80, 1.25)
    status: str = "draft"
    date_start: date | None = None
    date_end: date | None = None
    max_assets: int | None = None
    asset_sampling: str = "balanced_by_board"
    industry_label_change_policy: str = "reject"
    max_changed_asset_fraction: float = 0.01

    @classmethod
    def from_json(cls, path: Path, *, input_run_id: str | None = None) -> "G43V2Config":
        payload = json.loads(path.read_text(encoding="utf-8"))
        parse_date = lambda value: date.fromisoformat(value) if value else None
        return cls(
            input_run_id=input_run_id or str(payload["input_run_id"]),
            strata_definition=str(payload["permutation"]["strata_definition"]),
            groupings=tuple(payload.get("groupings", ("index_CSI300", "index_CSI500"))),
            negative_controls=tuple(payload["negative_control_gate"].get("controls", ("industry", "board"))),
            n_permutations=int(payload["permutation"].get("n_permutations", 1000)),
            seed=int(payload["permutation"].get("seed", 20260817)),
            winsor_mad_k=float(payload.get("winsor_mad_k", 5.0)),
            min_group_size=int(payload.get("min_group_size", 20)),
            min_obs_per_day=int(payload["statistic"].get("minimum_observations_per_day", 5)),
            min_effective_days=int(payload.get("min_effective_days", 20)),
            fdr_q=float(payload.get("fdr_q", 0.05)),
            control_band=tuple(float(x) for x in payload["negative_control_gate"].get("control_band", (0.80, 1.25))),
            status=str(payload.get("status", "draft")),
            date_start=parse_date(payload.get("date_start")),
            date_end=parse_date(payload.get("date_end")),
            max_assets=None if payload.get("max_assets") is None else int(payload["max_assets"]),
            asset_sampling=str(payload.get("asset_sampling", "balanced_by_board")),
            industry_label_change_policy=str(payload.get("industry_label_change_policy", "reject")),
            max_changed_asset_fraction=float(payload.get("max_changed_asset_fraction", 0.01)),
        )

    def validate(self) -> None:
        if self.status not in {"draft", "frozen"}:
            raise ValueError("G43_V2_PROTOCOL_STATUS_INVALID")
        if self.n_permutations < 1 or self.seed < 0:
            raise ValueError("G43_V2_PERMUTATION_CONFIG_INVALID")
        if self.winsor_mad_k <= 0 or self.min_group_size < 2:
            raise ValueError("G43_V2_GROUP_CONFIG_INVALID")
        if self.min_obs_per_day < 2 or self.min_effective_days < 1:
            raise ValueError("G43_V2_OBSERVATION_CONFIG_INVALID")
        if not 0 < self.fdr_q < 1:
            raise ValueError("G43_V2_FDR_CONFIG_INVALID")
        if len(self.control_band) != 2 or not self.control_band[0] < self.control_band[1]:
            raise ValueError("G43_V2_CONTROL_BAND_INVALID")
        if self.date_start and self.date_end and self.date_end < self.date_start:
            raise ValueError("G43_V2_DATE_RANGE_INVALID")
        if self.max_assets is not None and self.max_assets < 4:
            raise ValueError("G43_V2_ASSET_SAMPLE_INVALID")
        if self.asset_sampling not in {"balanced_by_board", "lexicographic"}:
            raise ValueError("G43_V2_ASSET_SAMPLING_INVALID")
        if self.industry_label_change_policy not in {"reject", "exclude_below_fraction"}:
            raise ValueError("G43_V2_LABEL_CHANGE_POLICY_INVALID")
        if not 0 <= self.max_changed_asset_fraction < 1:
            raise ValueError("G43_V2_LABEL_CHANGE_FRACTION_INVALID")
        allowed = {"industry", "board", "index_CSI300", "index_CSI500"}
        requested = set(self.groupings) | set(self.negative_controls)
        if not requested <= allowed:
            raise ValueError(f"G43_V2_GROUPING_NOT_SUPPORTED {sorted(requested - allowed)}")
        if not self.groupings:
            raise ValueError("G43_V2_GROUPINGS_EMPTY")


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _balanced_asset_selection(
    exposure: pl.DataFrame, asset_ids: list[str], max_assets: int | None,
    sampling: str,
) -> list[str]:
    ordered = sorted(set(asset_ids))
    if max_assets is None or len(ordered) <= max_assets:
        return ordered
    if sampling == "lexicographic":
        return ordered[:max_assets]
    latest = exposure.filter(pl.col("asset_id").cast(pl.Utf8).is_in(ordered)).sort(
        ["trade_date", "asset_id"], descending=[True, False]
    )
    latest_date = latest["trade_date"].min()
    latest = latest.filter(pl.col("trade_date") == latest_date)
    buckets = {
        str(board): sorted(
            value for value in frame["asset_id"].cast(pl.Utf8).to_list() if value in ordered
        )
        for board, frame in latest.group_by("board_id", maintain_order=True)
    }
    selected: list[str] = []
    position = 0
    keys = sorted(buckets)
    while len(selected) < max_assets and keys:
        progressed = False
        for key in keys:
            values = buckets[key]
            if position < len(values):
                selected.append(values[position])
                progressed = True
                if len(selected) >= max_assets:
                    break
        if not progressed:
            break
        position += 1
    if len(selected) < max_assets:
        selected.extend(value for value in ordered if value not in set(selected))
    return sorted(selected[:max_assets])


def _encode_categories(frame: pl.DataFrame, source: str, target: str) -> tuple[pl.DataFrame, int, list[str]]:
    values = sorted(str(value) for value in frame.get_column(source).drop_nulls().unique().to_list())
    mapping = pl.DataFrame({source: values, target: list(range(len(values)))})
    encoded = frame.with_columns(pl.col(source).cast(pl.Utf8)).join(mapping, on=source, how="left")
    return encoded, len(values), values


def _rank_daily_strata(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = frame.with_columns([
        pl.col("risk_size").rank("ordinal").over("trade_date").alias("size_rank"),
        pl.col("risk_size").count().over("trade_date").alias("size_count"),
        pl.col("risk_liquidity").rank("ordinal").over("trade_date").alias("liq_rank"),
        pl.col("risk_liquidity").count().over("trade_date").alias("liq_count"),
    ]).with_columns([
        (((pl.col("size_rank") - 1) * 10 / pl.col("size_count")).floor().clip(0, 9).cast(pl.Int64)).alias("size_decile"),
        (((pl.col("liq_rank") - 1) * 5 / pl.col("liq_count")).floor().clip(0, 4).cast(pl.Int64)).alias("liquidity_quintile"),
    ])
    return ranked.with_columns(
        (pl.col("size_decile") * 5 + pl.col("liquidity_quintile")).cast(pl.Int64).alias("stratum")
    )


def _label_change_summary(
    labels: dict[str, np.ndarray], ids: list[str], dates: list[str],
) -> dict[str, dict[str, Any]]:
    """Report within-window categorical label changes at asset granularity."""
    summary: dict[str, dict[str, Any]] = {}
    for name, array in labels.items():
        changed_assets: list[str] = []
        change_dates: set[str] = set()
        group_asset_counts: dict[str, int] = {}
        group_changed_counts: dict[str, int] = {}
        for asset_index, asset_id in enumerate(ids):
            values = array[:, asset_index]
            valid_positions = np.flatnonzero(values >= 0)
            if valid_positions.size == 0:
                continue
            first_group = str(int(values[valid_positions[0]]))
            group_asset_counts[first_group] = group_asset_counts.get(first_group, 0) + 1
            distinct = np.unique(values[valid_positions])
            if distinct.size <= 1:
                continue
            changed_assets.append(asset_id)
            group_changed_counts[first_group] = group_changed_counts.get(first_group, 0) + 1
            first_value = values[valid_positions[0]]
            change_dates.update(
                dates[position]
                for position in valid_positions
                if values[position] != first_value
            )
        summary[name] = {
            "changed_asset_count": len(changed_assets),
            "changed_assets": changed_assets,
            "change_dates": sorted(change_dates),
            "group_asset_counts": group_asset_counts,
            "group_changed_counts": group_changed_counts,
        }
    return summary


def _industry_label_change_decision(
    ids: list[str], label_changes: dict[str, dict[str, Any]],
    policy: str, max_fraction: float,
) -> dict[str, Any]:
    changed = list(label_changes.get("industry", {}).get("changed_assets", []))
    fraction = len(changed) / len(ids) if ids else 0.0
    industry = label_changes.get("industry", {})
    group_assets = industry.get("group_asset_counts", {})
    group_changed = industry.get("group_changed_counts", {})
    group_fractions = {
        group: group_changed.get(group, 0) / count
        for group, count in group_assets.items() if count
    }
    max_group_fraction = max(group_fractions.values(), default=0.0)
    if not changed:
        return {
            "action": "keep", "changed_asset_fraction": 0.0,
            "max_group_changed_asset_fraction": 0.0, "changed_assets": [],
        }
    if policy == "exclude_below_fraction" and max_group_fraction <= max_fraction:
        return {
            "action": "exclude",
            "changed_asset_fraction": fraction,
            "max_group_changed_asset_fraction": max_group_fraction,
            "group_changed_asset_fractions": group_fractions,
            "changed_assets": changed,
        }
    return {
        "action": "reject",
        "changed_asset_fraction": fraction,
        "max_group_changed_asset_fraction": max_group_fraction,
        "group_changed_asset_fractions": group_fractions,
        "changed_assets": changed,
    }


def _build_panel(
    residual: pl.DataFrame, exposure: pl.DataFrame, cfg: G43V2Config,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    mapping = residual.select(["trade_date", "exposure_date"]).unique()
    if mapping.group_by("trade_date").len().get_column("len").max() != 1:
        raise RuntimeError("G43_V2_MULTIPLE_EXPOSURE_DATES_PER_RETURN_DATE")
    if cfg.date_start:
        residual = residual.filter(pl.col("trade_date") >= cfg.date_start)
    if cfg.date_end:
        residual = residual.filter(pl.col("trade_date") <= cfg.date_end)
    if residual.is_empty():
        raise RuntimeError("G43_V2_RESIDUAL_PANEL_EMPTY_AFTER_DATE_FILTER")
    dates = sorted(str(value) for value in residual["trade_date"].unique().to_list())
    date_values = [date.fromisoformat(value) for value in dates]
    mapping = residual.select(["trade_date", "exposure_date"]).unique().sort("trade_date")
    exposure_dates = mapping["exposure_date"].to_list()
    residual = residual.with_columns(pl.col("asset_id").cast(pl.Utf8))
    all_ids = sorted(str(value) for value in residual["asset_id"].unique().to_list())
    exposure = exposure.with_columns(pl.col("asset_id").cast(pl.Utf8)).filter(
        pl.col("trade_date").is_in(exposure_dates)
        & pl.col("asset_id").is_in(all_ids)
    )
    ids = _balanced_asset_selection(exposure, all_ids, cfg.max_assets, cfg.asset_sampling)
    residual = residual.filter(pl.col("asset_id").is_in(ids))
    exposure = exposure.filter(pl.col("asset_id").is_in(ids))
    ids = sorted(str(value) for value in residual["asset_id"].unique().to_list())
    if len(ids) < cfg.min_group_size:
        raise RuntimeError("G43_V2_TOO_FEW_ASSETS_AFTER_SELECTION")
    date_index = {value: index for index, value in enumerate(dates)}
    asset_index = {value: index for index, value in enumerate(ids)}
    U = np.full((len(ids), len(dates)), np.nan, dtype=np.float64)
    residual = residual.with_columns([
        pl.col("trade_date").cast(pl.Utf8).replace(date_index).cast(pl.Int64).alias("date_index"),
        pl.col("asset_id").replace(asset_index).cast(pl.Int64).alias("asset_index"),
    ])
    U[
        residual["asset_index"].to_numpy(), residual["date_index"].to_numpy()
    ] = residual["specific_return"].to_numpy()

    exposure = exposure.join(
        mapping.select(["exposure_date", "trade_date"]).with_columns(
            pl.int_range(0, mapping.height).alias("date_index")
        ),
        left_on="trade_date", right_on="exposure_date", how="inner",
    ).with_columns(pl.col("asset_id").replace(asset_index).cast(pl.Int64).alias("asset_index"))
    exposure = _rank_daily_strata(exposure)
    T, N = len(dates), len(ids)
    strata = np.full((T, N), -1, dtype=np.int64)
    strata_valid = exposure.filter(pl.col("stratum").is_not_null())
    strata[strata_valid["date_index"].to_numpy(), strata_valid["asset_index"].to_numpy()] = strata_valid["stratum"].to_numpy()
    labels: dict[str, np.ndarray] = {}
    group_counts: dict[str, int] = {}
    label_sources = {
        "industry": "sw_l1_code", "board": "board_id",
        "index_CSI300": "risk_index_CSI300", "index_CSI500": "risk_index_CSI500",
    }
    for name in sorted(set(cfg.groupings) | set(cfg.negative_controls)):
        source = label_sources[name]
        if name.startswith("index_"):
            exposure = exposure.with_columns((pl.col(source) > 0).cast(pl.Int64).alias("_label"))
            encoded, count, categories = _encode_categories(exposure, "_label", "_label_code")
        else:
            encoded, count, categories = _encode_categories(exposure, source, "_label_code")
        array = np.full((T, N), -1, dtype=np.int64)
        valid = encoded.filter(pl.col("_label_code").is_not_null())
        array[valid["date_index"].to_numpy(), valid["asset_index"].to_numpy()] = valid["_label_code"].to_numpy()
        labels[name] = array
        group_counts[name] = count
        exposure = encoded.drop([column for column in ["_label", "_label_code"] if column in encoded.columns])
    if np.sum(strata >= 0) == 0:
        raise RuntimeError("G43_V2_NO_DAILY_STRATA")
    label_changes = _label_change_summary(labels, ids, dates)
    coverage_rows = []
    for index, value in enumerate(date_values):
        coverage_rows.append({
            "trade_date": value,
            "strata_assigned_assets": int(np.sum(strata[index] >= 0)),
            "strata_cell_count": int(len(np.unique(strata[index][strata[index] >= 0]))),
            **{
                f"{name}_assigned_assets": int(np.sum(labels[name][index] >= 0))
                for name in labels
            },
        })
    return U, np.isfinite(U), ids, dates, labels, strata, {
        "coverage": pl.DataFrame(coverage_rows),
        "group_counts": group_counts,
        "exposure_dates": [str(value) for value in exposure_dates],
        "sample_mode": cfg.asset_sampling,
        "max_assets": cfg.max_assets,
        "label_changes": label_changes,
    }


def _load_inputs(lake: DataLake, g3: dict[str, Any], g3_manifest_path: Path, cfg: G43V2Config) -> tuple[pl.DataFrame, pl.DataFrame, Path, Path]:
    residual_path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    residual = pl.read_parquet(residual_path, columns=[
        "trade_date", "exposure_date", "asset_id", "regression_mode",
        "specific_return", "in_estimation_domain",
    ]).filter(
        (pl.col("regression_mode") == "risk_only")
        & pl.col("in_estimation_domain")
        & pl.col("specific_return").is_not_null()
    )
    return residual, pl.DataFrame(), g3_manifest_path, residual_path


def _result_frame(results: dict[str, dict[str, Any]], *, control: bool) -> pl.DataFrame:
    def finite_or_none(value: Any) -> float | None:
        value = float(value)
        return value if np.isfinite(value) else None

    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        for index in range(len(result["q_observed"])):
            rows.append({
                "grouping": name,
                "group_index": index,
                "q_observed": finite_or_none(result["q_observed"][index]),
                "q_ratio": finite_or_none(result["q_ratio"][index]),
                "p_value": finite_or_none(result["p_value"][index]),
                "q_value": None if control else finite_or_none(result["q_value"][index]),
                "bh_passed": None if control or not np.isfinite(result["q_value"][index]) else bool(result["bh_passed"][index]),
                "null_median": finite_or_none(result["null_median"][index]),
                "null_q95": finite_or_none(result["null_q95"][index]),
                "effective_days": int(result["effective_days"][index]),
                "median_group_size": finite_or_none(result["median_group_size"][index]),
                "is_negative_control": control,
            })
    return pl.DataFrame(rows)


def _apply_fdr(results: dict[str, dict[str, Any]], q: float) -> dict[str, dict[str, Any]]:
    entries = [
        (name, index, float(value))
        for name, result in results.items()
        for index, value in enumerate(result["p_value"])
        if np.isfinite(value)
        and result["effective_days"][index] >= 1
        and result["median_group_size"][index] >= 1
    ]
    ordered = sorted(entries, key=lambda item: item[2])
    q_values = {(name, index): np.nan for name, result in results.items() for index in range(len(result["p_value"]))}
    running = 1.0
    for rank, (name, index, value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, value * len(ordered) / rank)
        q_values[(name, index)] = running
    for name, result in results.items():
        result["q_value"] = np.asarray([q_values[(name, index)] for index in range(len(result["p_value"]))])
        result["bh_passed"] = np.isfinite(result["q_value"]) & (result["q_value"] <= q)
    return results


def run_g4_3_v2(
    lake: DataLake, *, config_path: Path = Path("config/g4_3_residual_permutation_v2.json"),
) -> Path:
    cfg = G43V2Config.from_json(config_path)
    cfg.validate()
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    if cfg.input_run_id != g3["run_id"]:
        raise RuntimeError("G43_V2_CONFIG_CURRENT_G3_MISMATCH")
    exposure, l1_manifest_path, exposure_path = _load_l1_exposure_frame(lake, g3)
    residual, _, _, residual_path = _load_inputs(lake, g3, g3_manifest_path, cfg)
    U, mask, ids, dates, labels, strata, panel_meta = _build_panel(residual, exposure, cfg)
    U, _, _ = _prepare_panel(U, cfg.winsor_mad_k)
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "g3_run_id": g3["run_id"],
        "l1_run_id": g3["l1_run_id"],
        "config_sha256": file_sha256(config_path),
        "g3_manifest_sha256": file_sha256(g3_manifest_path),
        "specific_returns_sha256": file_sha256(residual_path),
        "l1_manifest_sha256": file_sha256(l1_manifest_path),
        "exposure_sha256": file_sha256(exposure_path),
        "code_sha": source_tree_hash(),
        "date_start": cfg.date_start.isoformat() if cfg.date_start else None,
        "date_end": cfg.date_end.isoformat() if cfg.date_end else None,
        "max_assets": cfg.max_assets,
    }
    run_id = f"g4_3_residual_permutation_v2_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_3_residual_permutation_v2" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    label_changes = panel_meta["label_changes"]
    label_change_decision = _industry_label_change_decision(
        ids, label_changes, cfg.industry_label_change_policy,
        cfg.max_changed_asset_fraction,
    )
    if label_change_decision["action"] == "exclude":
        excluded = set(label_change_decision["changed_assets"])
        keep = np.asarray([asset_id not in excluded for asset_id in ids], dtype=bool)
        ids = [asset_id for asset_id, keep_asset in zip(ids, keep) if keep_asset]
        U = U[keep]
        mask = mask[keep]
        strata = strata[:, keep]
        labels = {name: array[:, keep] for name, array in labels.items()}
        panel_meta["excluded_assets"] = sorted(excluded)
    elif label_change_decision["action"] == "reject":
        coverage_path = run_dir / "g4_3_daily_label_coverage_v2.parquet"
        _write_parquet(coverage_path, panel_meta["coverage"])
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "runner": IMPLEMENTATION_ID,
            "job": "g4_3_stratified_residual_permutation",
            "mode": "read_only",
            "status": "INVALID_RUNNER",
            "release_eligible": False,
            "created_at": utc_now().isoformat(),
            "identity": identity,
            "input_run_id": g3["run_id"],
            "input_hashes": {
                "g3_manifest": file_sha256(g3_manifest_path),
                "specific_returns": file_sha256(residual_path),
                "l1_manifest": file_sha256(l1_manifest_path),
                "risk_exposure_matrix": file_sha256(exposure_path),
                "config": file_sha256(config_path),
            },
            "frozen_config_refs": {
                "protocol_status": cfg.status,
                "strata_definition": cfg.strata_definition,
                "groupings": list(cfg.groupings),
                "negative_controls": list(cfg.negative_controls),
                "n_permutations": cfg.n_permutations,
                "seed": cfg.seed,
                "control_band": list(cfg.control_band),
            },
            "lineage": {
                "n_assets": len(ids), "n_dates": len(dates),
                "g3_risk_basis_id": g3.get("risk_basis_id"),
                "g3_risk_metric_id": g3.get("risk_metric_id"),
            },
            "consumes_attempt_budget": False,
            "reads_forward_returns": False,
            "current_pointer_mutated": False,
            "null_hypothesis": "Daily labels are exchangeable within each date x size-liquidity stratum; residual values and missingness remain fixed.",
            "invalid_reasons": ["G43_V2_INDUSTRY_LABEL_CHANGED_WITHIN_WINDOW"],
            "label_changes": label_changes,
            "label_change_decision": label_change_decision,
            "headline": {
                "effective_group_counts": {},
                "q_ratio_medians": {},
            },
            "sampling": {"mode": panel_meta["sample_mode"], "max_assets": panel_meta["max_assets"]},
            "outputs": {
                "g4_3_daily_label_coverage_v2": lake.artifact_record(coverage_path),
            },
        }
        return lake.write_immutable_json(manifest_path, manifest)
    result_by_name: dict[str, dict[str, Any]] = {}
    for name in cfg.negative_controls:
        result = permutation_test_v2(
            labels[name], strata, U, mask, n_groups=panel_meta["group_counts"][name],
            n_permutations=cfg.n_permutations, seed=cfg.seed, min_obs_per_day=cfg.min_obs_per_day,
        )
        result_by_name[name] = _mask_underpopulated(result, cfg)
    gate = negative_control_gate_v2(result_by_name, cfg.control_band)
    outputs = {
        "g4_3_negative_controls_v2": run_dir / "g4_3_negative_controls_v2.parquet",
        "g4_3_daily_label_coverage_v2": run_dir / "g4_3_daily_label_coverage_v2.parquet",
    }
    _write_parquet(outputs["g4_3_negative_controls_v2"], _result_frame(result_by_name, control=True))
    _write_parquet(outputs["g4_3_daily_label_coverage_v2"], panel_meta["coverage"])
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "runner": IMPLEMENTATION_ID,
        "job": "g4_3_stratified_residual_permutation",
        "mode": "read_only",
        "status": ("passed" if cfg.status == "frozen" else "passed_draft") if gate["passed"] else "INVALID_RUNNER",
        "release_eligible": bool(cfg.status == "frozen" and gate["passed"]),
        "created_at": utc_now().isoformat(),
        "identity": identity,
        "input_run_id": g3["run_id"],
        "input_hashes": {
            "g3_manifest": file_sha256(g3_manifest_path),
            "specific_returns": file_sha256(residual_path),
            "l1_manifest": file_sha256(l1_manifest_path),
            "risk_exposure_matrix": file_sha256(exposure_path),
            "config": file_sha256(config_path),
        },
        "frozen_config_refs": {
            "protocol_status": cfg.status,
            "strata_definition": cfg.strata_definition,
            "groupings": list(cfg.groupings),
            "negative_controls": list(cfg.negative_controls),
            "n_permutations": cfg.n_permutations,
            "seed": cfg.seed,
            "control_band": list(cfg.control_band),
            "industry_label_change_policy": cfg.industry_label_change_policy,
            "max_changed_asset_fraction": cfg.max_changed_asset_fraction,
        },
        "lineage": {
            "n_assets": len(ids), "n_dates": len(dates),
            "g3_risk_basis_id": g3.get("risk_basis_id"),
            "g3_risk_metric_id": g3.get("risk_metric_id"),
        },
        "consumes_attempt_budget": False,
        "reads_forward_returns": False,
        "current_pointer_mutated": False,
        "null_hypothesis": "Daily labels are exchangeable within each date x size-liquidity stratum; residual values and missingness remain fixed.",
        "negative_control": gate,
        "label_changes": label_changes,
        "label_change_decision": label_change_decision,
        "excluded_assets": panel_meta.get("excluded_assets", []),
        "headline": {
            "effective_group_counts": {
                name: int(np.isfinite(result["q_ratio"]).sum())
                for name, result in result_by_name.items()
            },
            "q_ratio_medians": {
                name: (
                    float(np.nanmedian(result["q_ratio"]))
                    if np.isfinite(result["q_ratio"]).any() else None
                )
                for name, result in result_by_name.items()
            },
        },
        "sampling": {"mode": panel_meta["sample_mode"], "max_assets": panel_meta["max_assets"]},
        "invalid_reasons": [] if gate["passed"] else ["G43_V2_NEGATIVE_CONTROL_GATE_FAILED"],
        "outputs": {},
    }
    if gate["passed"]:
        candidates = {}
        for name in cfg.groupings:
            result = permutation_test_v2(
                labels[name], strata, U, mask, n_groups=panel_meta["group_counts"][name],
                n_permutations=cfg.n_permutations, seed=cfg.seed, min_obs_per_day=cfg.min_obs_per_day,
            )
            candidates[name] = _mask_underpopulated(result, cfg)
        candidates = _apply_fdr(candidates, cfg.fdr_q)
        candidate_path = run_dir / "g4_3_group_results_v2.parquet"
        _write_parquet(candidate_path, _result_frame(candidates, control=False))
        outputs["g4_3_group_results_v2"] = candidate_path
    manifest["outputs"] = {name: lake.artifact_record(path) for name, path in outputs.items()}
    return lake.write_immutable_json(manifest_path, manifest)


def _prepare_panel(U: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(U, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(U - med), axis=1, keepdims=True)
    delta = 1.4826 * np.maximum(mad, 1e-12)
    standardized = np.clip((U - med) / delta, -k, k)
    mask = ~np.isnan(standardized)
    return np.nan_to_num(standardized), mask, np.nanvar(standardized, axis=1)


def _mask_underpopulated(result: dict[str, Any], cfg: G43V2Config) -> dict[str, Any]:
    valid = (
        result["effective_days"] >= cfg.min_effective_days
    ) & (result["median_group_size"] >= cfg.min_group_size)
    for key in ("q_observed", "p_value", "median_group_size", "null_median", "null_q95", "q_ratio"):
        result[key] = np.where(valid, result[key], np.nan)
    result["effective_days"] = np.where(valid, result["effective_days"], 0)
    return result
