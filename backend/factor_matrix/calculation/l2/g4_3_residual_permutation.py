"""G4.3 stratified residual-group permutation diagnostics.

The runner permutes group labels within frozen strata and never permutes the
residual panel.  It is read-only, does not consume the research-attempt ledger,
and cannot mutate either L1 or G3 ``_CURRENT`` pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now
from .g3_runner import load_current_g3_manifest


IMPLEMENTATION_ID = "g4_3_stratified_residual_permutation_v1"
_MAD_NORMALIZER = 1.4826
_MIN_LABEL_STABILITY = 0.9


@dataclass(frozen=True)
class G43Config:
    input_run_id: str
    strata_definition: str
    groupings: tuple[str, ...]
    negative_controls: tuple[str, ...] = ("industry", "board")
    n_permutations: int = 1000
    seed: int = 20260817
    winsor_mad_k: float = 5.0
    min_group_size: int = 20
    min_obs_per_day: int = 5
    fdr_q: float = 0.05
    control_band: tuple[float, float] = (0.80, 1.25)
    on_significant: str = (
        "record known_residual_structure in G6; do not add X automatically; "
        "an added column requires the preregistered G4.2 membership protocol"
    )
    interpretation_note: str = (
        "common movement is not predictability; significance indicates possible "
        "risk-model incompleteness and is not an alpha lead"
    )

    @classmethod
    def from_json(cls, path: Path, *, input_run_id: str | None = None) -> "G43Config":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            input_run_id=input_run_id or str(payload["input_run_id"]),
            strata_definition=str(payload["strata_definition"]),
            groupings=tuple(payload.get("groupings", ())),
            negative_controls=tuple(payload.get("negative_controls", ("industry", "board"))),
            n_permutations=int(payload.get("n_permutations", 1000)),
            seed=int(payload.get("seed", 20260817)),
            winsor_mad_k=float(payload.get("winsor_mad_k", 5.0)),
            min_group_size=int(payload.get("min_group_size", 20)),
            min_obs_per_day=int(payload.get("min_obs_per_day", 5)),
            fdr_q=float(payload.get("fdr_q", 0.05)),
            control_band=tuple(float(x) for x in payload.get("control_band", (0.80, 1.25))),
            on_significant=str(payload.get("on_significant", cls.on_significant)),
            interpretation_note=str(payload.get("interpretation_note", cls.interpretation_note)),
        )

    def validate(self) -> None:
        if self.n_permutations < 1 or self.seed < 0:
            raise ValueError("G43_PERMUTATION_CONFIG_INVALID")
        if self.winsor_mad_k <= 0 or self.min_group_size < 2 or self.min_obs_per_day < 2:
            raise ValueError("G43_ROBUST_STATISTIC_CONFIG_INVALID")
        if not 0 < self.fdr_q < 1:
            raise ValueError("G43_FDR_CONFIG_INVALID")
        if len(self.control_band) != 2 or not self.control_band[0] < self.control_band[1]:
            raise ValueError("G43_CONTROL_BAND_INVALID")
        allowed = {"industry", "board", "index_CSI300", "index_CSI500"}
        requested = set(self.groupings) | set(self.negative_controls)
        if not requested <= allowed:
            raise ValueError(f"G43_GROUPING_NOT_PIT_SUPPORTED {sorted(requested - allowed)}")
        if not self.groupings:
            raise ValueError("G43_GROUPINGS_EMPTY")


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _load_l1_exposure_frame(lake: DataLake, g3: dict[str, Any]) -> tuple[pl.DataFrame, Path, Path]:
    current_path = lake.root / "gold" / "risk_exposure_matrix" / "_CURRENT.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("run_id") != g3.get("l1_run_id"):
        raise RuntimeError("G43_G3_L1_LINEAGE_MISMATCH")
    manifest_path = lake.root / current["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal_candidate = (
        manifest.get("publish_mode") == "formal_candidate_not_current"
        and manifest.get("counts", {}).get("post_burn_invalid_dates") == 0
    )
    if manifest.get("run_id") != g3.get("l1_run_id") or not (
        manifest.get("status") == "passed" or formal_candidate
    ):
        raise RuntimeError("G43_L1_MANIFEST_NOT_PASSED_OR_MISMATCHED")
    exposure_path = lake.root / manifest["outputs"]["risk_exposure_matrix_v1"]["path"]
    schema = pl.read_parquet_schema(exposure_path)
    required = ["trade_date", "asset_id", "board_id", "sw_l1_code", "risk_size", "risk_liquidity"]
    required += ["risk_index_CSI300", "risk_index_CSI500"]
    missing = sorted(set(required) - set(schema))
    if missing:
        raise RuntimeError(f"G43_L1_EXPOSURE_FIELDS_MISSING {missing}")
    frame = pl.read_parquet(exposure_path, columns=required)
    return frame, manifest_path, exposure_path


def load_residual_panel(
    cfg: G43Config, lake: DataLake, g3: dict[str, Any], g3_manifest_path: Path,
) -> tuple[np.ndarray, list[str], list[str], dict[str, str]]:
    """Load one N×T panel from the committed G3 specific-return artifact."""
    path = lake.root / g3["outputs"]["specific_returns_v1"]["path"]
    frame = pl.read_parquet(path, columns=[
        "trade_date", "exposure_date", "asset_id", "regression_mode",
        "specific_return", "in_estimation_domain",
    ]).filter(
        (pl.col("regression_mode") == "risk_only")
        & pl.col("in_estimation_domain")
        & pl.col("specific_return").is_not_null()
    )
    if frame.is_empty():
        raise RuntimeError("G43_RESIDUAL_PANEL_EMPTY")
    dates = sorted(str(value) for value in frame["trade_date"].unique().sort().to_list())
    ids = sorted(str(value) for value in frame["asset_id"].unique().sort().to_list())
    date_index = {value: index for index, value in enumerate(dates)}
    asset_index = {value: index for index, value in enumerate(ids)}
    U = np.full((len(ids), len(dates)), np.nan, dtype=np.float64)
    for trade_date, asset_id, value in frame.select(
        "trade_date", "asset_id", "specific_return"
    ).iter_rows():
        U[asset_index[str(asset_id)], date_index[str(trade_date)]] = float(value)
    return U, ids, dates, {
        "g3_manifest": file_sha256(g3_manifest_path),
        "specific_returns": file_sha256(path),
    }


def _modal_labels(
    frame: pl.DataFrame, ids: list[str], value_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Collapse PIT daily labels to a modal label and retain stability metadata."""
    valid = frame.select("asset_id", value_column).drop_nulls()
    counts = valid.group_by(["asset_id", value_column]).len().sort(
        ["asset_id", "len", value_column], descending=[False, True, False]
    )
    totals = valid.group_by("asset_id").len().rename({"len": "total"})
    modal = counts.unique("asset_id", keep="first").join(totals, on="asset_id")
    modal = modal.with_columns((pl.col("len") / pl.col("total")).alias("stability"))
    categories = sorted(str(value) for value in modal[value_column].unique().to_list())
    category_index = {value: index for index, value in enumerate(categories)}
    values = {str(row["asset_id"]): row for row in modal.iter_rows(named=True)}
    labels = np.full(len(ids), -1, dtype=np.int64)
    stability = np.full(len(ids), np.nan, dtype=np.float64)
    for index, asset_id in enumerate(ids):
        row = values.get(asset_id)
        if row is not None:
            labels[index] = category_index[str(row[value_column])]
            stability[index] = float(row["stability"])
    return labels, {
        "value_column": value_column,
        "category_count": len(categories),
        "categories": categories,
        "median_stability": float(np.nanmedian(stability)) if np.isfinite(stability).any() else None,
        "minimum_stability": float(np.nanmin(stability)) if np.isfinite(stability).any() else None,
        "assigned_assets": int(np.sum(labels >= 0)),
    }


def _apply_label_stability_gate(
    labels: np.ndarray, metadata: dict[str, Any], *, threshold: float = _MIN_LABEL_STABILITY,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop modal labels whose PIT identity is not stable over the run interval."""
    updated = dict(metadata)
    median_stability = updated.get("median_stability")
    if median_stability is None or float(median_stability) < threshold:
        labels = labels.copy()
        labels[:] = -1
        updated["status"] = "dropped_label_stability_below_0.9"
        updated["stability_threshold"] = threshold
    else:
        updated["status"] = "usable"
        updated["stability_threshold"] = threshold
    updated["assigned_assets_after_gate"] = int(np.sum(labels >= 0))
    return labels, updated


def load_grouping_labels(
    cfg: G43Config, ids: list[str], exposure: pl.DataFrame,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, pl.DataFrame]:
    """Load only PIT-supported labels; concepts/themes and ownership are excluded."""
    labels: dict[str, dict[str, Any]] = {}
    for name in sorted(set(cfg.groupings) | set(cfg.negative_controls)):
        if name == "industry":
            values, meta = _modal_labels(exposure, ids, "sw_l1_code")
        elif name == "board":
            values, meta = _modal_labels(exposure, ids, "board_id")
        elif name == "index_CSI300":
            values, meta = _modal_labels(
                exposure.with_columns((pl.col("risk_index_CSI300") > 0).cast(pl.Int8).alias("label")),
                ids, "label",
            )
        elif name == "index_CSI500":
            values, meta = _modal_labels(
                exposure.with_columns((pl.col("risk_index_CSI500") > 0).cast(pl.Int8).alias("label")),
                ids, "label",
            )
        else:  # guarded by G43Config.validate, retained as a defensive gate
            raise ValueError(f"G43_GROUPING_NOT_IMPLEMENTED {name}")
        values, meta = _apply_label_stability_gate(values, meta)
        labels[name] = {"labels": values, "n_groups": meta["category_count"], "metadata": meta}

    strata, strata_meta = load_strata(cfg, ids, exposure)
    return labels, strata, pl.DataFrame([{"grouping": "strata", **strata_meta}])


def load_strata(
    cfg: G43Config, ids: list[str], exposure: pl.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build modal size-decile × liquidity-quintile strata from PIT exposures."""
    ranked = exposure.with_columns(
        [
            pl.col("risk_size").rank("ordinal").over("trade_date").alias("size_rank"),
            pl.col("risk_size").count().over("trade_date").alias("size_count"),
            pl.col("risk_liquidity").rank("ordinal").over("trade_date").alias("liq_rank"),
            pl.col("risk_liquidity").count().over("trade_date").alias("liq_count"),
        ]
    ).with_columns(
        [
            (((pl.col("size_rank") - 1) * 10 / pl.col("size_count"))
             .floor().clip(0, 9).cast(pl.Int64)).alias("size_decile"),
            (((pl.col("liq_rank") - 1) * 5 / pl.col("liq_count"))
             .floor().clip(0, 4).cast(pl.Int64)).alias("liquidity_quintile"),
        ]
    ).with_columns(
        pl.when(pl.col("size_decile").is_not_null() & pl.col("liquidity_quintile").is_not_null())
        .then(pl.col("size_decile") * 5 + pl.col("liquidity_quintile"))
        .otherwise(None)
        .alias("stratum")
    )
    values, metadata = _modal_labels(ranked, ids, "stratum")
    metadata["definition"] = cfg.strata_definition
    metadata["strata_count"] = int(np.sum(values >= 0) and len(set(values[values >= 0].tolist())))
    values, metadata = _apply_label_stability_gate(values, metadata)
    return values, metadata


def prepare(U: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, MAD-standardize, and winsorize each asset's residual history once."""
    med = np.nanmedian(U, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(U - med), axis=1, keepdims=True)
    delta = _MAD_NORMALIZER * np.maximum(mad, 1e-12)
    standardized = np.clip((U - med) / delta, -k, k)
    mask = ~np.isnan(standardized)
    variance = np.nanvar(standardized, axis=1)
    return np.nan_to_num(standardized), mask, variance


@dataclass(frozen=True)
class _GroupIndicator:
    """Small sparse-indicator equivalent using NumPy contiguous reductions."""

    member_indices: np.ndarray
    offsets: np.ndarray
    n_groups: int

    def __matmul__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        ordered = values[self.member_indices]
        return np.add.reduceat(ordered, self.offsets[:-1], axis=0)

    def sum(self, axis: int | None = None) -> np.ndarray | float:
        if axis == 1:
            return np.diff(self.offsets).astype(np.float64)
        if axis is None:
            return float(self.member_indices.size)
        raise ValueError("G43_INDICATOR_AXIS_UNSUPPORTED")


def build_indicator(labels: np.ndarray, n_groups: int) -> _GroupIndicator:
    keep = labels >= 0
    group_labels, member_indices = labels[keep], np.flatnonzero(keep)
    order = np.argsort(group_labels, kind="stable")
    sorted_groups = group_labels[order]
    counts = np.bincount(sorted_groups, minlength=n_groups)
    if np.any(counts == 0):
        raise ValueError("G43_EMPTY_GROUP_INDICATOR")
    return _GroupIndicator(
        member_indices=member_indices[order],
        offsets=np.concatenate(([0], np.cumsum(counts))),
        n_groups=n_groups,
    )


def group_stat(
    A: _GroupIndicator, U: np.ndarray, mask: np.ndarray,
    var_i: np.ndarray, min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return s_g = Var(group mean)/independent variance for each group."""
    count = np.asarray(A @ mask.astype(np.float64))
    valid = count >= min_obs
    means = np.asarray(A @ U) / np.maximum(count, 1.0)
    means = np.where(valid, means, np.nan)
    numerator = np.nanvar(means, axis=1)
    group_size = np.asarray(A.sum(axis=1)).ravel()
    denominator = np.asarray(A @ var_i).ravel() / np.maximum(group_size, 1.0) ** 2
    return numerator / np.maximum(denominator, 1e-300), group_size


def permute_within_strata(labels: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute labels only; residual values remain fixed."""
    output = labels.copy()
    for stratum in np.unique(strata[strata >= 0]):
        indices = np.flatnonzero(strata == stratum)
        output[indices] = labels[rng.permutation(indices)]
    return output


def drop_small_groups(labels: np.ndarray, min_size: int) -> tuple[np.ndarray, int, list[str]]:
    counts = np.bincount(labels[labels >= 0]) if np.any(labels >= 0) else np.array([], dtype=int)
    kept = [int(index) for index, count in enumerate(counts) if count >= min_size]
    mapping = {old: new for new, old in enumerate(kept)}
    output = np.full(labels.shape, -1, dtype=np.int64)
    for old, new in mapping.items():
        output[labels == old] = new
    return output, len(kept), [str(index) for index in kept]


def permutation_test(
    labels: np.ndarray, strata: np.ndarray, n_groups: int,
    U: np.ndarray, mask: np.ndarray, var_i: np.ndarray, cfg: G43Config,
) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    observed, group_size = group_stat(
        build_indicator(labels, n_groups), U, mask, var_i, cfg.min_obs_per_day
    )
    null = np.empty((cfg.n_permutations, n_groups), dtype=np.float64)
    for index in range(cfg.n_permutations):
        permuted = permute_within_strata(labels, strata, rng)
        null[index], _ = group_stat(
            build_indicator(permuted, n_groups), U, mask, var_i, cfg.min_obs_per_day
        )
    p_value = (1.0 + np.nansum(null >= observed, axis=0)) / (cfg.n_permutations + 1)
    return {
        "s_obs": observed,
        "p_value": p_value,
        "n_g": group_size,
        "null_median": np.nanmedian(null, axis=0),
        "null_q95": np.nanquantile(null, 0.95, axis=0),
        "implied_rho": (observed - 1.0) / np.maximum(group_size - 1.0, 1.0),
    }


def negative_control_gate(control_results: dict[str, dict[str, Any]], band: tuple[float, float]) -> dict[str, Any]:
    lo, hi = band
    detail: dict[str, Any] = {}
    for name, result in control_results.items():
        median = float(np.nanmedian(result["s_obs"]))
        detail[name] = {
            "s_median": median,
            "in_band": bool(np.isfinite(median) and lo <= median <= hi),
            "control_band": [lo, hi],
        }
    return {"passed": all(item["in_band"] for item in detail.values()), "detail": detail}


def apply_fdr_bh(results: dict[str, dict[str, Any]], q: float) -> dict[str, dict[str, Any]]:
    entries = [
        (name, index, float(value))
        for name, result in results.items()
        for index, value in enumerate(result["p_value"])
        if np.isfinite(value)
    ]
    q_values = {(name, index): np.nan for name, result in results.items() for index in range(len(result["p_value"]))}
    if entries:
        ordered = sorted(entries, key=lambda item: item[2])
        running = 1.0
        for rank, (name, index, value) in reversed(list(enumerate(ordered, start=1))):
            running = min(running, value * len(ordered) / rank)
            q_values[(name, index)] = running
    for name, result in results.items():
        q_array = np.asarray([q_values[(name, index)] for index in range(len(result["p_value"]))])
        result["q_value"] = q_array
        result["bh_passed"] = np.isfinite(q_array) & (q_array <= q)
    return results


def _result_frame(results: dict[str, dict[str, Any]], control: bool) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        for index in range(len(result["s_obs"])):
            rows.append({
                "grouping": name,
                "group_index": index,
                "s_observed": float(result["s_obs"][index]),
                "p_value": float(result["p_value"][index]),
                "q_value": None if control else float(result["q_value"][index]),
                "bh_passed": None if control else bool(result["bh_passed"][index]),
                "n_group": int(result["n_g"][index]),
                "null_median": float(result["null_median"][index]),
                "null_q95": float(result["null_q95"][index]),
                "implied_rho": float(result["implied_rho"][index]),
                "is_negative_control": control,
            })
    return pl.DataFrame(rows)


def _label_stability_frame(label_specs: dict[str, dict[str, Any]], strata_meta: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name, spec in label_specs.items():
        rows.append({"grouping": name, **spec["metadata"]})
    return pl.concat([pl.DataFrame(rows), strata_meta], how="diagonal_relaxed")


def run_g4_3(
    lake: DataLake, *, config_path: Path = Path("config/g4_3_residual_permutation_v1.json"),
) -> Path:
    cfg = G43Config.from_json(config_path)
    cfg.validate()
    g3, g3_manifest_path = load_current_g3_manifest(lake)
    if cfg.input_run_id != g3["run_id"]:
        raise RuntimeError("G43_CONFIG_CURRENT_G3_MISMATCH")
    exposure, l1_manifest_path, exposure_path = _load_l1_exposure_frame(lake, g3)
    U, ids, dates, residual_hashes = load_residual_panel(cfg, lake, g3, g3_manifest_path)
    labels, strata, strata_frame = load_grouping_labels(cfg, ids, exposure)
    strata_metadata = strata_frame.to_dicts()[0]
    U_prepared, mask, var_i = prepare(U, cfg.winsor_mad_k)
    pointer_before = (lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json").read_bytes()
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "g3_run_id": g3["run_id"],
        "l1_run_id": g3["l1_run_id"],
        "config_sha256": file_sha256(config_path),
        "g3_manifest_sha256": residual_hashes["g3_manifest"],
        "specific_returns_sha256": residual_hashes["specific_returns"],
        "l1_manifest_sha256": file_sha256(l1_manifest_path),
        "exposure_sha256": file_sha256(exposure_path),
        "code_sha": source_tree_hash(),
    }
    run_id = f"g4_3_residual_permutation_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "g4_3_residual_permutation" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)

    # A modal stratum with low interval stability does not define a valid
    # permutation universe.  Fail closed before spending the permutation
    # budget and preserve the reason as an immutable diagnostic artifact.
    if strata_metadata.get("status") != "usable":
        invalid_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "runner": IMPLEMENTATION_ID,
            "job": "g4_3_stratified_residual_permutation",
            "mode": "read_only",
            "status": "INVALID_RUNNER",
            "created_at": utc_now().isoformat(),
            "identity": identity,
            "input_run_id": g3["run_id"],
            "input_hashes": {
                "g3_manifest": residual_hashes["g3_manifest"],
                "specific_returns": residual_hashes["specific_returns"],
                "l1_manifest": file_sha256(l1_manifest_path),
                "risk_exposure_matrix": file_sha256(exposure_path),
                "config": file_sha256(config_path),
            },
            "frozen_config_refs": {
                "strata_definition": cfg.strata_definition,
                "n_permutations": cfg.n_permutations,
                "seed": cfg.seed,
                "stability_threshold": _MIN_LABEL_STABILITY,
            },
            "lineage": {
                "n_assets": len(ids),
                "n_dates": len(dates),
                "g3_risk_basis_id": g3.get("risk_basis_id"),
                "g3_risk_metric_id": g3.get("risk_metric_id"),
            },
            "consumes_attempt_budget": False,
            "reads_forward_returns": False,
            "current_pointer_mutated": False,
            "invalid_reasons": ["G43_STRATA_LABEL_STABILITY_BELOW_0.9"],
            "negative_control": {"status": "not_evaluated"},
            "strata": strata_metadata,
        }
        return lake.write_immutable_json(manifest_path, invalid_manifest)

    def test(name: str) -> dict[str, Any]:
        spec = labels[name]
        compact, n_groups, kept = drop_small_groups(spec["labels"], cfg.min_group_size)
        if n_groups == 0:
            raise RuntimeError(f"G43_NO_GROUPS_AFTER_MIN_SIZE grouping={name}")
        result = permutation_test(compact, strata, n_groups, U_prepared, mask, var_i, cfg)
        result["kept_group_indices"] = kept
        return result

    controls = {name: test(name) for name in cfg.negative_controls}
    gate = negative_control_gate(controls, cfg.control_band)
    base_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "runner": IMPLEMENTATION_ID,
        "job": "g4_3_stratified_residual_permutation",
        "mode": "read_only",
        "status": "passed" if gate["passed"] else "INVALID_RUNNER",
        "created_at": utc_now().isoformat(),
        "identity": identity,
        "input_run_id": g3["run_id"],
        "input_hashes": {
            "g3_manifest": residual_hashes["g3_manifest"],
            "specific_returns": residual_hashes["specific_returns"],
            "l1_manifest": file_sha256(l1_manifest_path),
            "risk_exposure_matrix": file_sha256(exposure_path),
            "config": file_sha256(config_path),
        },
        "frozen_config_refs": {
            "strata_definition": cfg.strata_definition,
            "groupings": list(cfg.groupings),
            "negative_controls": list(cfg.negative_controls),
            "n_permutations": cfg.n_permutations,
            "seed": cfg.seed,
            "winsor_mad_k": cfg.winsor_mad_k,
            "min_group_size": cfg.min_group_size,
            "min_obs_per_day": cfg.min_obs_per_day,
            "fdr_q": cfg.fdr_q,
            "control_band": list(cfg.control_band),
        },
        "lineage": {
            "n_assets": len(ids),
            "n_dates": len(dates),
            "g3_risk_basis_id": g3.get("risk_basis_id"),
            "g3_risk_metric_id": g3.get("risk_metric_id"),
        },
        "consumes_attempt_budget": False,
        "reads_forward_returns": False,
        "current_pointer_mutated": False,
        "rationale": "Group labels are permuted within strata; residual values are never permuted.",
        "excluded_groupings": {
            "concept_theme": "no historical PIT snapshot; backfilling current membership would be look-ahead",
            "ownership": "no PIT source in L0",
        },
        "negative_control": gate,
        "decision_rule_prereg": cfg.on_significant,
        "interpretation_note": cfg.interpretation_note,
        "strata": strata_metadata,
    }
    if not gate["passed"]:
        if pointer_before != (lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json").read_bytes():
            raise RuntimeError("G43_CURRENT_POINTER_MUTATED")
        return lake.write_immutable_json(manifest_path, base_manifest)

    results = {name: test(name) for name in cfg.groupings}
    results = apply_fdr_bh(results, cfg.fdr_q)
    outputs = {
        "g4_3_group_results_v1": run_dir / "g4_3_group_results_v1.parquet",
        "g4_3_negative_controls_v1": run_dir / "g4_3_negative_controls_v1.parquet",
        "g4_3_label_stability_v1": run_dir / "g4_3_label_stability_v1.parquet",
    }
    _write_parquet(outputs["g4_3_group_results_v1"], _result_frame(results, False))
    _write_parquet(outputs["g4_3_negative_controls_v1"], _result_frame(controls, True))
    _write_parquet(outputs["g4_3_label_stability_v1"], _label_stability_frame(labels, strata_frame))
    if pointer_before != (lake.root / "gold" / "l2b_risk_only" / "_CURRENT.json").read_bytes():
        raise RuntimeError("G43_CURRENT_POINTER_MUTATED")
    base_manifest["outputs"] = {name: lake.artifact_record(path) for name, path in outputs.items()}
    base_manifest["headline"] = {
        "candidate_groupings": list(cfg.groupings),
        "significant_group_count": int(sum(np.sum(result["bh_passed"]) for result in results.values())),
        "control_s_medians": gate["detail"],
    }
    return lake.write_immutable_json(manifest_path, base_manifest)
