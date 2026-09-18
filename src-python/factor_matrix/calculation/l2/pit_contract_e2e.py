"""End-to-end PIT alignment canaries without alpha-attempt accounting.

The probe deliberately uses future realized returns as *synthetic exposures*:
the t+1 return must produce IC=1 against the t+1 label, while the t+5 return
must not produce a t+1 signal.  It is a contract test, not an alpha result.
The placebo keeps each day's cross-sectional values intact and shuffles only
asset assignments.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...revisioned_silver import read_as_of_frame
from ...storage import DataLake, file_sha256, json_hash, source_tree_hash, utc_now


IMPLEMENTATION_ID = "pit_contract_e2e_alignment_v1"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _ic(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size != right.size or left.size < 3:
        return None
    x, y = _rank(left), _rank(right)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def build_alignment_rows(
    returns: pl.DataFrame, *, sample_days: int, minimum_cross_section: int,
) -> pl.DataFrame:
    required = {"trade_date", "asset_id", "total_return"}
    missing = sorted(required - set(returns.columns))
    if missing:
        raise ValueError(f"PIT_E2E_RETURNS_COLUMNS_MISSING columns={','.join(missing)}")
    clean = returns.select("trade_date", "asset_id", "total_return").filter(
        pl.col("total_return").is_finite()
    )
    dates = clean.get_column("trade_date").unique().sort().to_list()
    if len(dates) < sample_days + 5:
        raise ValueError("PIT_E2E_INSUFFICIENT_DATE_COVERAGE")
    evaluation_dates = dates[-sample_days - 5:-5]
    date_map = pl.DataFrame({
        "evaluation_date": evaluation_dates,
        "label_date": dates[-sample_days - 4:-4],
        "lead5_date": dates[-sample_days:],
    })

    def align(source_date: str, value_name: str) -> pl.DataFrame:
        return clean.join(
            date_map.select("evaluation_date", source_date),
            left_on="trade_date", right_on=source_date, how="inner",
        ).select("evaluation_date", "asset_id", pl.col("total_return").alias(value_name))

    labels = align("label_date", "label_t1")
    lead1 = align("label_date", "factor_t1")
    lead5 = align("lead5_date", "factor_t5")
    rows = labels.join(lead1, on=["evaluation_date", "asset_id"], how="inner").join(
        lead5, on=["evaluation_date", "asset_id"], how="inner"
    )
    valid_dates = rows.group_by("evaluation_date").len().filter(
        pl.col("len") >= minimum_cross_section
    ).get_column("evaluation_date").to_list()
    return rows.filter(pl.col("evaluation_date").is_in(valid_dates)).sort(
        ["evaluation_date", "asset_id"]
    )


def summarize_alignment(
    rows: pl.DataFrame, *, permutations: int, seed: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    if rows.is_empty():
        raise ValueError("PIT_E2E_EMPTY_ALIGNMENT_SAMPLE")
    daily: list[dict[str, Any]] = []
    for group in rows.partition_by("evaluation_date", maintain_order=True):
        label = group["label_t1"].to_numpy()
        t1 = group["factor_t1"].to_numpy()
        t5 = group["factor_t5"].to_numpy()
        daily.append({
            "evaluation_date": group["evaluation_date"][0],
            "cross_section": int(group.height),
            "ic_t1_factor_vs_t1_label": _ic(t1, label),
            "ic_t5_factor_vs_t1_label": _ic(t5, label),
            "factor_t1_available_date": group["evaluation_date"][0],
        })
    daily_frame = pl.DataFrame(daily)
    t1_values = [x for x in daily_frame["ic_t1_factor_vs_t1_label"].to_list() if x is not None]
    t5_values = [x for x in daily_frame["ic_t5_factor_vs_t1_label"].to_list() if x is not None]
    rng = np.random.default_rng(seed)
    placebo_means: list[float] = []
    for _ in range(permutations):
        daily_placebo: list[float] = []
        for group in rows.partition_by("evaluation_date", maintain_order=True):
            factor = group["factor_t1"].to_numpy()
            label = group["label_t1"].to_numpy()
            value = _ic(factor[rng.permutation(factor.size)], label)
            if value is not None:
                daily_placebo.append(value)
        if daily_placebo:
            placebo_means.append(float(np.mean(daily_placebo)))
    placebo = pl.DataFrame({"permutation": np.arange(len(placebo_means)), "mean_ic": placebo_means})
    summary = {
        "evaluation_dates": daily_frame.height,
        "minimum_cross_section": int(daily_frame["cross_section"].min()),
        "t_plus_1_factor_mean_daily_ic": float(np.mean(t1_values)),
        "t_plus_5_factor_mean_daily_ic": float(np.mean(t5_values)),
        "t_plus_1_factor_min_daily_ic": float(np.min(t1_values)),
        "t_plus_5_factor_abs_mean_daily_ic": float(abs(np.mean(t5_values))),
        "placebo_permutations": len(placebo_means),
        "placebo_mean_ic": float(np.mean(placebo_means)),
        "placebo_q01": float(np.quantile(placebo_means, 0.01)),
        "placebo_q99": float(np.quantile(placebo_means, 0.99)),
        "daily": daily_frame,
    }
    return summary, placebo


def run_pit_contract_e2e(
    lake: DataLake, *, config_path: Path = Path("config/pit_contract_e2e_v1.json")
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    as_of = utc_now()
    returns = read_as_of_frame(lake, "returns_daily", as_of)
    rows = build_alignment_rows(
        returns,
        sample_days=int(config["sample_days"]),
        minimum_cross_section=int(config["minimum_cross_section"]),
    )
    summary, placebo = summarize_alignment(
        rows, permutations=int(config["placebo_permutations"]), seed=int(config["seed"])
    )
    t1_passed = summary["t_plus_1_factor_min_daily_ic"] >= float(config["t_plus_1_min_ic"])
    t5_passed = summary["t_plus_5_factor_abs_mean_daily_ic"] <= float(config["t_plus_5_max_abs_mean_ic"])
    placebo_passed = (
        abs(summary["placebo_mean_ic"]) <= float(config["placebo_max_abs_mean_ic"])
        and summary["placebo_q99"] <= float(config["placebo_q99_max"])
    )
    identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "config_sha": file_sha256(config_path),
        "source_tree_sha": source_tree_hash(),
        "as_of_timestamp": as_of.isoformat(),
        "source_table": "returns_daily",
        "source_query": "read_as_of_frame(returns_daily, as_of_timestamp)",
    }
    run_id = f"pit_contract_e2e_{json_hash(identity)[:16]}"
    run_dir = lake.root / "diagnostics" / "pit_contract_e2e" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=False)
    daily_path = run_dir / "pit_alignment_daily_v1.parquet"
    placebo_path = run_dir / "pit_alignment_placebo_v1.parquet"
    summary["daily"].write_parquet(daily_path)
    placebo.write_parquet(placebo_path)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "pit_contract_e2e",
        "status": "passed" if t1_passed and t5_passed and placebo_passed else "failed",
        "created_at": utc_now().isoformat(),
        "implementation_id": IMPLEMENTATION_ID,
        "identity": identity,
        "reads_forward_returns": True,
        "produces_alpha_result": False,
        "consumes_research_attempt": False,
        "contract": {
            "t_plus_1_factor_vs_t_plus_1_label": "approximately_one",
            "t_plus_5_factor_vs_t_plus_1_label": "approximately_zero",
            "placebo": "within_day_cross_sectional_asset_assignment_permutation",
            "row_level_pit": "source queried through revisioned as_of reader; no carry-forward",
        },
        "summary": {key: value for key, value in summary.items() if key != "daily"},
        "gates": {"t_plus_1": t1_passed, "t_plus_5": t5_passed, "placebo": placebo_passed},
        "outputs": {
            "pit_alignment_daily_v1": lake.artifact_record(daily_path),
            "pit_alignment_placebo_v1": lake.artifact_record(placebo_path),
        },
    }
    return lake.write_immutable_json(manifest_path, manifest)
