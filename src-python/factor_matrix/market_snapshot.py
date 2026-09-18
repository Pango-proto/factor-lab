"""Registered market-matrix snapshot assembly and frontend summary."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from .artifact_io import write_json, write_parquet
from .storage import DataLake, utc_now
from .research import (
    FINANCIAL_FEATURES,
    MARKET_FEATURES,
    MATRIX_INPUT_TABLES,
    MatrixConfig,
    _existing_tables,
    _reason_counts,
    build_feature_values,
    build_market_universe,
    resolve_market_date,
)

def _frontend_payload(
    universe: pl.DataFrame,
    features: pl.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    sample = (
        universe.filter(pl.col("eligible_market_matrix"))
        .sort("float_mkt_cap", descending=True)
        .head(12)
        .select(
            "asset_id",
            "name",
            "exchange",
            "board_id",
            "industry",
            "l1_code",
            "l1_name",
            "l2_code",
            "l2_name",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "gross_open_index",
            "gross_close_index",
            "adj_factor",
            "limit_up",
            "limit_down",
            "total_return",
            "total_mkt_cap",
            "float_mkt_cap",
            "pb",
            "pe_ttm",
            "turnover_rate",
            "book_equity",
            "book_report_period",
            "financial_available_at",
        )
        .to_dicts()
    )
    return {
        "schema_version": 2,
        "generated_at": metadata["created_at"],
        "as_of_date": metadata["as_of_date"],
        "matrix_id": metadata["matrix_id"],
        "counts": metadata["counts"],
        "exclusions": metadata["exclusions"],
        "quality_checks": metadata["quality_checks"],
        "features": [
            {"id": "raw_close", "label": "原始收盘价", "format": "price"},
            {"id": "gross_close_index", "label": "累计复权收盘指数", "format": "index"},
            {"id": "total_return", "label": "当日总收益", "format": "percent"},
            {"id": "float_mkt_cap", "label": "流通市值", "format": "cny"},
            {"id": "total_mkt_cap", "label": "总市值", "format": "cny"},
            {"id": "pb", "label": "市净率", "format": "ratio"},
            {"id": "pe_ttm", "label": "PE TTM", "format": "ratio"},
            {"id": "turnover_rate", "label": "换手率", "format": "percent_point"},
            {"id": "book_equity", "label": "点时账面权益", "format": "cny"},
        ],
        "sample": sample,
        "stages": metadata["stages"],
        "lineage": {
            "run_id": metadata["matrix_id"],
            "data_snapshot_id": metadata["data_snapshot_id"],
            "config_hash": metadata["config_hash"],
            "code_hash": metadata["code_hash"],
            "market_features": "Silver prices_daily + valuation_daily → Gold feature_values",
            "price_policy": "研究仅使用原始OHLC、累计复权因子和无锚点总收益指数；前复权仅供K线展示",
            "lookahead_policy": "市场字段 available_at = as_of_date；财务字段只选 available_at <= as_of_date 的PIT版本",
        },
    }


def build_market_matrix(
    lake: DataLake,
    requested_date: date | None = None,
    config: MatrixConfig | None = None,
    tradable_universe: dict[str, Any] | None = None,
    frontend_output: Path | None = None,
) -> dict[str, Any]:
    config = config or MatrixConfig()
    as_of_date = resolve_market_date(lake, requested_date)
    if tradable_universe is None:
        raise ValueError("TRADABLE_UNIVERSE_ARTIFACT_REQUIRED")
    if not (
        date.fromisoformat(tradable_universe["start"])
        <= as_of_date
        <= date.fromisoformat(tradable_universe["end"])
    ):
        raise ValueError("TRADABLE_UNIVERSE_DATE_MISMATCH")
    universe_path = lake.root / tradable_universe["artifact"]
    if not universe_path.exists():
        raise RuntimeError("TRADABLE_UNIVERSE_ARTIFACT_MISSING")
    config_payload = {**asdict(config), "as_of_date": as_of_date.isoformat()}
    data_snapshot = lake.register_data_snapshot(
        as_of_date,
        _existing_tables(lake, MATRIX_INPUT_TABLES + ("board_membership_history",)),
    )
    parent_run_ids = [data_snapshot["run_id"], tradable_universe["run_id"]]
    matrix_id, config_hash, code_hash = lake.calculation_run_id(
        "market_matrix",
        as_of_date,
        config_payload,
        parent_run_ids,
    )
    output_dir = lake.root / "gold" / "matrices" / f"matrix_id={matrix_id}"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and (lake.manifests / f"{matrix_id}.json").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if frontend_output is not None:
            write_json(
                frontend_output,
                _frontend_payload(
                    pl.read_parquet(output_dir / "universe.parquet"),
                    pl.read_parquet(output_dir / "feature_values.parquet"),
                    metadata,
                ),
            )
        return metadata
    universe = build_market_universe(lake, as_of_date, config)
    authoritative_domain = (
        pl.read_parquet(universe_path)
        .filter(pl.col("trade_date") == as_of_date)
        .select("asset_id", "is_tradable", "can_buy", "can_sell", "exclusion_reasons")
        .unique("asset_id")
    )
    universe = (
        universe.drop("eligible_market_matrix")
        .join(authoritative_domain, on="asset_id", how="left")
        .with_columns(
            pl.col("is_tradable").fill_null(False).alias("eligible_market_matrix"),
            pl.col("can_buy").fill_null(False),
            pl.col("can_sell").fill_null(False),
        )
        .with_columns(
            (
                pl.col("is_tradable").fill_null(False)
                & pl.col("l2_code").is_not_null()
                & pl.col("l2_name").is_not_null()
            ).alias("eligible_industry_model")
        )
    )
    features = build_feature_values(universe, as_of_date)
    snapshot = universe.filter(pl.col("eligible_market_matrix")).select(
        "as_of_date", "asset_id", "name", "exchange", "board_id", "industry",
        "classification_source", "classification_version",
        "l1_code", "l1_name", "l2_code", "l2_name", "eligible_industry_model", *MARKET_FEATURES,
        *FINANCIAL_FEATURES, "book_report_period", "financial_available_at"
    )

    eligible_count = snapshot.height
    checks = {
        "duplicate_assets": snapshot.height - snapshot.unique("asset_id").height,
        "future_available_values": features.filter(pl.col("available_at") > as_of_date).height,
        "missing_market_cap": snapshot.filter(
            pl.col(config.market_cap_field).is_null() | (pl.col(config.market_cap_field) <= 0)
        ).height,
    }
    if eligible_count == 0 or any(checks.values()):
        raise RuntimeError(f"MATRIX_QUALITY_FAILED {checks} eligible={eligible_count}")

    write_parquet(output_dir / "matrix.parquet", snapshot)
    write_parquet(output_dir / "universe.parquet", universe)
    write_parquet(output_dir / "feature_values.parquet", features)

    created_at = utc_now().isoformat()
    completed_factor_manifests = []
    for path in lake.manifests.glob("factor_data_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("quality_gate", {}).get("status") == "passed":
            completed_factor_manifests.append(path)
    factor_data_ready = bool(completed_factor_manifests)
    financial_ready = factor_data_ready and (lake.silver / "financial_pit" / "data.parquet").exists()
    risk_free_ready = factor_data_ready and (lake.silver / "risk_free_daily" / "data.parquet").exists()
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "matrix_id": matrix_id,
        "as_of_date": as_of_date.isoformat(),
        "created_at": created_at,
        "config": config_payload,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "data_snapshot_id": data_snapshot["run_id"],
        "tradable_universe_run_id": tradable_universe["run_id"],
        "counts": {
            "listed": universe.height,
            "eligible_market_matrix": eligible_count,
            "eligible_industry_model": int(
                universe.get_column("eligible_industry_model").sum()
            ),
            "feature_values": features.height,
            "populated_feature_values": features.filter(pl.col("value").is_not_null()).height,
            "feature_count": len(MARKET_FEATURES) + len(FINANCIAL_FEATURES),
            "market_feature_count": len(MARKET_FEATURES),
            "financial_feature_count": len(FINANCIAL_FEATURES),
            "by_board": {
                board_id: {
                    "listed": universe.filter(pl.col("board_id") == board_id).height,
                    "eligible_market_matrix": universe.filter(
                        (pl.col("board_id") == board_id) & pl.col("eligible_market_matrix")
                    ).height,
                }
                for board_id in ("MAIN", "CHINEXT", "STAR", "BSE")
            },
        },
        "exclusions": _reason_counts(universe),
        "quality_checks": checks,
        "stages": [
            {"id": "market", "label": "市场标准层", "status": "complete"},
            {"id": "universe", "label": "市场股票池", "status": "complete"},
            {"id": "market_features", "label": "市场特征长表", "status": "complete"},
            {"id": "financial_pit", "label": "点时财务", "status": "complete" if financial_ready else "blocked"},
            {"id": "risk_free", "label": "无风险利率", "status": "complete" if risk_free_ready else "blocked"},
        ],
        "artifacts": {
            "matrix": str((output_dir / "matrix.parquet").relative_to(lake.root)),
            "universe": str((output_dir / "universe.parquet").relative_to(lake.root)),
            "feature_values": str((output_dir / "feature_values.parquet").relative_to(lake.root)),
        },
    }
    write_json(metadata_path, metadata)
    calculation_inputs = {
        "data_snapshot_manifest": lake.manifests / f"{data_snapshot['run_id']}.json",
        "tradable_universe_manifest": lake.manifests
        / f"{tradable_universe['run_id']}.json",
    }
    lake.register_calculation(
        run_id=matrix_id,
        job="market_matrix",
        as_of=as_of_date,
        mode="registered_run",
        config=config_payload,
        config_hash=config_hash,
        code_hash=code_hash,
        parent_run_ids=parent_run_ids,
        inputs=calculation_inputs,
        outputs={
            "matrix": output_dir / "matrix.parquet",
            "universe": output_dir / "universe.parquet",
            "feature_values": output_dir / "feature_values.parquet",
            "metadata": metadata_path,
        },
        quality_gate={"status": "passed", **checks, "eligible": eligible_count},
        extra={"counts": metadata["counts"], "exclusions": metadata["exclusions"]},
    )
    if frontend_output is not None:
        write_json(frontend_output, _frontend_payload(universe, features, metadata))
    return metadata
