#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from factor_matrix.calculation.l2 import (
    L2AtomicPublisher, L2BParquetInputs, RegressionMode,
    load_return_decomposition_config, run_l2b_from_parquet,
)
from factor_matrix.storage import DataLake
from factor_matrix.research_protocol import ProtocolSealStore, ResearchProtocol


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the stateless L2b Parquet pipeline")
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--exposure-path", type=Path, required=True)
    result.add_argument("--universe-path", type=Path, required=True)
    result.add_argument("--returns-path", type=Path, required=True)
    result.add_argument("--valuation-path", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("config/l2_return_decomposition_v1.json"))
    result.add_argument("--mode", choices=[item.value for item in RegressionMode], required=True)
    result.add_argument("--exposure-columns", required=True, help="Comma-separated physical columns")
    result.add_argument(
        "--factor-families-json", required=True,
        help='JSON mapping, for example {"country":"risk","size":"risk"}',
    )
    result.add_argument("--risk-set-version", type=int, required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    paths = L2BParquetInputs(
        exposure_path=arguments.exposure_path,
        tradable_universe_path=arguments.universe_path,
        realized_returns_path=arguments.returns_path,
        valuation_path=arguments.valuation_path,
    )
    mode = RegressionMode(arguments.mode)
    columns = tuple(item.strip() for item in arguments.exposure_columns.split(",") if item.strip())
    families = json.loads(arguments.factor_families_json)
    configuration_payload = json.loads(arguments.config.read_text(encoding="utf-8"))
    protocol = ResearchProtocol.load(Path("config/research_protocol_v1.json"))
    ProtocolSealStore(arguments.data_root / "metadata" / "research_protocol.sqlite").seal(protocol)
    exposure_stats = pl.scan_parquet(arguments.exposure_path).select(
        pl.len().alias("rows"), pl.col("trade_date").n_unique().alias("dates"),
        pl.col("industry_id").n_unique().alias("industries"),
    ).collect().row(0, named=True)
    derived_snapshot = protocol.derive(
        factor_count=len(columns), maximum_evaluation_horizon_days=1,
        cross_section_size=max(1, exposure_stats["rows"] // exposure_stats["dates"]),
        available_estimation_days=exposure_stats["dates"],
    )
    derived_params = derived_snapshot["values"]
    products = run_l2b_from_parquet(
        paths,
        exposure_columns=columns,
        family_by_column=families,
        mode=mode,
        config=load_return_decomposition_config(
            arguments.config, derived_params=derived_params,
        ),
    )
    quality = products["factor_regression_quality_v1"]
    date_range = (quality.get_column("trade_date").min(), quality.get_column("trade_date").max())
    run_id = L2AtomicPublisher(DataLake(arguments.data_root)).publish_l2b(
        products=products,
        mode=mode.value,
        configuration=configuration_payload,
        risk_set_version=arguments.risk_set_version,
        input_paths=paths.paths(),
        date_range=date_range,
        derived_params=derived_params,
    )
    print(json.dumps({"status": "published", "run_id": run_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
