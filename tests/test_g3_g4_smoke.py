"""One-day numeric G3 -> parquet -> DuckDB/Polars G4 smoke path."""

from pathlib import Path

import numpy as np
import polars as pl

from factor_matrix.calculation.l2 import (
    RegressionMode, build_daily_categorical_constrained_design,
    decompose_cross_section, load_return_decomposition_config,
)
from factor_matrix.research_protocol import ResearchProtocol
from factor_matrix.storage import open_duckdb


PROJECT = Path(__file__).resolve().parents[1]


def test_single_day_g3_to_g4_duckdb_polars_smoke(tmp_path: Path) -> None:
    count = 40
    cap = np.linspace(100.0, 4000.0, count)
    size = np.linspace(-1.0, 1.0, count)
    industry_a = np.asarray([index < count // 2 for index in range(count)], dtype=float)
    industry_b = 1.0 - industry_a
    returns = 0.01 + 0.02 * size + np.sin(np.arange(count)) * 0.003
    frame = pl.DataFrame({
        "asset_id": [f"A{index:03d}" for index in range(count)],
        "realized_return": returns, "float_mkt_cap": cap,
        "in_estimation_domain": [True] * count, "model_eligible": [True] * count,
        "risk_country": [1.0] * count, "risk_size": size,
        "risk_industry_a": industry_a, "risk_industry_b": industry_b,
    })
    design = build_daily_categorical_constrained_design(
        frame,
        exposure_columns=(
            "risk_country", "risk_size", "risk_industry_a", "risk_industry_b",
        ),
        family_by_column={
            name: "risk" for name in (
                "risk_country", "risk_size", "risk_industry_a", "risk_industry_b",
            )
        },
        categorical_blocks={
            "industry": ("risk_industry_a", "risk_industry_b"),
        },
        estimation_column="in_estimation_domain",
    )
    derived = ResearchProtocol.load(PROJECT / "config" / "research_protocol_v1.json").derive(
        factor_count=4, maximum_evaluation_horizon_days=1,
        cross_section_size=count, available_estimation_days=252,
    )["values"]
    config = load_return_decomposition_config(
        PROJECT / "config" / "l2_return_decomposition_v1.json",
        derived_params=derived,
    )
    result = decompose_cross_section(
        design.regression_input, mode=RegressionMode.RISK_ONLY, config=config,
    )
    path = tmp_path / "specific_returns_v1.parquet"
    pl.DataFrame({
        "asset_id": result.included_assets,
        "specific_return": result.specific_returns,
        "is_outlier_flagged": [value < 1.0 for value in result.huber_weight_multipliers],
    }).write_parquet(path)
    with open_duckdb() as connection:
        diagnostic = connection.execute(
            "SELECT count(*) n, avg(abs(specific_return)) mean_abs_u, "
            "avg(is_outlier_flagged::INTEGER) downweight_rate "
            f"FROM read_parquet('{str(path).replace("'", "''")}')"
        ).pl()
    assert diagnostic["n"][0] == count
    assert np.isfinite(diagnostic["mean_abs_u"][0])
    assert 0.0 <= diagnostic["downweight_rate"][0] <= 1.0
