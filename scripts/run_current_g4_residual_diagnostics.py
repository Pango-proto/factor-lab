"""Build lightweight structural G4 residual diagnostics from published G3 outputs.

This is deliberately read-only with respect to production outputs.  It does not
re-estimate a cross-section, rebuild factors, or mutate a CURRENT pointer.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
CURRENT = DATA / "gold/l2b_risk_only/_CURRENT.json"
L1_CURRENT = DATA / "gold/risk_exposure_matrix/_CURRENT.json"
CONTRACT = PROJECT / "config/weight_metric_contract_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _write_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, CODEC 'zstd')"
    )
    os.replace(temporary, path)


def main() -> int:
    current = _load(CURRENT)
    g3_path = DATA / current["manifest"]
    g3 = _load(g3_path)
    l1_current = _load(L1_CURRENT)
    l1_path = DATA / l1_current["manifest"]
    l1 = _load(l1_path)
    contract = _load(CONTRACT)

    base = contract["regression_base_weight"]
    weight_path = DATA / base["artifact_path"]
    if g3.get("base_weight_scheme_id") != base["scheme_id"]:
        raise RuntimeError("CURRENT_G3_WEIGHT_SCHEME_CONTRACT_MISMATCH")
    if g3["identity"]["input_hashes"].get("base_weight_artifact") != base["artifact_sha256"]:
        raise RuntimeError("CURRENT_G3_WEIGHT_ARTIFACT_HASH_MISMATCH")
    if _sha256(weight_path) != base["artifact_sha256"]:
        raise RuntimeError("WEIGHT_ARTIFACT_CONTENT_HASH_MISMATCH")

    specific_path = DATA / g3["outputs"]["specific_returns_v1"]["path"]
    quality_path = DATA / g3["outputs"]["factor_regression_quality_v1"]["path"]
    exposure_path = DATA / l1["outputs"]["risk_exposure_matrix_v1"]["path"]
    run_identity = {
        "job": "g4_current_structural_residual_diagnostics",
        "g3_run_id": g3["run_id"],
        "l1_run_id": l1["run_id"],
        "specific_sha256": g3["outputs"]["specific_returns_v1"]["sha256"],
        "quality_sha256": g3["outputs"]["factor_regression_quality_v1"]["sha256"],
        "exposure_sha256": l1["outputs"]["risk_exposure_matrix_v1"]["sha256"],
        "weight_sha256": base["artifact_sha256"],
        "aggregation": "canonical_g4_residual_group_queries_v1",
    }
    run_hash = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"g4_current_structural_residual_{g3['end'].replace('-', '')}_{run_hash}"
    run_dir = DATA / "diagnostics/g4_current_structural_residual" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        print(manifest_path)
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)

    connection = duckdb.connect()
    try:
        connection.execute(
            f"CREATE TEMP VIEW specific AS SELECT * FROM read_parquet('{_sql_path(specific_path)}')"
        )
        connection.execute(
            f"CREATE TEMP VIEW quality AS SELECT * FROM read_parquet('{_sql_path(quality_path)}')"
        )
        connection.execute(
            f"CREATE TEMP VIEW exposure AS SELECT * FROM read_parquet('{_sql_path(exposure_path)}')"
        )
        connection.execute(
            """
            CREATE TEMP VIEW ranked AS
            SELECT s.trade_date,s.exposure_date,s.asset_id,s.specific_return,
                   s.is_outlier_flagged,e.board_id,e.sw_l1_code,
                   e.risk_liquidity,e.risk_residual_volatility,
                   ntile(10) OVER(
                     PARTITION BY s.exposure_date ORDER BY e.risk_liquidity
                   ) AS liquidity_decile
            FROM specific s
            JOIN exposure e
              ON e.trade_date=s.exposure_date AND e.asset_id=s.asset_id
            WHERE s.in_estimation_domain
              AND s.specific_return IS NOT NULL
              AND e.is_valid
            """
        )

        _write_query(
            connection,
            """
            SELECT 'board' AS dimension,CAST(board_id AS VARCHAR) AS group_id,
                   count(*) AS n_observations,
                   avg(is_outlier_flagged::INTEGER) AS downweight_rate,
                   avg(abs(specific_return)) FILTER(WHERE is_outlier_flagged)
                     AS mean_abs_u_downweighted,
                   avg(abs(specific_return)) FILTER(WHERE NOT is_outlier_flagged)
                     AS mean_abs_u_not_downweighted
            FROM ranked GROUP BY board_id
            UNION ALL
            SELECT 'liquidity_decile',CAST(liquidity_decile AS VARCHAR),count(*),
                   avg(is_outlier_flagged::INTEGER),
                   avg(abs(specific_return)) FILTER(WHERE is_outlier_flagged),
                   avg(abs(specific_return)) FILTER(WHERE NOT is_outlier_flagged)
            FROM ranked GROUP BY liquidity_decile
            ORDER BY dimension,group_id
            """,
            run_dir / "downweight_group_distribution_v1.parquet",
        )
        _write_query(
            connection,
            """
            WITH moments AS (
              SELECT exposure_date,liquidity_decile,
                     avg(specific_return) AS mean_u,
                     var_pop(specific_return) AS variance_u,
                     avg(is_outlier_flagged::INTEGER) AS downweight_rate
              FROM ranked GROUP BY exposure_date,liquidity_decile
            ), daily AS (
              SELECT r.exposure_date,r.liquidity_decile,m.variance_u,m.downweight_rate,
                     CASE WHEN m.variance_u > 0 THEN
                       avg(pow(r.specific_return-m.mean_u,4))/pow(m.variance_u,2)-3
                     END AS excess_kurtosis_u
              FROM ranked r JOIN moments m USING(exposure_date,liquidity_decile)
              GROUP BY r.exposure_date,r.liquidity_decile,m.mean_u,
                       m.variance_u,m.downweight_rate
            )
            SELECT liquidity_decile,sum(1) AS n_days,
                   median(variance_u) AS median_daily_variance_u,
                   median(excess_kurtosis_u) AS median_daily_excess_kurtosis_u,
                   median(downweight_rate) AS median_daily_downweight_rate
            FROM daily GROUP BY liquidity_decile ORDER BY liquidity_decile
            """,
            run_dir / "liquidity_residual_shape_v1.parquet",
        )
        _write_query(
            connection,
            """
            WITH daily AS (
              SELECT trade_date,avg(is_outlier_flagged::INTEGER) AS downweight_rate,
                     corr(risk_liquidity,risk_residual_volatility)
                       AS liquidity_residual_volatility_corr
              FROM ranked GROUP BY trade_date
            )
            SELECT d.trade_date,d.downweight_rate,
                   d.liquidity_residual_volatility_corr,
                   q.full_label_r_squared,q.full_label_unweighted_r_squared,
                   q.huber_downweight_rate
            FROM daily d JOIN quality q USING(trade_date)
            ORDER BY d.trade_date
            """,
            run_dir / "daily_quality_join_v1.parquet",
        )
        _write_query(
            connection,
            """
            SELECT is_outlier_flagged,count(*) AS n_observations,
                   avg(abs(specific_return)) AS mean_abs_u,
                   quantile_cont(abs(specific_return),0.5) AS p50_abs_u,
                   quantile_cont(abs(specific_return),0.9) AS p90_abs_u,
                   quantile_cont(abs(specific_return),0.99) AS p99_abs_u
            FROM ranked GROUP BY is_outlier_flagged ORDER BY is_outlier_flagged
            """,
            run_dir / "residual_tail_comparison_v1.parquet",
        )
        headline = connection.execute(
            """
            WITH daily AS (
              SELECT trade_date,avg(is_outlier_flagged::INTEGER) AS downweight_rate,
                     corr(risk_liquidity,risk_residual_volatility)
                       AS liquidity_residual_volatility_corr
              FROM ranked GROUP BY trade_date
            )
            SELECT count(*) AS n_days,
                   avg(d.downweight_rate) AS mean_downweight_rate,
                   corr(d.downweight_rate,q.full_label_r_squared)
                     AS downweight_vs_r2_corr,
                   avg(d.liquidity_residual_volatility_corr)
                     AS mean_liquidity_residual_volatility_corr
            FROM daily d JOIN quality q USING(trade_date)
            """
        ).fetchone()
    finally:
        connection.close()

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": run_identity["job"],
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "reestimated": False,
        "current_pointer_mutated": False,
        "identity": run_identity,
        "headline": {
            "n_days": int(headline[0]),
            "mean_downweight_rate": float(headline[1]),
            "downweight_vs_r2_corr": float(headline[2]),
            "mean_liquidity_residual_volatility_corr": float(headline[3]),
        },
        "outputs": {
            name: {"path": str(path.relative_to(DATA)), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in {
                "downweight_group_distribution_v1": run_dir / "downweight_group_distribution_v1.parquet",
                "liquidity_residual_shape_v1": run_dir / "liquidity_residual_shape_v1.parquet",
                "daily_quality_join_v1": run_dir / "daily_quality_join_v1.parquet",
                "residual_tail_comparison_v1": run_dir / "residual_tail_comparison_v1.parquet",
            }.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
