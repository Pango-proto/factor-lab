"""Label-free listing-age volatility decay and market-dispersion diagnostics."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import polars as pl

from ..external_facts import ExternalFacts
from ..revisioned_silver import SilverAsOfReader, SilverVersionLedger
from ..storage import DataLake, json_hash, open_duckdb, source_tree_hash


DEFAULT_CONFIG = Path("config/new_listing_window_derivation_v1.json")
DEFAULT_FACTS = Path("config/external_facts_v1.json")


def _fetch_frame(connection, sql: str) -> pl.DataFrame:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return pl.DataFrame(cursor.fetchall(), schema=names, orient="row", infer_schema_length=None)


def _fact_date(facts: ExternalFacts, fact_key: str, scope: str) -> date:
    matches = [
        item.effective_from for item in facts.facts
        if item.fact_key == fact_key and item.scope == scope
    ]
    if len(matches) != 1:
        raise ValueError(f"LISTING_REGIME_FACT_NOT_UNIQUE key={fact_key} scope={scope}")
    return matches[0]


def _regimes(config: dict[str, Any], facts: ExternalFacts) -> dict[str, tuple[date, ...]]:
    estimation_start = date.fromisoformat(config["estimation_start"])
    result: dict[str, tuple[date, ...]] = {}
    for board, rule in config["boards"].items():
        sample = rule["sample_start"]
        if sample == "estimation_start":
            starts = [estimation_start]
        else:
            _, fact_key, scope = sample.split(":", 2)
            starts = [_fact_date(facts, fact_key, scope)]
        split_scope = rule.get("split_regime_break_scope")
        if split_scope:
            split = _fact_date(facts, "regime_break", split_scope)
            if split > starts[0]:
                starts.append(split)
        result[board] = tuple(sorted(set(starts)))
    return result


def _regime_case_sql(regimes: dict[str, tuple[date, ...]]) -> str:
    clauses: list[str] = []
    for board, starts in regimes.items():
        for start in reversed(starts):
            label = f"{board}_{start.isoformat()}"
            clauses.append(
                f"WHEN board_id='{board}' AND exchange_list_date>=DATE '{start.isoformat()}' "
                f"THEN '{label}'"
            )
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def _derive_group_result(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: item["listing_age_days"])
    steady_low, steady_high = config["steady_state_age_range"]
    minimum_stocks = int(config["minimum_stocks_per_age"])
    baseline = [
        float(item["sigma_adjusted"])
        for item in ordered
        if steady_low <= item["listing_age_days"] <= steady_high
        and item["n_stocks_used"] >= minimum_stocks
        and item["sigma_adjusted"] is not None
    ]
    sigma_infinity = mean(baseline) if baseline else None
    baseline_sufficient = len(baseline) >= int(config["minimum_steady_state_age_points"])
    smoothing = int(config["smoothing_window_days"])
    enriched: list[dict[str, Any]] = []
    history: list[float] = []
    for item in ordered:
        value = item["sigma_adjusted"]
        if value is not None:
            history.append(float(value))
        smooth = median(history[-smoothing:]) if history else None
        enriched.append({**item, "sigma_adjusted_smoothed": smooth})

    tolerance = float(config["steady_state_tolerance"])
    limit_max = float(config["maximum_one_price_limit_ratio"])
    persistence = int(config["required_persistence_days"])
    d_star = None
    if sigma_infinity is not None and baseline_sufficient:
        valid = []
        for item in enriched:
            valid.append(bool(
                item["sigma_adjusted_smoothed"] is not None
                and item["sigma_adjusted_smoothed"] <= (1 + tolerance) * sigma_infinity
                and item["one_price_limit_ratio"] <= limit_max
                and item["n_stocks_used"] >= minimum_stocks
            ))
        for index, item in enumerate(enriched):
            if all(valid[index:index + persistence]) and len(valid[index:index + persistence]) == persistence:
                consecutive = [
                    row["listing_age_days"] for row in enriched[index:index + persistence]
                ]
                if consecutive == list(range(consecutive[0], consecutive[0] + persistence)):
                    d_star = int(item["listing_age_days"])
                    break

    stable = d_star is not None
    selected = next(
        (item for item in enriched if item["listing_age_days"] == d_star), None
    )
    summary = {
        "regime_id": ordered[0]["regime_id"],
        "board_id": ordered[0]["board_id"],
        "regime_start": date.fromisoformat(ordered[0]["regime_id"].rsplit("_", 1)[1]),
        "d_star": d_star,
        "sigma_infinity": sigma_infinity,
        "steady_state_age_points": len(baseline),
        "n_stocks_used": selected["n_stocks_used"] if selected else 0,
        "one_price_limit_ratio_at_d_star": (
            selected["one_price_limit_ratio"] if selected else None
        ),
        "stable": stable,
        "fallback_days": None if stable else int(config["fallback_days"]),
        "reason": (
            None if stable else
            "insufficient steady-state age coverage" if not baseline_sufficient else
            "insufficient stable persistent crossing"
        ),
    }
    return summary, enriched


def run_new_listing_window_diagnostics(
    lake: DataLake,
    config_path: Path = DEFAULT_CONFIG,
    facts_path: Path = DEFAULT_FACTS,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("NEW_LISTING_DERIVATION_CONFIG_SCHEMA_UNSUPPORTED")
    facts = ExternalFacts.load(facts_path)
    regimes = _regimes(config, facts)
    version = SilverVersionLedger(lake).current()
    if version is None:
        raise RuntimeError("NEW_LISTING_DERIVATION_REQUIRES_SILVER_VERSION")
    identity = {
        "config_sha": json_hash(config),
        "external_facts_sha": json_hash(json.loads(facts_path.read_text(encoding="utf-8"))),
        "silver_version_id": version["version_id"],
        "code_hash": source_tree_hash(),
    }
    run_id = f"new_listing_window_{json_hash(identity)[:16]}"
    manifest_path = lake.manifests / f"{run_id}.json"
    if manifest_path.exists():
        return manifest_path

    as_of = datetime.fromisoformat(version["observed_at"])
    reader = SilverAsOfReader(lake, version["base_id"])
    state = reader.relation_sql("security_daily_state", as_of)
    returns = reader.relation_sql("returns_daily", as_of)
    prices = reader.relation_sql("prices_daily", as_of)
    rolling = int(config["rolling_window_trading_days"])
    minimum_rolling = int(config["minimum_rolling_observations"])
    maximum_age = int(config["maximum_age_days"])
    regime_case = _regime_case_sql(regimes)
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    try:
        common = f"""
            WITH events AS (
              SELECT s.trade_date,s.asset_id,s.board_id,s.exchange_list_date,
                     s.days_since_exchange_list,r.total_return,r.return_source,
                     CASE WHEN p.limit_up IS NOT NULL AND p.raw_open>=p.limit_up
                                AND p.raw_high>=p.limit_up AND p.raw_low>=p.limit_up
                                AND p.raw_close>=p.limit_up THEN TRUE
                          WHEN p.limit_down IS NOT NULL AND p.raw_open<=p.limit_down
                                AND p.raw_high<=p.limit_down AND p.raw_low<=p.limit_down
                                AND p.raw_close<=p.limit_down THEN TRUE
                          ELSE FALSE END AS is_one_price_limit,
                     CASE WHEN isfinite(r.total_return)
                               AND r.return_source!='resumption'
                               AND s.trade_date!=s.exchange_list_date
                          THEN r.total_return END AS model_return
              FROM ({state}) s
              LEFT JOIN ({returns}) r USING(trade_date,asset_id)
              LEFT JOIN ({prices}) p USING(trade_date,asset_id)
              WHERE s.in_a_share_scope AND s.board_id IN ('MAIN','CHINEXT','STAR','BSE')
            ), market AS (
              SELECT trade_date,stddev_samp(model_return) AS sigma_r,
                     count(model_return) AS n_valid
              FROM events GROUP BY 1
            ), rolled AS (
              SELECT events.*,market.sigma_r,market.n_valid,
                     stddev_samp(model_return) OVER (
                       PARTITION BY asset_id ORDER BY trade_date
                       ROWS BETWEEN {rolling - 1} PRECEDING AND CURRENT ROW
                     ) AS rolling_sigma,
                     count(model_return) OVER (
                       PARTITION BY asset_id ORDER BY trade_date
                       ROWS BETWEEN {rolling - 1} PRECEDING AND CURRENT ROW
                     ) AS rolling_observations,
                     {regime_case} AS regime_id
              FROM events JOIN market USING(trade_date)
            )
        """
        dispersion = _fetch_frame(connection, common + """
            SELECT trade_date,n_valid,sigma_r FROM market ORDER BY trade_date
        """)
        curve = _fetch_frame(connection, common + f"""
            SELECT regime_id,board_id,min(exchange_list_date) AS regime_start,
                   days_since_exchange_list AS listing_age_days,
                   median(rolling_sigma/nullif(sigma_r,0)) FILTER (
                     WHERE rolling_observations>={minimum_rolling}
                       AND rolling_sigma IS NOT NULL AND sigma_r>0
                   ) AS sigma_adjusted,
                   count(DISTINCT asset_id) FILTER (
                     WHERE rolling_observations>={minimum_rolling}
                       AND rolling_sigma IS NOT NULL AND sigma_r>0
                   ) AS n_stocks_used,
                   avg(is_one_price_limit::INTEGER) AS one_price_limit_ratio
            FROM rolled
            WHERE regime_id IS NOT NULL
              AND days_since_exchange_list BETWEEN 1 AND {maximum_age}
            GROUP BY 1,2,4 ORDER BY 1,4
        """)
    finally:
        connection.close()

    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for key, group in curve.partition_by("regime_id", as_dict=True).items():
        records = group.to_dicts()
        if not records:
            continue
        summary, enriched = _derive_group_result(records, config)
        summaries.append(summary)
        curves.extend(enriched)
    summary_frame = pl.DataFrame(summaries)
    curve_frame = pl.DataFrame(curves)
    output_root = lake.root / "diagnostics" / "new_listing_window" / f"run_id={run_id}"
    output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "market_return_dispersion_v1": output_root / "market_return_dispersion_v1.parquet",
        "new_listing_window_v1": output_root / "new_listing_window_v1.parquet",
        "new_listing_curve_v1": output_root / "new_listing_curve_v1.parquet",
    }
    dispersion.write_parquet(paths["market_return_dispersion_v1"])
    summary_frame.write_parquet(paths["new_listing_window_v1"])
    curve_frame.write_parquet(paths["new_listing_curve_v1"])
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "job": "new_listing_window_diagnostics",
        "execution_status": "passed",
        "as_of_timestamp": version["observed_at"],
        "silver_version_id": version["version_id"],
        "identity": identity,
        "config": config,
        "regimes": {
            board: [item.isoformat() for item in starts] for board, starts in regimes.items()
        },
        "formal_exposure_publish_allowed": False,
        "formal_publish_blocker": "D0_MODEL_EXCLUSION_NOT_FROZEN",
        "outputs": {name: lake.artifact_record(path) for name, path in paths.items()},
        "summary": summaries,
    }
    return lake.write_immutable_json(manifest_path, manifest)
