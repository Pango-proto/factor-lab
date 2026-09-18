"""Recover missing canonical market files from immutable Bronze responses.

This is an emergency utility, not a second ingestion implementation. It calls
the canonical normalizers and refuses to replace a non-empty destination.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from factor_matrix.normalize import canonicalize_bse_codes, normalize_market, recompute_price_series
from factor_matrix.storage import DataLake, file_sha256


def _latest_daily_files(lake: DataLake, api: str) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in lake.bronze_objects(api):
        response = lake.load_bronze(path)
        fields = response.fields
        items = response.items
        if not items or "trade_date" not in fields:
            continue
        index = fields.index("trade_date")
        dates = {str(row[index]) for row in items if row[index]}
        if len(dates) != 1:
            raise RuntimeError(f"BRONZE_RECOVERY_EXPECTED_ONE_DATE {path} {dates}")
        selected[dates.pop()] = path
    return selected


def _frame(lake: DataLake, path: Path) -> pl.DataFrame:
    response = lake.load_bronze(path)
    return pl.DataFrame(
        response.items,
        schema=response.fields,
        orient="row",
        infer_schema_length=None,
    )


def recover(root: Path, install: bool) -> dict[str, object]:
    lake = DataLake(root)
    selected = {
        api: _latest_daily_files(lake, api)
        for api in ("daily", "adj_factor", "daily_basic")
    }
    dates = sorted(set(selected["daily"]) & set(selected["adj_factor"]) & set(selected["daily_basic"]))
    if not dates:
        raise RuntimeError("BRONZE_RECOVERY_NO_COMMON_DATES")
    mapping_path = lake.silver / "bse_code_mapping" / "data.parquet"
    mapping = pl.read_parquet(mapping_path).select(
        pl.col("old_asset_id").alias("o_code"), pl.col("asset_id").alias("n_code")
    )
    limit_path = lake.silver / "price_limits_daily" / "data.parquet"
    limits = pl.read_parquet(limit_path) if limit_path.exists() else pl.DataFrame()
    recovered_at = datetime.now(UTC)
    price_parts: list[pl.DataFrame] = []
    valuation_parts: list[pl.DataFrame] = []
    for offset in range(0, len(dates), 40):
        batch = dates[offset : offset + 40]
        daily = canonicalize_bse_codes(
            pl.concat([_frame(lake, selected["daily"][day]) for day in batch], how="diagonal_relaxed"), mapping
        )
        adj = canonicalize_bse_codes(
            pl.concat([_frame(lake, selected["adj_factor"][day]) for day in batch], how="diagonal_relaxed"), mapping
        )
        valuation = canonicalize_bse_codes(
            pl.concat([_frame(lake, selected["daily_basic"][day]) for day in batch], how="diagonal_relaxed"), mapping
        )
        prices, valuations = normalize_market(
            daily, adj, valuation, recovered_at, dates[-1], None
        )
        if not limits.is_empty():
            prices = (
                prices.drop("limit_up", "limit_down")
                .join(
                    limits.select("trade_date", "asset_id", "limit_up", "limit_down"),
                    on=["trade_date", "asset_id"],
                    how="left",
                )
            )
        price_parts.append(prices)
        valuation_parts.append(valuations)
    calendar = pl.read_parquet(lake.silver / "trade_calendar" / "data.parquet")
    open_dates = calendar.filter(pl.col("is_open") == 1).get_column("cal_date")
    prices = recompute_price_series(
        pl.concat(price_parts, how="diagonal_relaxed", rechunk=False)
        .unique(["trade_date", "asset_id"], keep="last"),
        open_dates,
    ).sort(["trade_date", "asset_id"])
    valuations = (
        pl.concat(valuation_parts, how="diagonal_relaxed", rechunk=False)
        .unique(["trade_date", "asset_id"], keep="last")
        .sort(["trade_date", "asset_id"])
    )
    recovery_root = lake.metadata / "recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    price_candidate = recovery_root / "prices_daily.parquet"
    valuation_candidate = recovery_root / "valuation_daily.parquet"
    prices.write_parquet(price_candidate, compression="zstd", statistics=True)
    valuations.write_parquet(valuation_candidate, compression="zstd", statistics=True)
    checks = {
        "common_dates": len(dates),
        "min_date": str(prices.get_column("trade_date").min()),
        "max_date": str(prices.get_column("trade_date").max()),
        "price_rows": prices.height,
        "valuation_rows": valuations.height,
        "price_duplicates": prices.height - prices.unique(["trade_date", "asset_id"]).height,
        "valuation_duplicates": valuations.height - valuations.unique(["trade_date", "asset_id"]).height,
        "missing_adj_factor": prices.filter(pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)).height,
        "missing_raw_ohlc": prices.filter(
            pl.any_horizontal([pl.col(name).is_null() for name in ("raw_open", "raw_high", "raw_low", "raw_close")])
        ).height,
        "price_sha256": file_sha256(price_candidate),
        "valuation_sha256": file_sha256(valuation_candidate),
    }
    if any(checks[key] for key in ("price_duplicates", "valuation_duplicates", "missing_adj_factor", "missing_raw_ohlc")):
        raise RuntimeError(f"BRONZE_RECOVERY_QUALITY_FAILED {checks}")
    if install:
        for table, candidate in (
            ("prices_daily", price_candidate), ("valuation_daily", valuation_candidate)
        ):
            destination = lake.silver / table / "data.parquet"
            if destination.exists() and destination.stat().st_size:
                raise RuntimeError(f"BRONZE_RECOVERY_REFUSES_NONEMPTY_DESTINATION {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate, destination)
        lake.refresh_catalog()
        checks["installed"] = True
    else:
        checks["installed"] = False
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.data_root, args.install), indent=2))
