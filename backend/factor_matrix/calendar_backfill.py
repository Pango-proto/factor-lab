"""Small append-only Tushare trade-calendar backfill for listing-age observability."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .normalize import normalize_calendar, response_frame
from .revisioned_silver import publish_revisioned_batch
from .source import TushareClient
from .storage import DataLake, json_hash, utc_now


CALENDAR_FIELDS = ["exchange", "cal_date", "is_open", "pretrade_date"]


def backfill_trade_calendar(
    client: TushareClient,
    lake: DataLake,
    *,
    start: date,
    end: date,
    credential_fingerprint: str,
) -> Path:
    if end < start:
        raise ValueError("TRADE_CALENDAR_BACKFILL_RANGE_INVALID")
    config = {
        "start": start.isoformat(), "end": end.isoformat(), "exchange": "SSE",
        "purpose": "pre_estimation_listing_trading_age_lower_bound",
    }
    run_id = f"trade_calendar_backfill_{end:%Y%m%d}_{json_hash(config)[:12]}"
    manifest_path = lake.manifests / f"{run_id}.json"
    if manifest_path.exists():
        return manifest_path
    observed_at = utc_now()
    response = client.query(
        "trade_cal",
        {
            "exchange": "SSE", "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        },
        CALENDAR_FIELDS,
    )
    bronze = lake.save_bronze(response, observed_at)
    frame = response_frame(response)
    if frame.is_empty() or set(CALENDAR_FIELDS) - set(frame.columns):
        raise RuntimeError("TRADE_CALENDAR_BACKFILL_SOURCE_INCOMPLETE")
    calendar = normalize_calendar(frame, observed_at)
    observed_dates = set(calendar["cal_date"].to_list())
    expected_days = (end - start).days + 1
    quality = {
        "status": "passed" if len(observed_dates) == expected_days else "failed",
        "rows": calendar.height,
        "open_days": int(calendar["is_open"].sum()),
        "expected_calendar_days": expected_days,
        "minimum_date": str(calendar["cal_date"].min()),
        "maximum_date": str(calendar["cal_date"].max()),
    }
    if quality["status"] != "passed":
        raise RuntimeError(f"TRADE_CALENDAR_BACKFILL_QUALITY_FAILED {quality}")
    changes, version = publish_revisioned_batch(
        lake, {"trade_calendar": calendar}, effective_date=end,
        observed_at=observed_at, quality_gate=quality,
    )
    return lake.write_immutable_json(manifest_path, {
        "schema_version": 1, "run_id": run_id, "job": "trade_calendar_backfill",
        "status": "passed", "created_at": observed_at.isoformat(),
        "config": config, "config_hash": json_hash(config),
        "source_credential_fingerprint": credential_fingerprint,
        "bronze_object": bronze, "quality_gate": quality, "changes": changes,
        "silver_version_id": version["version_id"],
    })
