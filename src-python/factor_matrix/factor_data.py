from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .normalize import normalize_calendar, response_frame
from .source import TushareClient, TushareError, TushareResponse
from .storage import DataLake, utc_now
from .revisioned_silver import publish_revisioned_batch, read_as_of_frame

from .factor_normalize import (
    _canonical_code_map,
    BALANCE_FIELDS,
    CALENDAR_FIELDS,
    CASHFLOW_FIELDS,
    DISCLOSURE_FIELDS,
    INCOME_FIELDS,
    SHIBOR_FIELDS,
    normalize_financial_pit,
    normalize_shibor_daily,
    normalize_statement_pit,
)

class FactorDataPipeline:
    def __init__(
        self, client: TushareClient, lake: DataLake, token_fingerprint: str
    ) -> None:
        self.client = client
        self.lake = lake
        self.token_fingerprint = token_fingerprint
        self.sources: list[dict[str, str]] = []

    def _write_checkpoint(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _normalize_and_store_batch(
        self,
        frames: list[pl.DataFrame],
        disclosures: pl.DataFrame,
        calendar: pl.DataFrame,
        started_at: datetime,
        code_mapping: dict[str, str],
        *,
        adjustment_history_complete: bool = False,
    ) -> int:
        if not frames:
            return 0
        try:
            existing = read_as_of_frame(self.lake, "financial_pit", started_at)
        except FileNotFoundError:
            existing = None
        financial = normalize_financial_pit(
            pl.concat(frames, how="diagonal_relaxed"),
            disclosures,
            calendar,
            started_at,
            code_mapping=code_mapping,
            existing=existing,
            adjustment_history_complete=adjustment_history_complete,
        )
        if financial.is_empty():
            return 0
        published_at = utc_now()
        publish_revisioned_batch(
            self.lake, {"financial_pit": financial}, effective_date=started_at.date(),
            observed_at=published_at,
            quality_gate={"status": "passed", "table": "financial_pit", "rows": financial.height},
        )
        return financial.height

    def _normalize_and_store_statement(
        self,
        table: str,
        frames: list[pl.DataFrame],
        calendar: pl.DataFrame,
        started_at: datetime,
        code_mapping: dict[str, str],
        value_fields: tuple[str, ...],
        *,
        adjustment_history_complete: bool = False,
    ) -> int:
        if not frames:
            return 0
        try:
            existing = read_as_of_frame(self.lake, table, started_at)
        except FileNotFoundError:
            existing = None
        normalized = normalize_statement_pit(
            pl.concat(frames, how="diagonal_relaxed"),
            calendar,
            started_at,
            statement=table.removesuffix("_pit"),
            value_fields=value_fields,
            code_mapping=code_mapping,
            existing=existing,
            adjustment_history_complete=adjustment_history_complete,
        )
        if normalized.is_empty():
            return 0
        published_at = utc_now()
        publish_revisioned_batch(
            self.lake, {table: normalized}, effective_date=started_at.date(),
            observed_at=published_at,
            quality_gate={"status": "passed", "table": table, "rows": normalized.height},
        )
        return normalized.height

    def _fetch_financials_per_asset(
        self,
        years: list[int],
        disclosures: pl.DataFrame,
        calendar: pl.DataFrame,
        started_at: datetime,
        *,
        request_delay: float,
        batch_size: int = 50,
    ) -> int:
        prices = read_as_of_frame(self.lake, "prices_daily")
        asset_ids = sorted(prices.get_column("asset_id").unique().to_list())
        code_mapping = _canonical_code_map(self.lake)
        vendor_fallback = {canonical: old for old, canonical in code_mapping.items()}
        checkpoint_path = (
            self.lake.metadata
            / "checkpoints"
            / f"financial_{years[0]}_{years[-1]}.json"
        )
        checkpoint = {"completed_assets": [], "rows": 0}
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed = set(checkpoint.get("completed_assets", []))
        rows_written = int(checkpoint.get("rows", 0))
        pending_frames: list[pl.DataFrame] = []
        pending_assets: list[str] = []
        for index, asset_id in enumerate(asset_ids, start=1):
            if asset_id in completed:
                continue
            vendor_code = vendor_fallback.get(asset_id, asset_id)
            response = self._fetch(
                "balancesheet",
                {
                    "ts_code": vendor_code,
                    "start_date": f"{years[0]}0101",
                    "end_date": f"{years[-1]}1231",
                },
                BALANCE_FIELDS,
            )
            if response.items:
                pending_frames.append(response_frame(response))
            pending_assets.append(asset_id)
            if len(pending_assets) >= batch_size or index == len(asset_ids):
                rows_written += self._normalize_and_store_batch(
                    pending_frames,
                    disclosures,
                    calendar,
                    started_at,
                    code_mapping,
                )
                completed.update(pending_assets)
                self._write_checkpoint(
                    checkpoint_path,
                    {
                        "financial_years": years,
                        "completed_assets": sorted(completed),
                        "total_assets": len(asset_ids),
                        "rows": rows_written,
                        "updated_at": utc_now().isoformat(),
                    },
                )
                print(
                    f"Financial PIT progress {len(completed)}/{len(asset_ids)}; rows={rows_written}",
                    flush=True,
                )
                pending_frames = []
                pending_assets = []
            if request_delay:
                time.sleep(request_delay)
        return rows_written

    def _fetch(
        self, api_name: str, params: dict[str, Any], fields: list[str]
    ) -> TushareResponse:
        response = self.client.query(api_name, params, fields)
        self.sources.append(self.lake.save_bronze(response, utc_now()))
        missing = set(fields) - set(response.fields)
        if missing:
            raise RuntimeError(f"SOURCE_SCHEMA_DRIFT {api_name}: {sorted(missing)}")
        if len(response.items) >= 10_000:
            raise RuntimeError(f"SOURCE_POSSIBLY_TRUNCATED {api_name}: rows={len(response.items)}")
        return response

    def sync(
        self,
        financial_years: Iterable[int],
        risk_start: date,
        risk_end: date,
        *,
        request_delay: float = 0.15,
    ) -> Path:
        started_at = utc_now()
        years = sorted(set(financial_years))
        if not years:
            raise ValueError("At least one financial year is required")
        calendar_response = self._fetch(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": f"{years[0]}0101",
                "end_date": f"{risk_end.year}1231",
            },
            CALENDAR_FIELDS,
        )
        calendar_update = normalize_calendar(response_frame(calendar_response), started_at)
        publish_revisioned_batch(
            self.lake, {"trade_calendar": calendar_update},
            effective_date=risk_end, observed_at=utc_now(),
            quality_gate={
                "status": "passed", "table": "trade_calendar", "rows": calendar_update.height,
            },
        )
        disclosure_frames = []
        for year in years:
            period = f"{year}1231"
            disclosure_frames.append(
                response_frame(
                    self._fetch("disclosure_date", {"end_date": period}, DISCLOSURE_FIELDS)
                )
            )
        disclosures = pl.concat(disclosure_frames, how="diagonal_relaxed")
        calendar = read_as_of_frame(self.lake, "trade_calendar")

        periods = [
            f"{year}{month_day}"
            for year in years
            for month_day in ("0331", "0630", "0930", "1231")
        ]
        balance_frames = []
        financial_mode = "balancesheet_vip_by_period_with_adjustment_history"
        try:
            for period in periods:
                for report_type in (None, "5"):
                    params = {"period": period}
                    if report_type is not None:
                        params["report_type"] = report_type
                    balance_frames.append(
                        response_frame(
                            self._fetch(
                                "balancesheet_vip",
                                params,
                                BALANCE_FIELDS,
                            )
                        )
                    )
        except (TushareError, RuntimeError) as exc:
            financial_mode = "balancesheet_per_asset_resumable"
            print(
                f"VIP batch unavailable; fallback to resumable per-asset mode: {exc}",
                flush=True,
            )
            balance_frames = []

        code_mapping = _canonical_code_map(self.lake)
        if balance_frames:
            financial_rows = self._normalize_and_store_batch(
                balance_frames,
                disclosures,
                calendar,
                started_at,
                code_mapping,
                adjustment_history_complete=True,
            )
        else:
            financial_rows = self._fetch_financials_per_asset(
                years,
                disclosures,
                calendar,
                started_at,
                request_delay=request_delay,
            )

        income_frames = []
        cashflow_frames = []
        for period in periods:
            for report_type in (None, "5"):
                params = {"period": period}
                if report_type is not None:
                    params["report_type"] = report_type
                income_frames.append(
                    response_frame(self._fetch("income_vip", params, INCOME_FIELDS))
                )
                cashflow_frames.append(
                    response_frame(self._fetch("cashflow_vip", params, CASHFLOW_FIELDS))
                )
        income_rows = self._normalize_and_store_statement(
            "income_pit",
            income_frames,
            calendar,
            started_at,
            code_mapping,
            ("n_income_attr_p", "revenue", "total_revenue"),
            adjustment_history_complete=True,
        )
        cashflow_rows = self._normalize_and_store_statement(
            "cashflow_pit",
            cashflow_frames,
            calendar,
            started_at,
            code_mapping,
            ("n_cashflow_act", "free_cashflow"),
            adjustment_history_complete=True,
        )

        shibor = response_frame(
            self._fetch(
                "shibor",
                {
                    "start_date": risk_start.strftime("%Y%m%d"),
                    "end_date": risk_end.strftime("%Y%m%d"),
                },
                SHIBOR_FIELDS,
            )
        )
        try:
            financial_current = read_as_of_frame(self.lake, "financial_pit")
        except FileNotFoundError:
            financial_current = pl.DataFrame()
        if financial_current.is_empty():
            raise RuntimeError("SOURCE_EMPTY_REQUIRED_DATA financial_pit")

        risk_free = normalize_shibor_daily(shibor, calendar, started_at)
        risk_free = risk_free.filter(
            (pl.col("trade_date") >= risk_start) & (pl.col("trade_date") <= risk_end)
        )
        if risk_free.is_empty():
            raise RuntimeError("SOURCE_EMPTY_REQUIRED_DATA risk_free_daily")
        publish_revisioned_batch(
            self.lake, {"risk_free_daily": risk_free},
            effective_date=risk_end, observed_at=utc_now(),
            quality_gate={"status": "passed", "table": "risk_free_daily", "rows": risk_free.height},
        )
        self.lake.refresh_catalog()

        config = {
            "financial_years": years,
            "risk_start": risk_start.isoformat(),
            "risk_end": risk_end.isoformat(),
            "financial_revision_policy": "original/unchanged latest at announcement; adjusted latest first-seen delayed; report_type=5 preserves pre-adjustment history",
            "financial_availability_lag": "next open trading day",
            "financial_fetch_mode": financial_mode,
            "financial_periods": periods,
            "value_inputs": "PIT net income, operating cash flow, debt and cash; TTM derived downstream",
            "adjustment_history": "report_type=5 adjusted-before rows synced for every requested period",
            "risk_free": "Shibor 3M, strictly prior fixing, ACT/365 compound conversion",
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        run_id = f"factor_data_{years[0]}_{years[-1]}_{config_hash[:8]}"
        manifest = {
            "run_id": run_id,
            "job": "factor_data_sync",
            "started_at": started_at.isoformat(),
            "completed_at": utc_now().isoformat(),
            "source_id": "tushare",
            "source_credential_fingerprint": self.token_fingerprint,
            "config": config,
            "config_hash": config_hash,
            "bronze_objects": self.sources,
            "row_counts": {
                "financial_pit_fetched": financial_rows,
                "income_pit_fetched": income_rows,
                "cashflow_pit_fetched": cashflow_rows,
                "risk_free_daily_fetched": risk_free.height,
            },
        }
        return self.lake.write_manifest(run_id, manifest)

    def sync_risk_free(self, risk_start: date, risk_end: date) -> Path:
        """Increment only the strictly lagged Shibor series used by daily FF3."""
        if risk_end < risk_start:
            raise ValueError("risk end must not be before risk start")
        started_at = utc_now()
        try:
            calendar = read_as_of_frame(self.lake, "trade_calendar")
        except FileNotFoundError as exc:
            raise RuntimeError("RISK_FREE_CALENDAR_MISSING") from exc
        query_start = risk_start - timedelta(days=10)
        config = {
            "risk_start": risk_start.isoformat(),
            "risk_end": risk_end.isoformat(),
            "query_start": query_start.isoformat(),
            "risk_free": "Shibor 3M, strictly prior fixing, ACT/365 compound conversion",
            "storage_contract": "revisioned_silver_v1",
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"risk_free_daily_{risk_start:%Y%m%d}_{risk_end:%Y%m%d}_{config_hash}"
        manifest_path = self.lake.manifests / f"{run_id}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("quality_gate", {}).get("status") == "passed" and existing.get(
                "silver_version_id"
            ):
                return manifest_path
        shibor = response_frame(
            self._fetch(
                "shibor",
                {
                    "start_date": query_start.strftime("%Y%m%d"),
                    "end_date": risk_end.strftime("%Y%m%d"),
                },
                SHIBOR_FIELDS,
            )
        )
        risk_free = normalize_shibor_daily(shibor, calendar, started_at).filter(
            (pl.col("trade_date") >= risk_start) & (pl.col("trade_date") <= risk_end)
        )
        expected = calendar.filter(
            (pl.col("is_open") == 1)
            & pl.col("cal_date").is_between(risk_start, risk_end, closed="both")
        ).get_column("cal_date").n_unique()
        if risk_free.height != expected or risk_free.filter(
            pl.col("rate_date") >= pl.col("trade_date")
        ).height:
            raise RuntimeError(
                f"RISK_FREE_DAILY_GATE_FAILED expected={expected} actual={risk_free.height}"
            )
        quality = {"status": "passed", "expected_rows": expected, "rows": risk_free.height}
        changes, version = publish_revisioned_batch(
            self.lake, {"risk_free_daily": risk_free}, effective_date=risk_end,
            observed_at=started_at, quality_gate=quality,
        )
        self.lake.refresh_catalog()
        return self.lake.write_manifest(
            run_id,
            {
                "run_id": run_id,
                "job": "risk_free_daily_sync",
                "created_at": started_at.isoformat(),
                "source_credential_fingerprint": self.token_fingerprint,
                "config": config,
                "row_count": risk_free.height,
                "bronze_objects": self.sources,
                "changes": changes,
                "silver_version_id": version["version_id"],
                "quality_gate": quality,
            },
        )
