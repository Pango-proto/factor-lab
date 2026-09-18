from __future__ import annotations

import bisect
import hashlib
import json
from datetime import UTC, date, datetime
from typing import Any

import polars as pl

from .storage import DataLake

BALANCE_FIELDS = [
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
    "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int",
    "oth_eqt_tools_p_shr",
    "defer_tax_liab",
    "total_assets",
    "money_cap",
    "st_borr",
    "lt_borr",
    "bond_payable",
    "non_cur_liab_due_1y",
    "update_flag",
]
INCOME_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "end_type", "n_income_attr_p", "revenue", "total_revenue", "update_flag",
]
CASHFLOW_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "end_type", "n_cashflow_act", "free_cashflow", "update_flag",
]

DISCLOSURE_FIELDS = ["ts_code", "ann_date", "end_date", "pre_date", "actual_date"]
SHIBOR_FIELDS = ["date", "on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"]
CALENDAR_FIELDS = ["exchange", "cal_date", "is_open", "pretrade_date"]


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    return datetime.strptime(str(value), "%Y%m%d").date()


def _canonical_code_map(lake: DataLake) -> dict[str, str]:
    path = lake.silver / "bse_code_mapping" / "data.parquet"
    if not path.exists():
        return {}
    mapping = pl.read_parquet(path)
    return dict(zip(mapping.get_column("old_asset_id"), mapping.get_column("asset_id")))


def _utc_observation(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("FINANCIAL_OBSERVATION_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)


def _revision_signature(row: dict[str, Any], asset_id: str) -> str:
    payload = {
        "asset_id": asset_id,
        "report_period": row.get("end_date"),
        "report_type": row.get("report_type"),
        "comp_type": row.get("comp_type"),
        "equity_parent": row.get("total_hldr_eqy_exc_min_int"),
        "total_equity": row.get("total_hldr_eqy_inc_min_int"),
        "preferred_equity": row.get("oth_eqt_tools_p_shr"),
        "deferred_tax": row.get("defer_tax_liab"),
        "total_assets": row.get("total_assets"),
        "cash_and_equivalents": row.get("money_cap"),
        "short_term_borrowings": row.get("st_borr"),
        "long_term_borrowings": row.get("lt_borr"),
        "bonds_payable": row.get("bond_payable"),
        "current_portion_long_term_debt": row.get("non_cur_liab_due_1y"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _next_open_date(open_dates: list[date], knowledge_date: date | None) -> date | None:
    if knowledge_date is None:
        return None
    position = bisect.bisect_right(open_dates, knowledge_date)
    return open_dates[position] if position < len(open_dates) else None


def normalize_financial_pit(
    balances: pl.DataFrame,
    disclosures: pl.DataFrame,
    calendar: pl.DataFrame,
    ingested_at: datetime,
    *,
    code_mapping: dict[str, str] | None = None,
    existing: pl.DataFrame | None = None,
    adjustment_history_complete: bool = False,
) -> pl.DataFrame:
    """Build conservative, revision-preserving point-in-time financial records.

    An original vendor row can become available on the first open day after its
    confirmed announcement. A latest/update-only row with no matching original
    value is delayed until the first open day after this project first observed
    it, preventing a current revised value from leaking into old snapshots.
    """
    if balances.is_empty():
        return pl.DataFrame()
    code_mapping = code_mapping or {}
    open_dates = (
        calendar.filter(pl.col("is_open") == 1)
        .get_column("cal_date")
        .unique()
        .sort()
        .to_list()
    )
    disclosure_lookup: dict[tuple[str, date], date] = {}
    if not disclosures.is_empty():
        for row in disclosures.to_dicts():
            asset_id = code_mapping.get(row["ts_code"], row["ts_code"])
            period = _parse_date(row.get("end_date"))
            actual = _parse_date(row.get("actual_date"))
            if period and actual:
                key = (asset_id, period)
                disclosure_lookup[key] = max(actual, disclosure_lookup.get(key, actual))

    rows = []
    signatures_with_original: set[str] = set()
    adjusted_before_keys: set[tuple[str, date | None, str | None]] = set()
    prepared: list[tuple[dict[str, Any], str, str]] = []
    for row in balances.to_dicts():
        asset_id = code_mapping.get(row["ts_code"], row["ts_code"])
        signature = _revision_signature(row, asset_id)
        prepared.append((row, asset_id, signature))
        if str(row.get("update_flag") or "") == "0":
            signatures_with_original.add(signature)
        if str(row.get("report_type") or "") == "5":
            adjusted_before_keys.add(
                (asset_id, _parse_date(row.get("end_date")), row.get("comp_type"))
            )

    existing_first_seen: dict[str, datetime] = {}
    if existing is not None and not existing.is_empty():
        existing_first_seen = dict(
            zip(existing.get_column("revision_id"), existing.get_column("first_seen_at"))
        )

    for row, asset_id, signature in prepared:
        revision_id = f"fin_{signature[:20]}"
        first_seen_at = _utc_observation(existing_first_seen.get(revision_id) or ingested_at)
        report_period = _parse_date(row.get("end_date"))
        stated_dates = [
            _parse_date(row.get("ann_date")),
            _parse_date(row.get("f_ann_date")),
            disclosure_lookup.get((asset_id, report_period)) if report_period else None,
        ]
        announcement_date = max((value for value in stated_dates if value), default=None)
        statement_key = (asset_id, report_period, row.get("comp_type"))
        unchanged_latest = (
            adjustment_history_complete
            and str(row.get("report_type") or "") == "1"
            and statement_key not in adjusted_before_keys
        )
        is_original_value = signature in signatures_with_original or unchanged_latest
        if is_original_value:
            availability_basis = announcement_date
            revision_policy = (
                "confirmed_unchanged_latest_no_adjustment_record"
                if unchanged_latest and signature not in signatures_with_original
                else "confirmed_original_announcement"
            )
        else:
            availability_basis = max(
                (value for value in (announcement_date, first_seen_at.date()) if value),
                default=None,
            )
            revision_policy = "conservative_revision_first_seen"
        available_at = _next_open_date(open_dates, availability_basis)
        equity_parent = row.get("total_hldr_eqy_exc_min_int")
        preferred_equity = row.get("oth_eqt_tools_p_shr") or 0.0
        deferred_tax = row.get("defer_tax_liab") or 0.0
        book_equity = (
            float(equity_parent) - float(preferred_equity) + float(deferred_tax)
            if equity_parent is not None
            else None
        )
        rows.append(
            {
                "asset_id": asset_id,
                "report_period": report_period,
                "report_type": row.get("report_type"),
                "comp_type": row.get("comp_type"),
                "announcement_date": announcement_date,
                "available_at": available_at,
                "revision_id": revision_id,
                "vendor_update_flag": row.get("update_flag"),
                "revision_policy": revision_policy,
                "first_seen_at": first_seen_at,
                "total_equity": row.get("total_hldr_eqy_inc_min_int"),
                "equity_parent": equity_parent,
                "preferred_equity": preferred_equity,
                "deferred_tax": deferred_tax,
                "book_equity": book_equity,
                "total_assets": row.get("total_assets"),
                "cash_and_equivalents": row.get("money_cap"),
                "short_term_borrowings": row.get("st_borr"),
                "long_term_borrowings": row.get("lt_borr"),
                "bonds_payable": row.get("bond_payable"),
                "current_portion_long_term_debt": row.get("non_cur_liab_due_1y"),
                "interest_bearing_debt": sum(
                    float(row.get(field) or 0.0)
                    for field in ("st_borr", "lt_borr", "bond_payable", "non_cur_liab_due_1y")
                ),
                "source_id": "tushare",
                "ingested_at": ingested_at,
            }
        )
    return pl.DataFrame(rows).unique("revision_id", keep="last")


def normalize_statement_pit(
    frame: pl.DataFrame,
    calendar: pl.DataFrame,
    ingested_at: datetime,
    *,
    statement: str,
    value_fields: tuple[str, ...],
    code_mapping: dict[str, str] | None = None,
    existing: pl.DataFrame | None = None,
    adjustment_history_complete: bool = False,
) -> pl.DataFrame:
    """Normalize income/cash-flow revisions with the same conservative PIT clock."""
    if frame.is_empty():
        return pl.DataFrame()
    code_mapping = code_mapping or {}
    open_dates = (
        calendar.filter(pl.col("is_open") == 1)
        .get_column("cal_date").unique().sort().to_list()
    )
    existing_first_seen = (
        dict(zip(existing.get_column("revision_id"), existing.get_column("first_seen_at")))
        if existing is not None and not existing.is_empty()
        else {}
    )
    prepared: list[tuple[dict[str, Any], str, str]] = []
    original_signatures: set[str] = set()
    adjusted_before_keys: set[tuple[str, date | None, str | None]] = set()
    for row in frame.to_dicts():
        asset_id = code_mapping.get(row["ts_code"], row["ts_code"])
        signature_payload = {
            "statement": statement,
            "asset_id": asset_id,
            "report_period": row.get("end_date"),
            "report_type": row.get("report_type"),
            "comp_type": row.get("comp_type"),
            **{field: row.get(field) for field in value_fields},
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        prepared.append((row, asset_id, signature))
        if str(row.get("update_flag") or "") == "0":
            original_signatures.add(signature)
        if str(row.get("report_type") or "") == "5":
            adjusted_before_keys.add(
                (asset_id, _parse_date(row.get("end_date")), row.get("comp_type"))
            )
    rows: list[dict[str, Any]] = []
    for row, asset_id, signature in prepared:
        revision_id = f"{statement}_{signature[:20]}"
        first_seen_at = _utc_observation(existing_first_seen.get(revision_id) or ingested_at)
        announcement_date = max(
            (value for value in (_parse_date(row.get("ann_date")), _parse_date(row.get("f_ann_date"))) if value),
            default=None,
        )
        statement_key = (
            asset_id,
            _parse_date(row.get("end_date")),
            row.get("comp_type"),
        )
        unchanged_latest = (
            adjustment_history_complete
            and str(row.get("report_type") or "") == "1"
            and statement_key not in adjusted_before_keys
        )
        if signature in original_signatures or unchanged_latest:
            availability_basis = announcement_date
            revision_policy = (
                "confirmed_unchanged_latest_no_adjustment_record"
                if unchanged_latest and signature not in original_signatures
                else "confirmed_original_announcement"
            )
        else:
            availability_basis = max(
                (value for value in (announcement_date, first_seen_at.date()) if value),
                default=None,
            )
            revision_policy = "conservative_revision_first_seen"
        rows.append(
            {
                "asset_id": asset_id,
                "report_period": _parse_date(row.get("end_date")),
                "report_type": row.get("report_type"),
                "comp_type": row.get("comp_type"),
                "announcement_date": announcement_date,
                "available_at": _next_open_date(open_dates, availability_basis),
                "revision_id": revision_id,
                "vendor_update_flag": row.get("update_flag"),
                "revision_policy": revision_policy,
                "first_seen_at": first_seen_at,
                **{field: row.get(field) for field in value_fields},
                "source_id": "tushare",
                "ingested_at": ingested_at,
            }
        )
    return pl.DataFrame(rows).unique("revision_id", keep="last")


def normalize_shibor_daily(
    shibor: pl.DataFrame,
    calendar: pl.DataFrame,
    ingested_at: datetime,
    *,
    tenor: str = "3m",
    day_count: int = 365,
) -> pl.DataFrame:
    """Align the latest strictly prior Shibor fixing to each trading day."""
    if shibor.is_empty():
        return pl.DataFrame()
    rates = (
        shibor.select(
            pl.col("date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias("rate_date"),
            pl.col(tenor).cast(pl.Float64).alias("annual_rate_percent"),
        )
        .drop_nulls()
        .sort("rate_date")
    )
    trading_days = (
        calendar.filter(pl.col("is_open") == 1)
        .select(pl.col("cal_date").alias("trade_date"))
        .unique()
        .sort("trade_date")
        .with_columns((pl.col("trade_date") - pl.duration(days=1)).alias("lookup_date"))
    )
    return (
        trading_days.join_asof(
            rates,
            left_on="lookup_date",
            right_on="rate_date",
            strategy="backward",
        )
        .drop("lookup_date")
        .with_columns(
            ((1 + pl.col("annual_rate_percent") / 100) ** (1 / day_count) - 1).alias(
                "risk_free_return"
            ),
            pl.col("rate_date").alias("available_at"),
            pl.lit(f"shibor_{tenor}").alias("rate_id"),
            pl.lit(day_count).alias("day_count"),
            pl.lit("strictly_prior_fixing").alias("lookahead_policy"),
            pl.lit("tushare").alias("source_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        )
        .drop_nulls(["rate_date", "risk_free_return"])
    )
