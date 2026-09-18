"""Opt-in core-six raw momentum; not installed in the legacy L1 registry.

The caller supplies an exchange calendar and a revision-resolved return view.
No price path, signal label, cross-sectional imputation or publication lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from ...storage import file_sha256


@dataclass(frozen=True)
class MomentumDefinition:
    nearest_lag: int
    farthest_lag: int
    minimum_observations: int

    def __post_init__(self) -> None:
        values = (self.nearest_lag, self.farthest_lag, self.minimum_observations)
        if any(type(v) is not int for v in values) or not (
            1 <= self.nearest_lag <= self.farthest_lag
            and 1 <= self.minimum_observations <= self.window_sessions
        ):
            raise ValueError("MOMENTUM_PARAMETERS_INVALID")

    @property
    def window_sessions(self) -> int:
        return self.farthest_lag - self.nearest_lag + 1


def load_core6_design(path: Path, *, project_root: Path) -> tuple[dict, MomentumDefinition]:
    """Verify inherited evidence and load the explicitly inactive candidate."""
    spec = json.loads(path.read_text())
    if (spec["design_id"] != "risk_core6_design_v2"
        or spec["status"] != "approved_design_inactive"
        or spec["active"] is not False
        or spec["risk_set_version"] is not None
        or spec["risk_basis_id"] is not None):
        raise ValueError("CORE6_INACTIVE_DESIGN_REQUIRED")
    for relative, expected in spec["inherited_file_hashes"].items():
        if file_sha256(project_root / relative) != expected:
            raise ValueError("CORE6_INHERITED_EVIDENCE_CHANGED:" + relative)
    styles = spec["style_order"]
    if styles != ["size", "beta", "residual_volatility", "liquidity", "nonlinear_size", "momentum"]:
        raise ValueError("CORE6_MEMBER_AXIS_INVALID")
    if spec["members"] != ["country", "industry_sw1", *styles]:
        raise ValueError("CORE6_MEMBER_AXIS_INVALID")
    industry = spec["industry_columns"]
    if len(industry) != 31 or len(set(industry)) != 31:
        raise ValueError("CORE6_INDUSTRY_AXIS_INVALID")
    expected_axis = ["risk_country", *industry, *["risk_" + f for f in styles]]
    if spec["expanded_columns"] != expected_axis or spec["expanded_factor_count"] != len(expected_axis):
        raise ValueError("CORE6_EXPANDED_AXIS_INVALID")
    definition = spec["momentum"]
    if (definition["role"] != "basis" or definition["bettable"] is not False
        or definition["orthogonalize_after"] != ["size", "beta"]
        or definition["missing_policy"] != "observed_product_no_fill_no_rescale"
        or definition["resumption_policy"] != "exclude_with_count"
        or definition["invalid_return_policy"] != "raw_unavailable"
        or definition["calendar_policy"] != "complete_common_exchange_window"):
        raise ValueError("CORE6_MOMENTUM_POLICY_UNSUPPORTED")
    return spec, MomentumDefinition(**definition["window"])


def momentum_descriptor(
    *, as_of: date, decision_time: datetime, calendar: list[date],
    assets: pl.DataFrame, returns: pl.DataFrame, definition: MomentumDefinition,
) -> pl.DataFrame:
    """Compound observed decimal total returns at inclusive lags 21..251.

    A full common calendar window is required, then >=200 usable asset rows.
    Partial products are labelled as such, never rescaled to a full-window return.
    available_at must be backed by observed/conservative first-seen evidence;
    this function does not manufacture historical availability timestamps.
    """
    tz = ZoneInfo("Asia/Shanghai")
    if decision_time.tzinfo is None:
        raise ValueError("MOMENTUM_TIMEZONE_REQUIRED")
    if type(as_of) is not date or as_of > decision_time.astimezone(tz).date():
        raise ValueError("MOMENTUM_ASOF_IN_FUTURE")
    if any(type(d) is not date for d in calendar) or len(set(calendar)) != len(calendar):
        raise ValueError("MOMENTUM_CALENDAR_INVALID")
    dates = sorted(d for d in calendar if d <= as_of)
    if not dates or dates[-1] != as_of:
        raise ValueError("MOMENTUM_ASOF_NOT_IN_CALENDAR")
    asset_ids = assets.get_column("asset_id").to_list()
    if not asset_ids or len(set(asset_ids)) != len(asset_ids) or any(not isinstance(a, str) or not a for a in asset_ids):
        raise ValueError("MOMENTUM_ASSET_AXIS_INVALID")
    required = {"asset_id", "trade_date", "total_return", "return_source", "available_at", "availability_evidence"}
    if not required <= set(returns.columns):
        raise ValueError("MOMENTUM_INPUT_COLUMNS_REQUIRED")
    if returns.schema["trade_date"] != pl.Date:
        raise ValueError("MOMENTUM_TRADE_DATE_TYPE")
    time_type = returns.schema["available_at"]
    if not isinstance(time_type, pl.Datetime) or time_type.time_zone is None:
        raise ValueError("MOMENTUM_TIMEZONE_REQUIRED")

    complete = len(dates) >= definition.farthest_lag + 1
    window = [dates[-1-k] for k in range(definition.nearest_lag, definition.farthest_lag+1) if k < len(dates)]
    groups: dict[str, list[dict]] = {a: [] for a in asset_ids}
    visible_dates: dict[str, set[date]] = {a: set() for a in asset_ids}
    counts = {a: {"n_unavailable": 0, "n_unknown_availability": 0} for a in asset_ids}
    rows = returns.filter(pl.col("asset_id").is_in(asset_ids) & pl.col("trade_date").is_in(window))
    for row in rows.iter_rows(named=True):
        asset = row["asset_id"]
        available = row["available_at"]
        if available is None:
            counts[asset]["n_unknown_availability"] += 1
            continue
        # Exclude unavailable versions before inspecting their value or source.
        if available > decision_time:
            counts[asset]["n_unavailable"] += 1
            continue
        if available.astimezone(tz).date() < row["trade_date"]:
            raise ValueError("MOMENTUM_AVAILABILITY_PRECEDES_EVENT")
        if row["availability_evidence"] not in ("observed_first_seen", "conservative_base_first_seen"):
            counts[asset]["n_unknown_availability"] += 1
            continue
        if row["trade_date"] in visible_dates[asset]:
            raise ValueError("MOMENTUM_REVISION_RESOLUTION_REQUIRED")
        visible_dates[asset].add(row["trade_date"])
        groups[asset].append(row)

    output = []
    for asset in asset_ids:
        values, available_times = [], []
        resumption = invalid = nulls = conservative = 0
        for row in sorted(groups[asset], key=lambda r: r["trade_date"]):
            available_times.append(row["available_at"])
            conservative += row["availability_evidence"] == "conservative_base_first_seen"
            if row["return_source"] == "resumption":
                resumption += 1
                continue
            value = row["total_return"]
            if value is None:
                nulls += 1
            elif (not isinstance(value, (int, float)) or isinstance(value, bool)
                  or not math.isfinite(value) or value < -1 or not row["return_source"]):
                invalid += 1
            else:
                values.append(value)
        reason = ("calendar_history_insufficient" if not complete else
                  "invalid_return" if invalid else
                  "insufficient_observations" if len(values) < definition.minimum_observations else None)
        raw = None
        if reason is None:
            raw = math.prod(1 + value for value in values) - 1
            if not math.isfinite(raw):
                raw, reason = None, "nonfinite_product"
        output.append({
            "asset_id": asset, "momentum_raw": raw, "n_momentum_obs": len(values),
            "window_start": min(window) if window else None,
            "window_end": max(window) if window else None,
            "expected_window_sessions": definition.window_sessions,
            "calendar_window_sessions": len(window), "calendar_complete": complete,
            "n_visible_rows": len(groups[asset]), "n_resumption_excluded": resumption,
            "n_null_return": nulls, "n_invalid_return": invalid,
            "n_conservative_availability": conservative, **counts[asset],
            "source_available_at": max(available_times) if available_times else None,
            "raw_missing": raw is None, "unavailable_reason": reason,
            "partial_observed_product": raw is not None and len(values) < definition.window_sessions,
            "imputed": False, "fallback": False,
        })
    return pl.DataFrame(output, schema_overrides={
        "momentum_raw": pl.Float64, "window_start": pl.Date, "window_end": pl.Date,
        "source_available_at": time_type, "unavailable_reason": pl.String,
    })
