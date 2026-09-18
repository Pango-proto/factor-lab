"""Canonical quantitative definitions shared by every calculation layer.

Downstream modules import the functions in this module.  They must not recreate
the same quantities from raw columns, because that silently changes the model's
economic meaning.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Sequence
from collections.abc import Sequence
from datetime import date


FLOAT_MKT_CAP = "daily_basic.float_share * prices_daily.close_unadjusted"
TOTAL_MKT_CAP = "daily_basic.total_share * prices_daily.close_unadjusted"
RETURN_SIMPLE = "(prices_daily.raw_close[t] * prices_daily.adj_factor[t]) / (last_traded_raw_close * last_traded_adj_factor) - 1"
TRADE_CALENDAR = "trade_calendar where is_open = 1"
TRADABLE_CROSS_SECTION = "tradable_universe_v1 where is_tradable = true"
# These names deliberately separate two economic semantics.  The default
# formula is shared for bootstrap runs, but formal runs bind both semantics to
# the same versioned weight artifact through weight_metric_contract_v1.json.
REGRESSION_BASE_WEIGHT = "sqrt(FLOAT_MKT_CAP)"
EXPOSURE_ORTHOGONALIZATION_WEIGHT = "BOUND_TO_REGRESSION_BASE_WEIGHT"
# Backwards-compatible display alias.  New calculation code must use the
# semantic names above rather than treating one symbol as two concepts.
WLS_WEIGHT = REGRESSION_BASE_WEIGHT
CONSTRAINT_WEIGHT = "FLOAT_MKT_CAP / sum(FLOAT_MKT_CAP)"
SIZE_RAW = "log(FLOAT_MKT_CAP)"
SPECIFIC_VOLATILITY = "sqrt(specific_risk_v1.specific_variance)"
SIGMA_R = "population_std(realized_return | trade_date, board_id, TRADABLE_CROSS_SECTION)"
LISTING_AGE_CALENDAR_DAYS = "(trade_date - exchange_list_date).days"


def regression_base_weight(float_market_cap: float) -> float:
    if not math.isfinite(float_market_cap) or float_market_cap <= 0:
        raise ValueError("CANONICAL_FLOAT_MKT_CAP_INVALID")
    return math.sqrt(float_market_cap)


def exposure_orthogonalization_weight(float_market_cap: float) -> float:
    """Bootstrap metric; formal runs override it with the bound PIT artifact."""
    return regression_base_weight(float_market_cap)


def wls_weight(float_market_cap: float) -> float:
    """Compatibility wrapper for callers not yet renamed."""
    return regression_base_weight(float_market_cap)


def constraint_weights(float_market_caps: Sequence[float]) -> tuple[float, ...]:
    if not float_market_caps:
        raise ValueError("CANONICAL_CONSTRAINT_WEIGHT_EMPTY")
    values = tuple(float(value) for value in float_market_caps)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("CANONICAL_FLOAT_MKT_CAP_INVALID")
    total = sum(values)
    return tuple(value / total for value in values)


def raw_size(float_market_cap: float) -> float:
    if not math.isfinite(float_market_cap) or float_market_cap <= 0:
        raise ValueError("CANONICAL_FLOAT_MKT_CAP_INVALID")
    return math.log(float_market_cap)


def listing_age_calendar_days(trade_date: date, exchange_list_date: date) -> int:
    """Exact PIT age for continuous descriptors; d0 separately uses trading-day age."""
    value = (trade_date - exchange_list_date).days
    if value < 0:
        raise ValueError("CANONICAL_LISTING_AGE_NEGATIVE")
    return value


def cross_section_sigma_r(realized_returns: Sequence[float]) -> float:
    """Equal-weight population standard deviation on the canonical tradable slice."""
    values = tuple(float(value) for value in realized_returns if math.isfinite(value))
    if not values:
        raise ValueError("CANONICAL_SIGMA_R_EMPTY")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


# ---- R-squared: six coexisting conventions, one authority ----
# G3 _MANIFEST publishes exactly one of these; the other five are report-only.
# Bound to g6_freeze_protocol descriptive_expectations.r_squared_statistic.
R2_DEFINITION = "mean_of_daily_full_label_weighted_r_squared"
R2_REPORTED_VARIANTS = (
    "mean_of_daily_estimation_domain_weighted_r_squared",
    "mean_of_daily_full_label_unweighted_r_squared",
    "row_weighted_r_squared",
    "median_of_daily_r_squared",
    "pooled_full_label_unweighted_r_squared",
)


# ---- EWMA half-lives: derived from K, never hand-entered ----
# T*_corr = 10K, T*_var = 3K, halflife = T* * ln(2) / 2
# K is the generated-column count of the current risk_set_expansion (44),
# NOT the logical member count (10).
EWMA_HALFLIFE_FORMULA = "T_star * log(2) / 2 ; T_star_corr = 10K ; T_star_var = 3K"


def ewma_halflives(k_columns: int) -> tuple[float, float]:
    """Return (correlation_halflife, variance_halflife) in trading days.

    Ordering is (corr, var) everywhere.  Reversed reporting produced the
    apparent 45/149 vs 152.49/45.75 conflict; both were this formula.
    """
    if k_columns <= 0:
        raise ValueError("CANONICAL_K_COLUMNS_INVALID")
    scale = math.log(2.0) / 2.0
    return (10 * k_columns * scale, 3 * k_columns * scale)


# ---- Risk exposure column selection ----
# risk_exposure_matrix carries two metadata columns that share the risk_ prefix
# with the 44 generated exposure columns. Selecting the design matrix by prefix
# alone silently pulls them in: both are strings, so they either raise on matrix
# construction or coerce to NaN and corrupt neutralisation. Prefix matching is
# never sufficient here.
RISK_EXPOSURE_METADATA_COLUMNS = frozenset({
    "risk_factor_set_id",
    "risk_factor_set_version",
})

RISK_EXPOSURE_COLUMN_COUNT = 44

RISK_SET_CANDIDATE_V1_MEMBERS: tuple[str, ...] = (
    "country", "industry_sw1", "board", "size", "beta",
    "residual_volatility", "liquidity", "nonlinear_size", "listing_age",
)


def risk_exposure_columns(
    *, risk_set_version: int, trade_date: date, available: Sequence[str],
) -> tuple[str, ...]:
    """Return the configured, physically available evaluation exposures.

    Index membership is intentionally absent from the evaluation set: its PIT
    history starts inside the development sample.  BSE is a coverage removal
    before the exchange opened, not a missing configured member.
    """
    if risk_set_version != 1:
        raise ValueError(f"RISK_EXPOSURE_RISK_SET_UNSUPPORTED {risk_set_version}")
    available_set = set(available)
    prefixes = {
        "country": ("risk_country",),
        "industry_sw1": ("risk_industry_",),
        "board": ("risk_board_",),
        "size": ("risk_size",),
        "beta": ("risk_beta",),
        "residual_volatility": ("risk_residual_volatility",),
        "liquidity": ("risk_liquidity",),
        "nonlinear_size": ("risk_nonlinear_size",),
        "listing_age": ("risk_listing_age",),
    }
    selected: list[str] = []
    for member in RISK_SET_CANDIDATE_V1_MEMBERS:
        names = tuple(sorted(
            name for name in available_set
            if any(name == prefix or name.startswith(prefix) for prefix in prefixes[member])
        ))
        if member == "board" and trade_date < date(2021, 11, 15):
            names = tuple(name for name in names if name != "risk_board_BSE")
        if not names:
            raise ValueError(
                f"RISK_EXPOSURE_MEMBER_UNAVAILABLE member={member} date={trade_date}"
            )
        selected.extend(names)
    return tuple(selected)
