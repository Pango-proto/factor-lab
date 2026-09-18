from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Mapping


ALLOWED_PREREGISTERED_GROUPS = frozenset({
    "market_cap_quantile", "liquidity_quantile", "board", "market_volatility_regime",
})


class L2Chain(StrEnum):
    PREDICTIVITY = "l2a_predictivity"
    RETURN_DECOMPOSITION = "l2b_return_decomposition"


class RegressionMode(StrEnum):
    RISK_ONLY = "risk_only"
    RISK_PLUS_ALPHA = "risk_plus_alpha"


@dataclass(frozen=True)
class PredictivityConfig:
    reported_horizons_days: tuple[int, ...]
    scoring_horizon_policy: str
    minimum_burn_in_days: int
    embargo_days: int
    folds: int
    correlation_method: str
    factor_weight_estimator: str
    shrinkage_prior_precision: float
    fdr_q: float
    allowed_preregistered_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reported_horizons_days or any(horizon < 1 for horizon in self.reported_horizons_days):
            raise ValueError("L2A_HORIZONS_INVALID")
        if tuple(sorted(set(self.reported_horizons_days))) != self.reported_horizons_days:
            raise ValueError("L2A_HORIZONS_MUST_BE_SORTED_UNIQUE")
        if self.scoring_horizon_policy != "alpha_assertion.horizon_days":
            raise ValueError("L2A_RESULT_DRIVEN_HORIZON_SELECTION_FORBIDDEN")
        if self.minimum_burn_in_days < 1:
            raise ValueError("L2A_BURN_IN_INVALID")
        if self.embargo_days < 0 or self.folds < 2:
            raise ValueError("L2A_VALIDATION_CONFIG_INVALID")
        if self.correlation_method != "spearman":
            raise ValueError("L2A_CORRELATION_METHOD_UNSUPPORTED")
        if self.factor_weight_estimator != "diagonal_icir":
            raise ValueError("L2A_FULL_COVARIANCE_WEIGHTING_NOT_ENABLED")
        if self.shrinkage_prior_precision <= 0 or not 0 < self.fdr_q < 1:
            raise ValueError("L2A_REGULARIZATION_INVALID")
        if not set(self.allowed_preregistered_groups) <= ALLOWED_PREREGISTERED_GROUPS:
            raise ValueError("L2A_AD_HOC_GROUP_FORBIDDEN")


@dataclass(frozen=True)
class ReturnDecompositionConfig:
    maximum_condition_number: float
    constraint_tolerance: float
    attribution_identity_tolerance: float
    huber_delta: float
    maximum_iterations: int
    convergence_tolerance: float
    expected_r_squared_min: float
    expected_r_squared_max: float
    maximum_invalid_date_ratio: float

    def __post_init__(self) -> None:
        if self.maximum_condition_number <= 1:
            raise ValueError("L2B_CONDITION_THRESHOLD_INVALID")
        if self.constraint_tolerance > 1e-10 or self.attribution_identity_tolerance > 1e-10:
            raise ValueError("L2B_IDENTITY_TOLERANCE_TOO_LOOSE")
        if self.huber_delta <= 0 or self.maximum_iterations < 1:
            raise ValueError("L2B_ROBUSTIFIER_INVALID")
        if self.convergence_tolerance <= 0:
            raise ValueError("L2B_NUMERICAL_PARAMETER_INVALID")
        if not 0 <= self.expected_r_squared_min < self.expected_r_squared_max <= 1:
            raise ValueError("L2B_EXPECTED_R_SQUARED_RANGE_INVALID")
        if not 0 <= self.maximum_invalid_date_ratio < 1:
            raise ValueError("L2B_INVALID_DATE_RATIO_INVALID")


@dataclass(frozen=True)
class ResearchEvent:
    event_id: str
    run_at: datetime
    factor_ids: tuple[str, ...]
    config_sha: str
    data_end_date: date
    touched_holdout: bool
    headline_ic: float | None
    decision: str
    operator: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.event_id or not self.factor_ids:
            raise ValueError("RESEARCH_EVENT_IDENTITY_REQUIRED")
        if len(self.config_sha) != 64:
            raise ValueError("RESEARCH_EVENT_CONFIG_SHA_INVALID")
        if not self.decision.strip() or not self.operator.strip():
            raise ValueError("RESEARCH_EVENT_DECISION_AND_OPERATOR_REQUIRED")


@dataclass(frozen=True)
class PredictivityPoint:
    evaluation_index: int
    training_end_index: int | None
    raw_ic: float | None
    shrunk_ic: float
    standard_error: float | None
    confidence_low: float | None
    confidence_high: float | None
    factor_weight: float
    status_signal: str


@dataclass(frozen=True)
class NeutralizationFidelityPoint:
    evaluation_index: int
    raw_shrunk_ic: float
    neutralized_shrunk_ic: float
    absolute_ic_retention_ratio: float | None
    sign_preserved: bool | None
    status: str


@dataclass(frozen=True)
class FamaMacBethPoint:
    evaluation_index: int
    training_end_index: int | None
    coefficients: tuple[float, ...]
    standard_errors: tuple[float | None, ...]
    nw_t_values: tuple[float | None, ...]


@dataclass(frozen=True)
class RegressionInput:
    asset_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    factor_families: tuple[str, ...]
    exposures: tuple[tuple[float, ...], ...]
    realized_returns: tuple[float, ...]
    base_weights: tuple[float, ...]
    is_tradable: tuple[bool, ...]
    equality_constraints: tuple[tuple[float, ...], ...] = ()
    identification_prevalidated: bool = False
    prevalidated_matrix_rank: int | None = None


@dataclass(frozen=True)
class RegressionResult:
    mode: RegressionMode
    factor_returns: tuple[float, ...]
    specific_returns: tuple[float, ...]
    estimation_weights: tuple[float | None, ...]
    huber_weight_multipliers: tuple[float | None, ...]
    in_estimation_domain: tuple[bool, ...]
    included_assets: tuple[str, ...]
    excluded_assets: tuple[str, ...]
    sample_count: int
    label_sample_count: int
    matrix_rank: int
    condition_number: float
    r_squared: float
    estimation_domain_r_squared: float
    unweighted_r_squared: float
    estimation_domain_unweighted_r_squared: float
    constraint_error: float
    regression_identity_error: float
    base_weight_alpha_risk_cross_gram_max: float | None
    effective_weight_alpha_risk_cross_gram_max: float | None
    r_squared_in_expected_range: bool
    status: str


@dataclass(frozen=True)
class DailyDesign:
    regression_input: RegressionInput
    present_industries: tuple[str, ...]
    industry_member_counts: Mapping[str, int]
    industry_wls_weight_sums: Mapping[str, float]
    industry_constraint_weights: Mapping[str, float]
    matrix_rank: int
    expected_matrix_rank: int
    constraint_identification_passed: bool
    warnings: tuple[str, ...]
    categorical_blocks: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    category_member_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    category_constraint_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    category_wls_weight_sums: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    constraint_rank: int = 0
    stacked_rank: int = 0
    kkt_rank: int = 0
