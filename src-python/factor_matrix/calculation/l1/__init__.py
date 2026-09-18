"""Deterministic L1 exposure transformations; this package never consumes returns."""

from .projection import apply_alpha_direction, weighted_residualize, weighted_scale_only
from .config import (
    CategoricalConstraintBlock, L1RiskExposureConfig,
    assert_exposure_publish_allowed, load_l1_risk_exposure_config,
)
from .universe_variants import apply_universe_variant, summarize_variant_sensitivity
from .runner import run_l1_risk_exposure
from .history_runner import run_l1_risk_exposure_history
from .history_diagnostics import run_l1_history_diagnostics, promote_l1_history

__all__ = [
    "CategoricalConstraintBlock", "L1RiskExposureConfig", "apply_alpha_direction",
    "apply_universe_variant", "assert_exposure_publish_allowed",
    "load_l1_risk_exposure_config", "weighted_residualize", "weighted_scale_only",
    "summarize_variant_sensitivity",
    "run_l1_risk_exposure",
    "run_l1_risk_exposure_history",
    "run_l1_history_diagnostics", "promote_l1_history",
]
