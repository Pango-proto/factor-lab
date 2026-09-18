from __future__ import annotations

import json
from pathlib import Path

from .contracts import PredictivityConfig, ReturnDecompositionConfig
from .return_decomposition import HUBER_IMPLEMENTATION_ID


def load_predictivity_config(
    path: Path, *, derived_params: dict[str, object], frozen_choices: dict[str, object],
) -> PredictivityConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PredictivityConfig(
        reported_horizons_days=tuple(payload["reported_horizons_days"]),
        scoring_horizon_policy=payload["scoring_horizon_policy"],
        minimum_burn_in_days=int(derived_params["burn_in_days"]),
        embargo_days=payload["embargo_days"],
        folds=payload["cross_validation"]["folds"],
        correlation_method=payload["correlation_method"],
        factor_weight_estimator=payload["factor_weight_estimator"],
        shrinkage_prior_precision=float(frozen_choices["shrinkage_prior_precision"]),
        fdr_q=float(frozen_choices["fdr_q"]),
        allowed_preregistered_groups=tuple(payload["allowed_preregistered_groups"]),
    )


def load_return_decomposition_config(
    path: Path, *, derived_params: dict[str, object],
) -> ReturnDecompositionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("weight_scheme")
        != "weight_metric_contract_v1.regression_base_weight"
        or payload.get("weight_artifact_required") is not True
    ):
        raise ValueError("L2B_WEIGHT_METRIC_CONTRACT_REQUIRED")
    robust = payload["robustifier"]
    if robust.get("implementation_id") != HUBER_IMPLEMENTATION_ID:
        raise ValueError("L2B_HUBER_IMPLEMENTATION_CONFIG_MISMATCH")
    quality = payload["quality_gates"]
    return ReturnDecompositionConfig(
        maximum_condition_number=float(derived_params["maximum_condition_number"]),
        constraint_tolerance=quality["constraint_tolerance"],
        attribution_identity_tolerance=quality["attribution_identity_tolerance"],
        huber_delta=float(derived_params["huber_delta"]),
        maximum_iterations=robust["maximum_iterations"],
        convergence_tolerance=robust["convergence_tolerance"],
        expected_r_squared_min=quality["expected_daily_r_squared_range"][0],
        expected_r_squared_max=quality["expected_daily_r_squared_range"][1],
        maximum_invalid_date_ratio=float(derived_params["maximum_invalid_date_ratio"]),
    )
