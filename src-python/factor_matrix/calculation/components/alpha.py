from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaPlan:
    exposure_artifact_type: str
    predictive_coefficient_artifact_type: str
    return_scale_estimator_id: str
    coefficient_shrinkage_id: str
    factor_weight_estimator_id: str
    forecast_combiner_id: str
    interval_estimator_id: str
