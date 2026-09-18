from dataclasses import dataclass


@dataclass(frozen=True)
class RiskModelPlan:
    factor_return_artifact_type: str
    specific_return_artifact_type: str
    covariance_estimator_id: str
    covariance_adjustment_ids: tuple[str, ...]
    specific_risk_estimator_id: str
    specific_risk_adjustment_ids: tuple[str, ...]
    security_covariance_assembler_id: str
    calibration_test_ids: tuple[str, ...]
