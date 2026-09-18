from dataclasses import dataclass


@dataclass(frozen=True)
class RiskExposurePlan:
    risk_factor_set_id: str
    risk_factor_set_version: str
    raw_builder_ids: tuple[tuple[str, str], ...]
    ordered_transform_ids: tuple[tuple[str, str], ...]
    weight_model_id: str
    constrained_regression_solver_id: str
    orthogonality_check_id: str
    coverage_check_id: str
    persistence_check_id: str

