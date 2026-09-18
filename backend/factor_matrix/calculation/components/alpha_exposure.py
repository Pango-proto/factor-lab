from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaExposurePlan:
    alpha_factor_ids: tuple[str, ...]
    frozen_risk_factor_set_artifact_type: str
    frozen_risk_factor_set_version: str
    raw_builder_ids: tuple[tuple[str, str], ...]
    imputer_id: str
    outlier_handler_id: str
    initial_scaler_id: str
    weighted_projection_solver_id: str
    weight_model_id: str
    residual_scaler_id: str
    residual_centering_allowed: bool
    board_scope_policy_id: str

    def __post_init__(self) -> None:
        if self.residual_centering_allowed:
            raise ValueError("ALPHA_RESIDUAL_RECENTERING_FORBIDDEN")

