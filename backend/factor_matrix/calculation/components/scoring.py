from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringPlan:
    alpha_artifact_type: str
    cost_artifact_type: str
    holding_period_estimator_id: str
    contribution_decomposer_id: str
    ranking_method_ids: tuple[str, ...]
    display_transform_id: str
    stability_measure_ids: tuple[str, ...]
    board_display_policy_id: str
