from dataclasses import dataclass


@dataclass(frozen=True)
class AttributionPlan:
    exposure_artifact_type: str
    factor_return_artifact_type: str
    specific_return_artifact_type: str
    cost_artifact_type: str
    decomposition_id: str
    identity_check_id: str
    contribution_group_ids: tuple[str, ...]
