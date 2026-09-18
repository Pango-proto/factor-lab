from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioPlan:
    constructor_id: str
    objective_id: str
    solver_id: str
    constraint_ids: tuple[str, ...]
    turnover_model_id: str
    cost_artifact_type: str
    risk_artifact_type: str
    comparison_constructor_ids: tuple[str, ...]
