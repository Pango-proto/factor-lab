from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestPlan:
    signal_timing_policy_id: str
    rebalance_schedule_id: str
    execution_simulator_id: str
    fill_policy_id: str
    corporate_action_policy_id: str
    cost_artifact_type: str
    capacity_policy_id: str
    evaluation_metric_ids: tuple[str, ...]
