from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPlan:
    order_scheduler_id: str
    participation_policy_id: str
    venue_router_id: str
    price_limit_policy_id: str
    suspension_policy_id: str
    unsettled_position_policy_id: str
    shortfall_measure_id: str
