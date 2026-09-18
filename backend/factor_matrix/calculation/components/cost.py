from dataclasses import dataclass


@dataclass(frozen=True)
class CostPlan:
    component_ids: tuple[str, ...]
    calibration_model_id: str
    liquidity_measure_id: str
    order_size_model_id: str
    board_parameter_set_id: str
    capacity_curve_id: str
