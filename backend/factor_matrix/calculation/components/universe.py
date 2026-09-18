from dataclasses import dataclass


@dataclass(frozen=True)
class UniversePlan:
    membership_resolver_id: str
    tradability_rule_ids: tuple[str, ...]
    return_adjustment_audit_id: str
    liquidity_measure_id: str
    board_scope_policy_id: str
