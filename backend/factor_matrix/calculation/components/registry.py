from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.contracts import Stage


@dataclass(frozen=True)
class ComponentPlanSpec:
    plan_id: str
    version: str
    stage: Stage
    configuration: object
    description: str

    def __post_init__(self) -> None:
        if not self.plan_id or any(char.isspace() for char in self.plan_id):
            raise ValueError("COMPONENT_PLAN_ID_INVALID")
        if not self.version:
            raise ValueError("COMPONENT_PLAN_VERSION_REQUIRED")
        if not type(self.configuration).__name__.endswith("Plan"):
            raise ValueError("COMPONENT_PLAN_CONFIGURATION_REQUIRED")


class ComponentPlanRegistry:
    def __init__(self, plans: Iterable[ComponentPlanSpec] = ()) -> None:
        self._plans: dict[tuple[str, str], ComponentPlanSpec] = {}
        for plan in plans:
            self.register(plan)

    def register(self, plan: ComponentPlanSpec) -> None:
        key = (plan.plan_id, plan.version)
        if key in self._plans:
            raise ValueError(f"COMPONENT_PLAN_DUPLICATE id={key[0]} version={key[1]}")
        self._plans[key] = plan

    def remove(self, plan_id: str, version: str) -> None:
        key = (plan_id, version)
        if key not in self._plans:
            raise KeyError(key)
        del self._plans[key]

    def get(self, plan_id: str, version: str) -> ComponentPlanSpec:
        try:
            return self._plans[(plan_id, version)]
        except KeyError as exc:
            raise KeyError(f"COMPONENT_PLAN_UNKNOWN id={plan_id} version={version}") from exc

    def search(self, *, stage: Stage | None = None, query: str = "") -> tuple[ComponentPlanSpec, ...]:
        needle = query.strip().lower()
        return tuple(sorted(
            (
                plan for plan in self._plans.values()
                if (stage is None or plan.stage is stage)
                and (not needle or needle in " ".join(
                    (plan.plan_id, plan.version, plan.stage.value, plan.description)
                ).lower())
            ),
            key=lambda plan: (plan.stage.value, plan.plan_id, plan.version),
        ))
