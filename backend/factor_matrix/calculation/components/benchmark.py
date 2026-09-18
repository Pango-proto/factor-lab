from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPlan:
    construction_id: str
    constituent_source_id: str
    weighting_id: str
    return_definition_id: str
    usage_tag: str
