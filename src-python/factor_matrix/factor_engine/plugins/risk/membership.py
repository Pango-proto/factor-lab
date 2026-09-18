from pathlib import Path

from ...contracts import FactorRole, FactorStatus
from ._factory import risk_definition


PATH = Path(__file__)
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="index_membership", display_name="指数成分",
        role=FactorRole.MEMBERSHIP,
        formula_expr="PIT configured-index membership dummies",
        source_tables=("benchmark_membership_history",),
        source_fields=("asset_id", "benchmark_id", "effective_from", "effective_to"),
        params={}, pit_key="eff_date",
        hypothesis="被动资金申赎可能造成指数成分股共同流动性冲击。",
        family_root_id="passive_flow", variant_count=1,
    ),
    risk_definition(
        definition_path=PATH, factor_id="ownership", display_name="产权性质",
        role=FactorRole.MEMBERSHIP,
        formula_expr="PIT configured ownership-category dummies",
        source_tables=("ownership_pit",),
        source_fields=("asset_id", "ownership_type", "available_at"), params={},
        pit_key="available_at",
        hypothesis="产权性质影响政策敏感度、融资成本与尾部风险。",
        family_root_id="ownership", variant_count=1,
        status=FactorStatus.BLOCKED,
        status_reason="ownership_pit is unavailable in L0; synthetic ownership is forbidden",
    ),
)
