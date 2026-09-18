from pathlib import Path

from ...contracts import FactorRole
from ...risk_parameters import descriptor_parameters
from ._factory import risk_definition


PATH = Path(__file__)
NONLINEAR = descriptor_parameters("nonlinear_size")
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="size", display_name="规模",
        role=FactorRole.STYLE_RISK, formula_expr="log(float_mkt_cap)",
        source_tables=("valuation_daily",), source_fields=("float_mkt_cap",),
        params={}, pit_key="trade_date", hypothesis="规模是 A 股持续的截面共同风险。",
        family_root_id="size", variant_count=1,
    ),
    risk_definition(
        definition_path=PATH, factor_id="nonlinear_size", display_name="非线性规模",
        role=FactorRole.STYLE_RISK,
        formula_expr=(
            "pow(winsorized_weighted_z(size), power) residualized against size"
        ),
        source_tables=("valuation_daily",), source_fields=("float_mkt_cap",),
        params=NONLINEAR, pit_key="trade_date",
        hypothesis="捕捉线性规模无法解释的中盘共同风险。",
        orthogonalize_after=("size",), depends_on=("size",),
        family_root_id="size", variant_count=2,
    ),
)
