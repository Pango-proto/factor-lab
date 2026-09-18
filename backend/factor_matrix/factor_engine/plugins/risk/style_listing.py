from pathlib import Path

from ...contracts import FactorRole
from ...risk_parameters import descriptor_parameters
from ._factory import risk_definition


PATH = Path(__file__)
PARAMS = descriptor_parameters("listing_age")
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="listing_age", display_name="上市年限",
        role=FactorRole.STYLE_RISK,
        formula_expr="log1p((trade_date - exchange_list_date).calendar_days)",
        source_tables=("tradable_universe_v1",),
        source_fields=("trade_date", "asset_id", "exchange_list_date"),
        params=PARAMS, pit_key="trade_date",
        hypothesis="可交易域次新过滤后仍可能存在上市年限共同风险。",
        orthogonalize_after=("size",), family_root_id="listing_age", variant_count=2,
    ),
)
