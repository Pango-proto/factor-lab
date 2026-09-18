from pathlib import Path

from ...contracts import FactorRole
from ...risk_parameters import descriptor_parameters
from ._factory import risk_definition


PATH = Path(__file__)
PARAMS = descriptor_parameters("liquidity")
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="liquidity", display_name="流动性",
        role=FactorRole.STYLE_RISK,
        formula_expr="log(configured_turnover_aggregation)",
        source_tables=("valuation_daily", "prices_daily"),
        source_fields=("turnover_rate", "amount"), params=PARAMS,
        lookback_days=int(PARAMS["lookback_days"]),
        min_obs=int(PARAMS["minimum_observations"]), pit_key="trade_date",
        hypothesis="A 股流动性冲击具有独立于规模的共同变化。",
        orthogonalize_after=("size",), family_root_id="liquidity", variant_count=3,
    ),
)
