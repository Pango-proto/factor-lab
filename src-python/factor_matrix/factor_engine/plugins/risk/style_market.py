from pathlib import Path

from ...contracts import FactorRole
from ...risk_parameters import descriptor_parameters
from ._factory import risk_definition


PATH = Path(__file__)
BETA = descriptor_parameters("beta")
RESIDUAL_VOL = descriptor_parameters("residual_volatility")
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="beta", display_name="贝塔",
        role=FactorRole.STYLE_RISK,
        formula_expr="rolling_beta(asset_total_return, board_benchmark_return)",
        source_tables=("returns_daily", "board_benchmark_daily"),
        source_fields=("total_return", "board_return"), params=BETA,
        lookback_days=int(BETA["lookback_days"]),
        min_obs=int(BETA["minimum_observations"]), pit_key="trade_date",
        hypothesis="解释证券对系统性市场收益的敏感度。",
        orthogonalize_after=("size",), family_root_id="market_sensitivity",
        variant_count=2,
    ),
    risk_definition(
        definition_path=PATH, factor_id="residual_volatility", display_name="残差波动率",
        role=FactorRole.STYLE_RISK,
        formula_expr="std(resid(asset_total_return ~ beta * board_benchmark_return))",
        source_tables=("returns_daily", "board_benchmark_daily"),
        source_fields=("total_return", "board_return"), params=RESIDUAL_VOL,
        lookback_days=int(RESIDUAL_VOL["lookback_days"]),
        min_obs=int(RESIDUAL_VOL["minimum_observations"]), pit_key="trade_date",
        hypothesis="高特异波动股票存在共同运动，遗漏会污染反转类 Alpha。",
        orthogonalize_after=("size", "beta"), depends_on=("beta",),
        family_root_id="volatility", variant_count=3,
    ),
)
