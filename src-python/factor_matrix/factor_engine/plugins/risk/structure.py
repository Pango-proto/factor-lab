from pathlib import Path

from ...contracts import FactorRole
from ._factory import risk_definition


PATH = Path(__file__)
PLUGINS = (
    risk_definition(
        definition_path=PATH, factor_id="country", display_name="中国市场",
        role=FactorRole.COUNTRY, formula_expr="1 for every eligible security",
        source_tables=("tradable_universe_v1",),
        source_fields=("trade_date", "asset_id"), params={}, pit_key="trade_date",
        hypothesis="全市场共同收益截距，定义风险回归的 country 维度。",
        family_root_id="market_structure", variant_count=1,
    ),
    risk_definition(
        definition_path=PATH, factor_id="industry_sw1", display_name="申万一级行业",
        role=FactorRole.INDUSTRY, formula_expr="PIT SW2021 L1 full one-hot",
        source_tables=("tradable_universe_v1",),
        source_fields=("trade_date", "asset_id", "sw_l1_code"),
        params={"classification_standard": "SW2021", "metadata_level": "L1"},
        pit_key="trade_date",
        hypothesis="行业内股票存在持续共同变动，遗漏会留下组内残差相关。",
        family_root_id="industry", variant_count=2,
    ),
    risk_definition(
        definition_path=PATH, factor_id="board", display_name="上市板块",
        role=FactorRole.BOARD, formula_expr="PIT listing-board full one-hot",
        source_tables=("tradable_universe_v1",),
        source_fields=("trade_date", "asset_id", "board_id"), params={},
        pit_key="trade_date",
        hypothesis="涨跌幅限制、投资者门槛与交易制度差异可能产生板块共同风险。",
        family_root_id="market_structure", variant_count=2,
    ),
)
