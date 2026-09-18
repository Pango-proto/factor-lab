from pathlib import Path


def test_frontend_does_not_define_strategy_formula_or_factor_ids() -> None:
    source = (Path(__file__).parents[1] / "frontend" / "App.tsx").read_text(encoding="utf-8")
    forbidden = (
        "momentum_12_1",
        "reversal_20d",
        "value_reversal_20d",
        "prod(1+r)",
        "skip_recent_trading_days",
    )
    assert not [token for token in forbidden if token in source]
