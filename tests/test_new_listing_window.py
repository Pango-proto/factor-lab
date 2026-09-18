from __future__ import annotations

from datetime import date

from factor_matrix.diagnostics.new_listing_window import _derive_group_result


def test_listing_window_requires_persistent_crossing_and_low_one_price_ratio() -> None:
    config = {
        "steady_state_age_range": [20, 25],
        "minimum_stocks_per_age": 30,
        "minimum_steady_state_age_points": 6,
        "smoothing_window_days": 1,
        "steady_state_tolerance": 0.2,
        "maximum_one_price_limit_ratio": 0.05,
        "required_persistence_days": 3,
        "fallback_days": 120,
    }
    rows = []
    for age in range(1, 31):
        rows.append({
            "regime_id": "MAIN_2019-07-01", "board_id": "MAIN",
            "regime_start": date(2019, 7, 1), "listing_age_days": age,
            "sigma_adjusted": 2.0 if age < 10 else 1.0,
            "n_stocks_used": 40,
            "one_price_limit_ratio": 0.10 if age < 12 else 0.0,
        })
    summary, _ = _derive_group_result(rows, config)
    assert summary["d_star"] == 12
    assert summary["stable"]
