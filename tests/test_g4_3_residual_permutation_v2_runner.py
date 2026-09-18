from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.g4_3_residual_permutation_v2_runner import (
    G43V2Config,
    _balanced_asset_selection,
    _industry_label_change_decision,
    _label_change_summary,
    _rank_daily_strata,
)


def test_v2_config_is_draft_and_reads_daily_strata_contract() -> None:
    config = G43V2Config.from_json(Path("config/g4_3_residual_permutation_v2.json"))
    config.validate()
    assert config.status == "draft"
    assert config.strata_definition.startswith("daily PIT")
    assert config.control_band == (0.8, 1.25)


def test_balanced_asset_selection_round_robins_board_buckets() -> None:
    exposure = pl.DataFrame({
        "trade_date": ["2026-08-14"] * 8,
        "asset_id": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        "board_id": ["MAIN"] * 4 + ["STAR"] * 4,
    })
    selected = _balanced_asset_selection(
        exposure, exposure["asset_id"].to_list(), 4, "balanced_by_board"
    )
    assert selected == ["A1", "A2", "B1", "B2"]


def test_daily_strata_are_bounded_and_deterministic() -> None:
    frame = pl.DataFrame({
        "trade_date": ["2026-08-13"] * 10 + ["2026-08-14"] * 10,
        "asset_id": [f"A{i:02d}" for i in range(10)] * 2,
        "risk_size": list(range(10)) * 2,
        "risk_liquidity": list(reversed(range(10))) * 2,
    })
    ranked = _rank_daily_strata(frame)
    assert ranked["stratum"].min() >= 0
    assert ranked["stratum"].max() <= 49
    assert ranked.select("trade_date", "asset_id", "stratum").equals(
        _rank_daily_strata(frame).select("trade_date", "asset_id", "stratum")
    )


def test_label_change_summary_reports_asset_and_change_date() -> None:
    labels = {
        "industry": np.array([
            [0, 1],
            [0, 1],
            [2, 1],
        ], dtype=int),
        "board": np.array([
            [0, 1],
            [0, 1],
            [0, 1],
        ], dtype=int),
    }
    summary = _label_change_summary(labels, ["A", "B"], ["d1", "d2", "d3"])
    assert summary["industry"]["changed_asset_count"] == 1
    assert summary["industry"]["changed_assets"] == ["A"]
    assert summary["industry"]["change_dates"] == ["d3"]
    assert summary["board"]["changed_asset_count"] == 0


def test_label_change_summary_does_not_treat_listing_coverage_as_reclassification() -> None:
    labels = {"industry": np.array([[-1], [2], [2]], dtype=int)}
    summary = _label_change_summary(labels, ["NEW"], ["d1", "d2", "d3"])
    assert summary["industry"]["changed_asset_count"] == 0


def test_label_change_decision_has_preregistered_reject_and_exclude_paths() -> None:
    ids = ["A", "B", "C", "D"]
    no_change = {"industry": {"changed_assets": [], "group_asset_counts": {"0": 4}, "group_changed_counts": {}}}
    assert _industry_label_change_decision(ids, no_change, "reject", 0.01)["action"] == "keep"

    one_change = {
        "industry": {
            "changed_assets": ["A"],
            "group_asset_counts": {"0": 2, "1": 2},
            "group_changed_counts": {"0": 1},
        }
    }
    assert _industry_label_change_decision(ids, one_change, "reject", 0.01)["action"] == "reject"
    decision = _industry_label_change_decision(ids, one_change, "exclude_below_fraction", 0.50)
    assert decision["action"] == "exclude"
    assert decision["max_group_changed_asset_fraction"] == 0.5
    assert _industry_label_change_decision(ids, one_change, "exclude_below_fraction", 0.10)["action"] == "reject"
