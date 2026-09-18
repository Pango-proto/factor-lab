from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import polars as pl

import pytest

from factor_matrix.calculation.l1 import (
    apply_universe_variant, assert_exposure_publish_allowed,
    load_l1_risk_exposure_config, summarize_variant_sensitivity,
)
from factor_matrix.calculation.l2 import apply_l2b_label_policy, load_return_label_policy


PROJECT = Path(__file__).resolve().parents[1]


def test_l1_contract_freezes_all_share_regression_and_dual_industry_metadata() -> None:
    config = load_l1_risk_exposure_config(PROJECT / "config" / "l1_risk_exposure_v1.json")
    assert config.regression_universe == "all_a_share"
    assert config.regression_industry_column == "sw_l1_code"
    assert config.display_industry_column == "sw_l2_code"
    assert {block.factor_id for block in config.constraint_blocks} == {
        "industry_sw1", "board",
    }
    assert config.category_failure_policy == "mark_date_invalid"
    assert set(config.development_universe_variants) == {"d120", "derived"}
    assert config.formal_universe_variants == ("frozen_d0",)


def test_formal_exposure_publish_is_blocked_until_d0_is_frozen() -> None:
    policy = PROJECT / "config" / "new_listing_policy_v1.json"
    assert_exposure_publish_allowed(
        policy_path=policy, universe_variant="d120", update_current_pointer=False
    )
    with pytest.raises(RuntimeError, match="REQUIRES_FROZEN_D0_VARIANT"):
        assert_exposure_publish_allowed(
            policy_path=policy, universe_variant="d120", update_current_pointer=True
        )
    assert_exposure_publish_allowed(
        policy_path=policy, universe_variant="frozen_d0", update_current_pointer=True
    )


def test_d0_filter_and_d_star_reference_have_distinct_frozen_semantics() -> None:
    payload = json.loads(
        (PROJECT / "config" / "new_listing_policy_v1.json").read_text()
    )
    assert payload["d0_model_exclusion"]["status"] == "frozen"
    assert payload["d0_model_exclusion"]["value"] == 20
    assert payload["d0_model_exclusion"]["class"] == "B_frozen_modeling_choice"
    assert payload["d_star_reference"]["class"] == "A_label_free_derived_reference"
    assert payload["d_star_reference"]["not_used_as_exclusion"] is True
    assert payload["formal_exposure_publish_gate"]["formal_universe_variant"] == (
        "frozen_d0"
    )


def test_listing_age_segment_test_is_preregistered_before_g4() -> None:
    payload = json.loads(
        (PROJECT / "config" / "risk_factor_set_candidate_v1.json").read_text()
    )
    test = payload["candidate_specific_tests"]["listing_age"]
    assert test["preregistered"] is True
    assert [segment["segment_id"] for segment in test["segments"]] == [
        "A_residual_price_discovery",
        "B_post_steady_state_to_250",
        "C_mature",
    ]
    assert test["decision_rule"]["A_only"].startswith("use_bounded_linear_decay")
    assert test["per_board_regression_forbidden"] is True


def test_l2_label_policy_excludes_exchange_first_day_and_resumption() -> None:
    policy = load_return_label_policy(PROJECT / "config" / "return_label_policy_v1.json")
    frame = pl.DataFrame({
        "trade_date": [date(2026, 8, 10), date(2026, 8, 11), date(2026, 8, 12)],
        "exchange_list_date": [date(2026, 8, 10), date(2020, 1, 1), date(2020, 1, 1)],
        "return_source": ["price", "resumption", "price"],
        "total_return": [2.0, 0.8, 0.01],
    })
    result = apply_l2b_label_policy(frame, policy)
    assert result.get_column("label_eligible").to_list() == [False, False, True]
    assert result.get_column("label_exclusion_reason").to_list() == [
        "exchange_first_day", "resumption_day", None,
    ]


def test_universe_variants_change_membership_only_through_listing_age() -> None:
    universe = pl.DataFrame({
        "trade_date": [date(2026, 8, 14)] * 2,
        "asset_id": ["A", "B"], "board_id": ["MAIN", "MAIN"],
        "exchange_list_date": [date(2024, 1, 1), date(2010, 1, 1)],
        "days_since_exchange_list": [50, 130],
        "base_tradable_without_listing_age": [True, True],
    })
    derived = pl.DataFrame({
        "board_id": ["MAIN"], "regime_start": [date(2023, 4, 10)],
        "d_star": [46], "fallback_days": [None], "stable": [True],
    })
    d120 = apply_universe_variant(universe, universe_variant="d120")
    resolved = apply_universe_variant(
        universe, universe_variant="derived", derived_listing_window=derived
    )
    formal = apply_universe_variant(
        universe, universe_variant="frozen_d0",
        policy_path=PROJECT / "config" / "new_listing_policy_v1.json",
    )
    assert d120.get_column("is_variant_tradable").to_list() == [False, True]
    assert resolved.get_column("is_variant_tradable").to_list() == [True, True]
    assert formal.get_column("listing_age_threshold").to_list() == [20, 20]
    assert formal.get_column("is_variant_tradable").to_list() == [True, True]
    sensitivity = summarize_variant_sensitivity(
        universe.lazy(), derived_listing_window=derived
    )
    overall = sensitivity.filter(pl.col("board_id") == "ALL").row(0, named=True)
    assert overall["d120_eligible_rows"] == 1
    assert overall["derived_eligible_rows"] == 2
    assert overall["rows_added_by_derived"] == 1
    assert overall["rows_removed_by_derived"] == 0
