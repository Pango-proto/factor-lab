"""Validated L1 risk-exposure input and identification contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CategoricalConstraintBlock:
    factor_id: str
    source_column: str
    required: bool


@dataclass(frozen=True)
class L1RiskExposureConfig:
    config_id: str
    regression_universe: str
    required_metadata: tuple[str, ...]
    regression_industry_column: str
    display_industry_column: str
    board_grouping_column: str
    constraint_blocks: tuple[CategoricalConstraintBlock, ...]
    minimum_category_members_source: str
    category_failure_policy: str
    development_universe_variants: tuple[str, ...]
    formal_universe_variants: tuple[str, ...]
    formal_publish_policy: str

    def __post_init__(self) -> None:
        if self.regression_universe != "all_a_share":
            raise ValueError("L1_PER_BOARD_REGRESSION_FORBIDDEN")
        required = set(self.required_metadata)
        expected = {
            "trade_date", "asset_id", "board_id", "sw_l1_code", "sw_l2_code",
            "exchange_list_date", "days_since_exchange_list",
        }
        if not expected <= required:
            raise ValueError("L1_REQUIRED_ROW_METADATA_INCOMPLETE")
        if self.regression_industry_column != "sw_l1_code":
            raise ValueError("L1_RISK_REGRESSION_REQUIRES_SW_L1")
        if self.display_industry_column != "sw_l2_code":
            raise ValueError("L1_SW_L2_MUST_BE_DISPLAY_ONLY")
        if self.board_grouping_column != "board_id":
            raise ValueError("L1_BOARD_GROUPING_METADATA_REQUIRED")
        if self.category_failure_policy != "mark_date_invalid":
            raise ValueError("L1_SMALL_CATEGORY_MUST_INVALIDATE_DATE")
        if set(self.development_universe_variants) != {"d120", "derived"}:
            raise ValueError("L1_DEVELOPMENT_UNIVERSE_VARIANTS_INVALID")
        if self.formal_universe_variants != ("frozen_d0",):
            raise ValueError("L1_FORMAL_UNIVERSE_VARIANT_INVALID")
        if self.formal_publish_policy != (
            "new_listing_policy_v1.formal_exposure_publish_gate"
        ):
            raise ValueError("L1_FORMAL_PUBLISH_POLICY_INVALID")


def load_l1_risk_exposure_config(path: Path) -> L1RiskExposureConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("L1_RISK_EXPOSURE_CONFIG_SCHEMA_UNSUPPORTED")
    if payload.get("per_board_regression") != "forbidden":
        raise ValueError("L1_PER_BOARD_REGRESSION_OPTION_FORBIDDEN")
    identification = payload["identification"]
    if (
        identification["expected_deficiency"] != "daily_rank_of_constraint_matrix"
        or not identification["require_constraint_rows_independent"]
        or identification["null_space_basis"]
        != "blockwise_qr_then_block_diagonal_assembly"
    ):
        raise ValueError("L1_CONSTRAINT_IDENTIFICATION_POLICY_INVALID")
    blocks = tuple(
        CategoricalConstraintBlock(
            factor_id=item["factor_id"],
            source_column=item["source_column"],
            required=bool(item["required"]),
        )
        for item in payload["categorical_constraint_blocks"]
    )
    metadata = payload["row_metadata"]
    minimum = payload["minimum_category_members"]
    variants = payload["universe_variants"]
    style = payload["style_pipeline"]
    if (
        style.get("orthogonalization_metric_source")
        != "weight_metric_contract_v1.exposure_orthogonalization_weight"
        or style.get("must_bind_regression_base_weight") is not True
    ):
        raise ValueError("L1_WEIGHT_METRIC_BINDING_INVALID")
    return L1RiskExposureConfig(
        config_id=payload["config_id"],
        regression_universe=payload["regression_universe"],
        required_metadata=tuple(metadata["required"]),
        regression_industry_column=metadata["risk_regression_industry_column"],
        display_industry_column=metadata["display_only_industry_column"],
        board_grouping_column=metadata["board_grouping_column"],
        constraint_blocks=blocks,
        minimum_category_members_source=minimum["source"],
        category_failure_policy=minimum["failure_policy"],
        development_universe_variants=tuple(variants["allowed_for_development"]),
        formal_universe_variants=tuple(variants["allowed_for_formal_publish"]),
        formal_publish_policy=variants["formal_publish_policy"],
    )


def assert_exposure_publish_allowed(
    *, policy_path: Path, universe_variant: str, update_current_pointer: bool
) -> None:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    variants = payload.get("universe_variants", {})
    if universe_variant not in variants:
        raise ValueError(f"UNIVERSE_VARIANT_UNKNOWN variant={universe_variant}")
    gate = payload["formal_exposure_publish_gate"]
    d0 = payload["d0_model_exclusion"]
    if update_current_pointer and d0["status"] != gate["requires_d0_status"]:
        raise RuntimeError("FORMAL_EXPOSURE_PUBLISH_REQUIRES_FROZEN_D0")
    if update_current_pointer and universe_variant != gate["formal_universe_variant"]:
        raise RuntimeError("FORMAL_EXPOSURE_PUBLISH_REQUIRES_FROZEN_D0_VARIANT")
