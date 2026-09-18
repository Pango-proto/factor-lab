"""Frozen research choices, label-free parameter derivation, and sample labels."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist
from typing import Mapping

from .storage import json_hash


FORBIDDEN_DERIVATION_INPUTS = frozenset({
    "return", "returns", "forward_return", "realized_return", "ic", "icir",
    "factor_predictivity", "fm_regression", "l2a",
})


@dataclass(frozen=True)
class ResearchProtocol:
    protocol_id: str
    targets: Mapping[str, float]
    frozen_choices: Mapping[str, object]
    external_facts_config: str

    @classmethod
    def load(cls, path: Path) -> "ResearchProtocol":
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = cls(payload["protocol_id"], payload["targets"],
                       payload["frozen_choices"], payload["external_facts"]["config_path"])
        protocol.validate()
        return protocol

    def validate(self) -> None:
        holdout = self.frozen_choices["holdout"]
        if not holdout["sealed"] or holdout["opened"]:
            raise ValueError("HOLDOUT_MUST_BE_SEALED_AND_UNOPENED")
        if "horizon_days" in self.targets:
            raise ValueError("EVALUATION_HORIZON_MUST_NOT_BE_DERIVED")

    @property
    def holdout_start(self) -> date:
        return date.fromisoformat(self.frozen_choices["holdout"]["start"])

    def frozen_horizon(self, factor_params: Mapping[str, object]) -> int:
        horizon = factor_params.get("evaluation_horizon_days")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("FROZEN_EVALUATION_HORIZON_REQUIRED")
        return horizon

    def derive(self, *, factor_count: int, maximum_evaluation_horizon_days: int,
               cross_section_size: int, available_estimation_days: int) -> dict[str, object]:
        if min(factor_count, maximum_evaluation_horizon_days, cross_section_size,
               available_estimation_days) < 1:
            raise ValueError("DERIVED_PARAMETER_INPUT_INVALID")
        derivation_inputs = {
            "factor_count": factor_count,
            "maximum_evaluation_horizon_days": maximum_evaluation_horizon_days,
            "cross_section_size": cross_section_size,
            "available_estimation_days": available_estimation_days,
        }
        assert_label_free_derivation_inputs(derivation_inputs)
        annual_days = int(self.targets["trading_days_per_year"])
        burn_in = max(
            math.ceil(self.targets["burn_in_horizon_factor_multiplier"]
                      * factor_count * maximum_evaluation_horizon_days),
            math.ceil(self.targets["minimum_annual_cycles"] * annual_days),
        )
        correlation_ess = self.targets["ewma_correlation_effective_obs_per_factor"] * factor_count
        variance_ess = self.targets["ewma_variance_effective_obs_per_factor"] * factor_count
        tail = self.targets["target_two_sided_tail_probability"]
        derived = {
            "burn_in_days": burn_in,
            "ewma_factor_covariance_half_life_days": _half_life_from_effective_size(correlation_ess),
            "ewma_specific_variance_half_life_days": _half_life_from_effective_size(variance_ess),
            "mad_k": NormalDist().inv_cdf(1 - tail / 2) / NormalDist().inv_cdf(0.75),
            "outlier_sigma": NormalDist().inv_cdf(1 - tail / 2),
            "coverage_min": 1 / self.targets["maximum_missingness_variance_inflation"],
            "minimum_category_members": max(
                int(self.targets["minimum_category_members_floor"]),
                math.ceil(
                    self.targets["minimum_category_cross_section_fraction"]
                    * cross_section_size
                ),
            ),
            "maximum_condition_number": self.targets["target_relative_numerical_error"]
                                        / float.fromhex("0x1.0000000000000p-52"),
            "huber_delta": _huber_c(self.targets["huber_target_normal_efficiency"]),
            "maximum_invalid_date_ratio": self.targets[
                "maximum_post_burn_invalid_date_ratio"
            ],
            "unresolved_required": ["new_listing_days_by_regime"],
            "deferred_optional": ["statistical_factor_count_mp"],
        }
        inputs = derivation_inputs
        return {"protocol_id": self.protocol_id, "protocol_sha": json_hash({
            "targets": dict(self.targets), "frozen_choices": dict(self.frozen_choices)}),
            "inputs": inputs, "values": derived}


def assert_label_free_derivation_inputs(inputs: Mapping[str, object]) -> None:
    offenders = sorted(key for key in inputs
                       if any(token in key.lower() for token in FORBIDDEN_DERIVATION_INPUTS))
    if offenders:
        raise ValueError(f"PARAMETER_DERIVATION_USES_RETURN_LABELS keys={','.join(offenders)}")


def _half_life_from_effective_size(effective_size: float) -> float:
    if effective_size <= 1:
        raise ValueError("EWMA_EFFECTIVE_SIZE_TOO_SMALL")
    decay = (effective_size - 1) / (effective_size + 1)
    return math.log(0.5) / math.log(decay)


def _huber_efficiency(c: float) -> float:
    normal = NormalDist()
    phi = math.exp(-0.5 * c * c) / math.sqrt(2 * math.pi)
    inside = 2 * normal.cdf(c) - 1
    second = 2 * ((normal.cdf(c) - 0.5) - c * phi) + 2 * c * c * (1 - normal.cdf(c))
    return inside * inside / second


def _huber_c(target_efficiency: float) -> float:
    if not 0 < target_efficiency < 1:
        raise ValueError("HUBER_TARGET_EFFICIENCY_INVALID")
    low, high = 0.05, 10.0
    for _ in range(100):
        middle = (low + high) / 2
        if _huber_efficiency(middle) < target_efficiency:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def label_sample_date(value: date, protocol: ResearchProtocol, *, burn_in_end: date) -> dict[str, object]:
    frozen = protocol.frozen_choices
    backfill_start = date.fromisoformat(frozen["estimation_start"])
    display_start = date.fromisoformat(frozen["backtest_display_start"])
    stress_start = date.fromisoformat(frozen["stress_test"]["start"])
    stress_end = date.fromisoformat(frozen["stress_test"]["end"])
    if stress_start <= value <= stress_end:
        role = "out_of_model"
    elif value < backfill_start:
        role = "excluded_pre_modern"
    elif value <= burn_in_end:
        role = "burn_in"
    elif value < protocol.holdout_start:
        role = "research"
    else:
        role = "holdout"
    return {
        "sample_role": role,
        "risk_estimation_eligible": value >= backfill_start,
        "alpha_estimation_eligible": role == "research",
        "backtest_display_eligible": value >= display_start,
        "stress_only": role == "out_of_model",
    }


def build_sample_calendar_rows(
    trading_dates: list[date], protocol: ResearchProtocol,
    derived_snapshot: Mapping[str, object],
) -> list[dict[str, object]]:
    dates = sorted(set(trading_dates))
    estimation_start = date.fromisoformat(protocol.frozen_choices["estimation_start"])
    eligible = [value for value in dates if value >= estimation_start]
    burn_in_days = int(derived_snapshot["values"]["burn_in_days"])
    if len(eligible) <= burn_in_days:
        raise ValueError("SAMPLE_CALENDAR_BURN_IN_NOT_COVERED")
    burn_in_end = eligible[burn_in_days - 1]
    snapshot_id = f"sample_{derived_snapshot['protocol_sha'][:16]}"
    return [
        {"trade_date": value, "snapshot_id": snapshot_id,
         **label_sample_date(value, protocol, burn_in_end=burn_in_end)}
        for value in dates
    ]


class ProtocolSealStore:
    """Persist the holdout boundary once; later config changes fail closed."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def seal(self, protocol: ResearchProtocol) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS research_protocol_seal (
                protocol_id TEXT PRIMARY KEY, holdout_start TEXT NOT NULL,
                sealed_at TEXT NOT NULL, opened INTEGER NOT NULL CHECK(opened=0),
                protocol_sha TEXT NOT NULL)""")
            holdout = protocol.frozen_choices["holdout"]
            row = (protocol.protocol_id, holdout["start"], holdout["sealed_at"], 0,
                   json_hash({"targets": dict(protocol.targets),
                              "frozen_choices": dict(protocol.frozen_choices)}))
            existing = connection.execute(
                "SELECT holdout_start, sealed_at, opened FROM research_protocol_seal WHERE protocol_id=?",
                (protocol.protocol_id,),).fetchone()
            if existing and existing != row[1:4]:
                raise RuntimeError("HOLDOUT_SEAL_IMMUTABLE_CONFLICT")
            connection.execute("INSERT OR IGNORE INTO research_protocol_seal VALUES (?,?,?,?,?)", row)
