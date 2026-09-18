"""Frozen event-return eligibility rules; L2 remains the only label consumer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl


DEFAULT_PATH = Path(__file__).resolve().parents[4] / "config" / "return_label_policy_v1.json"


def resolve_return_label_policy_path(
    data_root: Path, explicit_path: Path | None = None,
) -> Path:
    """Resolve one policy for both return materialization and forward labels.

    An explicit path is authoritative, including when missing or malformed;
    never fall through to a different policy after selecting a source.
    """
    if explicit_path is not None:
        return explicit_path
    for candidate in (
        data_root.parent / "config" / DEFAULT_PATH.name,
        data_root / "config" / DEFAULT_PATH.name,
    ):
        if candidate.exists():
            return candidate
    return DEFAULT_PATH


@dataclass(frozen=True)
class ReturnLabelPolicy:
    policy_id: str
    regression_universe: str
    exclude_exchange_first_day: bool
    exclude_resumption_day: bool
    l2a_forward_window_event_policy: str

    def __post_init__(self) -> None:
        if self.regression_universe != "all_a_share":
            raise ValueError("L2_PER_BOARD_REGRESSION_FORBIDDEN")
        if not self.exclude_exchange_first_day or not self.exclude_resumption_day:
            raise ValueError("L2_EVENT_RETURN_EXCLUSION_MUST_BE_FROZEN")
        if self.l2a_forward_window_event_policy != (
            "exclude_window_if_exchange_first_day_or_resumption_occurs"
        ):
            raise ValueError("L2A_EVENT_WINDOW_POLICY_INVALID")


def load_return_label_policy(path: Path = DEFAULT_PATH) -> ReturnLabelPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("RETURN_LABEL_POLICY_SCHEMA_UNSUPPORTED")
    if payload.get("per_board_regression") != "forbidden":
        raise ValueError("L2_PER_BOARD_REGRESSION_OPTION_FORBIDDEN")
    l2b = payload["l2b"]
    return ReturnLabelPolicy(
        policy_id=payload["policy_id"],
        regression_universe=payload["regression_universe"],
        exclude_exchange_first_day=l2b["exchange_first_day"]["action"] == "exclude",
        exclude_resumption_day=l2b["resumption_day"]["action"] == "exclude",
        l2a_forward_window_event_policy=payload["l2a"]["forward_window_event_policy"],
    )


def apply_l2b_label_policy(
    frame: pl.DataFrame, policy: ReturnLabelPolicy, *, label_date_column: str = "trade_date"
) -> pl.DataFrame:
    required = {
        label_date_column, "exchange_list_date", "return_source", "total_return",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"L2_LABEL_POLICY_COLUMNS_MISSING columns={','.join(missing)}")
    exchange_first = pl.col(label_date_column) == pl.col("exchange_list_date")
    resumption = pl.col("return_source") == "resumption"
    eligible = pl.col("total_return").is_finite() & ~exchange_first & ~resumption
    reason = (
        pl.when(exchange_first).then(pl.lit("exchange_first_day"))
        .when(resumption).then(pl.lit("resumption_day"))
        .when(pl.col("total_return").is_null() | ~pl.col("total_return").is_finite())
        .then(pl.lit("invalid_return"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    return frame.with_columns(
        eligible.alias("label_eligible"), reason.alias("label_exclusion_reason")
    )
