#!/usr/bin/env python3
"""Evaluate T3 next-day execution filters on the sealed estimation interval."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factor_matrix.calculation.l2.estimation_evaluation import (
    EvaluationContract,
    summarize_ic,
)
from factor_matrix.calculation.l2.estimation_panel import (
    DEFAULT_CONTRACT,
    DEFAULT_END,
    DEFAULT_START,
    _load_panel,
    _residualize,
    _signal_column,
    _spearman,
    attach_next_day_execution_eligibility,
)


DEFAULT_OUTPUT = Path(
    "data/diagnostics/evaluation_t3_execution_filter/"
    "reversal_20d_t3_execution_filter_20260818.json"
)
LIQUIDITY_QUANTILES = (0.10, 0.20, 0.30)


def _quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        key: float(np.quantile(array, probability))
        for key, probability in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
    }


def run(
    data_root: Path,
    *,
    start: date,
    end: date,
    contract_path: Path,
) -> dict[str, Any]:
    contract = EvaluationContract.from_json(contract_path)
    if end >= contract.holdout_start:
        raise ValueError("EVALUATION_T3_HOLDOUT_LOCKED")
    panel, risk_columns, safe_end = _load_panel(data_root, start, end, contract)
    panel = attach_next_day_execution_eligibility(
        panel,
        data_root,
        primary_horizon=contract.primary_horizon,
        liquidity_floor_quantile=0.20,
    )
    signal_column = _signal_column(contract.signal_id)
    daily: list[dict[str, Any]] = []
    for frame in panel.partition_by("trade_date", maintain_order=True):
        base = frame.filter(
            pl.col(signal_column).is_finite()
            & pl.all_horizontal([
                pl.col(f"forward_return_{horizon}").is_finite()
                for horizon in contract.reported_horizons
            ])
            & pl.all_horizontal([
                pl.col(column).is_finite() for column in risk_columns
            ])
        )
        if base.height < max(50, len(risk_columns) + 5):
            continue
        row: dict[str, Any] = {
            "trade_date": str(base["trade_date"][0]),
            "baseline_n": base.height,
        }
        for threshold in (None, *LIQUIDITY_QUANTILES):
            if threshold is None:
                selected = base
                arm = "baseline"
            else:
                arm = f"q{int(threshold * 100):02d}"
                selected = base.filter(
                    pl.col("execution_can_buy").fill_null(False)
                    & pl.col("execution_can_sell").fill_null(False)
                    & (
                        pl.col("signal_date_amount_percentile") >= threshold
                    ).fill_null(False)
                )
            if selected.height < max(50, len(risk_columns) + 5):
                break
            row[f"{arm}_n"] = selected.height
            exposures = selected.select(risk_columns).to_numpy()
            signal_residual, _, _ = _residualize(
                selected[signal_column].to_numpy(), exposures
            )
            for horizon in contract.reported_horizons:
                label_residual, _, _ = _residualize(
                    selected[f"forward_return_{horizon}"].to_numpy(), exposures
                )
                value = _spearman(signal_residual, label_residual)
                if value is None:
                    break
                row[f"{arm}_ic_{horizon}"] = value
            else:
                continue
            break
        else:
            daily.append(row)

    if len(daily) < 2:
        raise ValueError("EVALUATION_T3_NO_VALID_DAILY_SERIES")
    daily_frame = pl.DataFrame(daily)
    arms: dict[str, Any] = {}
    for arm in ("baseline", "q10", "q20", "q30"):
        by_horizon = {
            str(horizon): summarize_ic(
                daily_frame[f"{arm}_ic_{horizon}"].to_list(),
                horizon_days=horizon,
            )
            for horizon in contract.reported_horizons
        }
        arms[arm] = {
            "liquidity_floor_quantile": None if arm == "baseline" else int(arm[1:]) / 100,
            "daily_cross_section_quantiles": _quantiles(
                daily_frame[f"{arm}_n"].to_list()
            ),
            "retained_cross_section_ratio_median": float(
                (daily_frame[f"{arm}_n"] / daily_frame["baseline_n"]).median()
            ),
            "ic_by_horizon": by_horizon,
        }

    reason_counts = {
        "panel_rows": panel.height,
        "missing_next_day_universe_fact": panel.filter(
            pl.col("listed_as_of").is_null()
        ).height,
        "next_day_st": panel.filter(pl.col("is_st").fill_null(False)).height,
        "next_day_suspended": panel.filter(
            pl.col("is_suspended").fill_null(False)
        ).height,
        "below_board_regime_d_star": panel.filter(
            pl.col("execution_d_star").is_not_null()
            & (
                pl.col("days_since_exchange_list") < pl.col("execution_d_star")
            )
        ).height,
        "next_day_one_word_limit_up": panel.filter(
            pl.col("execution_one_word_limit_up").fill_null(False)
        ).height,
        "next_day_one_word_limit_down": panel.filter(
            pl.col("execution_one_word_limit_down").fill_null(False)
        ).height,
        "below_signal_day_amount_q20": panel.filter(
            (pl.col("signal_date_amount_percentile") < 0.20).fill_null(True)
        ).height,
        "final_q20_execution_eligible": panel.filter(
            pl.col("execution_eligible")
        ).height,
    }
    primary = str(contract.primary_horizon)
    return {
        "schema_version": 1,
        "diagnostic_id": "reversal_20d_t3_execution_filter_v1",
        "status": "completed",
        "scope": {
            "start": str(start),
            "end": str(end),
            "safe_feature_end": str(safe_end),
            "holdout_start": str(contract.holdout_start),
            "holdout_opened": False,
            "signal_id": contract.signal_id,
            "valid_daily_cross_sections": daily_frame.height,
        },
        "policy": {
            "execution_date": "next unified market trading day after signal date",
            "buy_block": "next-day one-word limit-up",
            "sell_block": "next-day one-word limit-down",
            "exclude_next_day_st": True,
            "exclude_next_day_suspended": True,
            "listing_age": "new_listing_policy_v1.d_star_reference by board and regime",
            "d_star": {
                "MAIN_pre_registration": 75,
                "MAIN_post_registration": 46,
                "CHINEXT_pre_registration": 75,
                "CHINEXT_post_registration": 43,
                "STAR": 35,
                "BSE": 25,
            },
            "liquidity": "signal-day amount cross-sectional percentile; q20 primary",
            "liquidity_lookahead_used": False,
        },
        "exclusion_counts_nonexclusive": reason_counts,
        "arms": arms,
        "primary_q20_effect": {
            "horizon_trading_days": contract.primary_horizon,
            "baseline_mean_ic": arms["baseline"]["ic_by_horizon"][primary]["mean"],
            "filtered_mean_ic": arms["q20"]["ic_by_horizon"][primary]["mean"],
            "filtered_ic_ir": arms["q20"]["ic_by_horizon"][primary]["ic_ir"],
            "filtered_newey_west_t": arms["q20"]["ic_by_horizon"][primary][
                "newey_west_t"
            ],
            "retained_cross_section_ratio_median": arms["q20"][
                "retained_cross_section_ratio_median"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.data_root,
        start=args.start,
        end=args.end,
        contract_path=args.contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "primary_q20_effect": result["primary_q20_effect"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
