"""Artifact-only evaluation of the registered, non-deployable probe.

Signals alone are residualized. Labels and every placebo use the same raw
return channel. Gate dispositions come from the versioned evaluation contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..core.contracts import (
    CalculationRequest, CalculationResult, LoadedArtifact, OperationSpec,
    ParameterField, ParameterSchema, ParameterType, QualityCheck, QualityReport,
    QualityStatus, ScopeMode, Stage, TableOutput,
)
from ..l2.estimation_evaluation import (
    EvaluationContract, centered_peak_gate, permutation_null_gate, placebo_zero_gate,
    serial_autocorrelation, summarize_ic,
)
from ..l2.estimation_panel import _spearman, prepare_daily_ic_context
from ..l2.estimation_probe import _derangement


INPUTS = ("risk_exposure_matrix", "forward_labels", "tradable_universe", "probe_signal_panel")


def _one(inputs, kind):
    return next(iter(next(x for x in inputs if x.reference.key.artifact_type == kind).tables.values()))


def permutation_means(signal, labels, *, strategy, repetitions, seed, progress=None):
    """Permute finite cells only; preserve the asset/date axes and raw labels."""
    rng = np.random.default_rng(seed)
    source = labels.copy()
    groups = [np.flatnonzero(np.isfinite(source[:, a])) for a in range(source.shape[1])]
    if strategy == "within_asset_time_shuffle_demeaned":
        for a, ix in enumerate(groups):
            if ix.size:
                source[ix, a] -= source[ix, a].mean()
    values = []
    for repetition in range(repetitions):
        shuffled = np.full_like(source, np.nan)
        if strategy.startswith("within_asset"):
            for a, ix in enumerate(groups):
                shuffled[ix, a] = source[rng.permutation(ix), a]
        elif strategy == "within_date_cross_sectional_return_shuffle":
            for d in range(source.shape[0]):
                ix = np.flatnonzero(np.isfinite(source[d]))
                shuffled[d, ix] = source[d, rng.permutation(ix)]
        elif strategy == "date_axis_shuffle":
            shuffled = source[_derangement(rng, len(source))]
        else:
            raise ValueError(f"PROBE_PERMUTATION_UNKNOWN {strategy}")
        ics = []
        for s, y in zip(signal, shuffled):
            valid = np.isfinite(s) & np.isfinite(y)
            ic = _spearman(s[valid], y[valid])
            if ic is not None:
                ics.append(ic)
        if len(ics) < 2:
            raise ValueError("PROBE_PERMUTATION_INSUFFICIENT_SUPPORT")
        values.append(float(np.mean(ics)))
        if progress and (repetition + 1) % 10 == 0:
            progress(f"{strategy}: {repetition + 1}/{repetitions}")
    return values


def alignment_curve(signal, labels, offsets, *, side):
    # Every offset uses the same intersection of dates AND securities, so a
    # changed support set cannot manufacture a center peak.
    pairs = [(o if side == "signal" else 0, o if side == "label" else 0) for o in offsets]
    first = max(0, -min(min(s, y) for s, y in pairs))
    stop = min(len(signal), len(signal) - max(max(s, y) for s, y in pairs))
    daily = {o: [] for o in offsets}
    counts = []
    for d in range(first, stop):
        valid = np.ones(signal.shape[1], dtype=bool)
        for so, yo in pairs:
            valid &= np.isfinite(signal[d + so]) & np.isfinite(labels[d + yo])
        if valid.sum() < 3:
            continue
        metrics = [_spearman(signal[d + so, valid], labels[d + yo, valid]) for so, yo in pairs]
        if any(x is None for x in metrics):
            continue
        counts.append(int(valid.sum()))
        for o, ic in zip(offsets, metrics):
            daily[o].append(ic)
    if len(counts) < 2:
        raise ValueError("PROBE_ALIGNMENT_INSUFFICIENT_SUPPORT")
    points = {o: float(np.mean(daily[o])) for o in offsets}
    return {"curve": [{"offset": o, "mean_ic": points[o]} for o in offsets],
            "observations": len(counts), "median_n": float(np.median(counts)),
            "support": "common_date_asset_intersection_across_offsets", **centered_peak_gate(points)}


@dataclass(frozen=True)
class EvaluateProbe:
    contract_path: Path
    progress: Callable[[str], None] | None = None
    spec: OperationSpec = field(default_factory=lambda: OperationSpec(
        operation_id="evaluate_research_probe", version="1", stage=Stage.L2_PREDICTIVITY,
        input_artifact_types=INPUTS, input_artifact_versions={x: "1" for x in INPUTS},
        output_artifact_types=("factor_evaluation_panel",),
        output_artifact_versions={"factor_evaluation_panel": "1"},
        parameters=ParameterSchema(fields=tuple(ParameterField(
            name=n, type_id=ParameterType.STRING, required=True, description=n,
        ) for n in ("start_date", "end_date")), allow_extra=False),
        scope_mode=ScopeMode.UNIFIED_WITH_BOARD, accepts_benchmark_inputs=False,
        description="Evaluate artifact-backed research probe; never issue a formal Alpha assertion.",
    ))

    def calculate(self, request: CalculationRequest, inputs: tuple[LoadedArtifact, ...]):
        c = EvaluationContract.from_json(self.contract_path)
        if c.ic_neutralization_sides != "signal_only" or c.singular_value_rcond != 1e-10:
            raise ValueError("PROBE_EVALUATION_CHANNEL_UNSUPPORTED")
        if c.signal_id != "reversal_20d_v1" or c.risk_exposure_column_set_id != "risk_set_candidate_v1_minus_index_membership":
            raise ValueError("PROBE_EVALUATION_DEFINITION_UNSUPPORTED")
        start, end = request.parameters["start_date"], request.parameters["end_date"]
        if start < str(c.development_start) or end >= str(c.holdout_start) or start > end:
            raise ValueError("PROBE_EVALUATION_SAMPLE_BOUNDARY")
        exposure, signal, labels, universe = (_one(inputs, t) for t in
            ("risk_exposure_matrix", "probe_signal_panel", "forward_labels", "tradable_universe"))
        if signal["signal_id"].unique().to_list() != [c.signal_id]:
            raise ValueError("PROBE_SIGNAL_ID_MISMATCH")
        risk = [x for x in exposure.columns if x.startswith("risk_")
                and x not in {"risk_factor_set_id", "risk_factor_set_version"} and not x.startswith("risk_index_")]
        if not risk or "risk_size" not in risk or "risk_country" not in risk:
            raise ValueError("PROBE_RISK_COLUMNS_MISSING")
        window = pl.col("trade_date").is_between(pl.lit(start).str.to_date(), pl.lit(end).str.to_date())
        key = ["trade_date", "asset_id"]
        base = signal.filter(window).rename({"signal_value": "reversal_20d"})
        # L1 already applies its published universe variant (currently frozen
        # D0). The legacy universe's is_tradable is D120: applying it again
        # silently mixes policies and removes otherwise valid L1 members.
        base = base.join(universe.select(*key), on=key, validate="1:1")
        base = base.join(exposure.select(*key, *risk, "is_valid").filter(pl.col("is_valid")), on=key, validate="1:1")
        for h in c.reported_horizons:
            y = labels.filter(window & (pl.col("horizon_id") == f"h{h}d")).select(*key, pl.col("target_return").alias(f"forward_return_{h}"))
            base = base.join(y, on=key, how="left", validate="1:1")
        base = base.sort(key)
        if self.progress:
            self.progress(f"评估面板 {base.height} 行；按日计算中性化")
        ctx = prepare_daily_ic_context(base, risk)
        present = ctx.lookup >= 0
        signal_grid = np.full(ctx.lookup.shape, np.nan)
        signal_grid[present] = ctx.signal_residual[ctx.lookup[present]] * c.declared_direction
        # Do not compress a missing/invalid cross-section out of the market
        # calendar: one offset must always mean one market trading session.
        market_dates = universe.filter(window)["trade_date"].unique().sort().to_list()
        market_index = {d:i for i,d in enumerate(market_dates)}
        context_index = {d:i for i,d in enumerate(ctx.dates)}
        positions = [market_index[d] for d in ctx.dates]
        full_signal = np.full((len(market_dates), len(ctx.assets)), np.nan)
        full_signal[positions] = signal_grid
        signal_grid = full_signal
        label_grids = {}
        records, summaries = [], {}
        for h in c.reported_horizons:
            grid = np.full(ctx.lookup.shape, np.nan)
            grid[present] = base[f"forward_return_{h}"].to_numpy()[ctx.lookup[present]]
            full_grid = np.full_like(signal_grid, np.nan)
            full_grid[positions] = grid
            grid = full_grid
            label_grids[h] = grid
            daily = []
            for d, day in enumerate(market_dates):
                valid = np.isfinite(signal_grid[d]) & np.isfinite(grid[d])
                ic = _spearman(signal_grid[d, valid], grid[d, valid])
                cd = context_index.get(day)
                if ic is None or cd not in ctx.diagnostics:
                    continue
                daily.append(ic)
                records.append({"trade_date": day, "feature_id": c.signal_id, "horizon_days": h,
                    "evaluation_variant": "baseline", "variant_class": "main",
                    "neutralization_mode": "signal_only_equal_weight_ols",
                    "risk_set_version": c.risk_set_version, "attempt_id": None,
                    "sample_role": "development_probe", "holdout_touched": False,
                    "daily_ic": ic, "daily_rank_ic": ic, "cross_section_count": int(valid.sum()),
                    "n_valid": int(valid.sum()), "signal_trade_date": day, "exposure_trade_date": day,
                    **ctx.diagnostics[cd]})
            summaries[str(h)] = summarize_ic(daily, horizon_days=h)
        if self.progress:
            self.progress("IC 主面板完成；开始对齐曲线与 100 次置换检验")
        controls = {}
        cross_daily = {}
        for h in c.reported_horizons:
            rng = np.random.default_rng(c.placebo_seed)
            shuffled_ics = []
            for s, y in zip(signal_grid, label_grids[h]):
                valid = np.isfinite(s) & np.isfinite(y)
                ic = _spearman(s[valid], rng.permutation(y[valid]))
                if ic is not None:
                    shuffled_ics.append(ic)
            cross_daily[str(h)] = placebo_zero_gate(shuffled_ics, horizon_days=h,
                threshold_standard_errors=c.zero_threshold_standard_errors)
        # A prespecified primary horizon is used for the expensive permutation
        # distribution; alignment and main IC retain all registered horizons.
        for name in ("within_date_cross_sectional_return_shuffle", "within_asset_time_shuffle_raw",
                     "within_asset_time_shuffle_demeaned", "date_axis_shuffle"):
            result = permutation_null_gate(permutation_means(signal_grid, label_grids[c.primary_horizon],
                strategy=name, repetitions=c.time_shuffle_repetitions, seed=c.placebo_seed, progress=self.progress))
            controls[name] = {**result, "disposition": c.variant_dispositions[name], "horizon_days": c.primary_horizon}
        controls["within_date_cross_sectional_return_shuffle"].update(
            {"permutation_null_passed": controls["within_date_cross_sectional_return_shuffle"]["passed"],
             "by_horizon": cross_daily, "passed": all(v["passed"] for v in cross_daily.values()),
             "acceptance_rule": "absolute mean daily IC <= registered Newey-West standard-error bound"})
        for name, side, wide in (("label_window_shift", "label", False), ("raw_signal_shift", "signal", False),
                                 ("label_window_shift_wide", "label", True), ("raw_signal_shift_wide", "signal", True)):
            by_horizon = {}
            for h in c.reported_horizons:
                offsets = c.sensitivity_alignment_offsets(h, side) if wide else c.gate_alignment_offsets(h, side)
                by_horizon[str(h)] = alignment_curve(signal_grid, label_grids[h], offsets, side=side)
            controls[name] = {"disposition": c.variant_dispositions[name], "by_horizon": by_horizon,
                              "passed": all(v["passed"] for v in by_horizon.values())}
        # Shifted exposures are intentionally not an acceptance gate. Record
        # the contract's structural limitation instead of inventing evidence.
        controls["shifted_exposure"] = {"disposition": "diagnostic", "status": "not_recomputed",
            "reason": "adjacent exposures are persistent; signal/exposure date equality is asserted in daily panel"}
        gates = [x for x in controls.values() if x["disposition"] == "gate"]
        passed = bool(gates) and all(x.get("passed") is True for x in gates)
        frame = pl.DataFrame(records).with_columns(pl.col("attempt_id").cast(pl.String))
        frame = frame.with_columns(
            pl.lit(None,dtype=pl.Float64).alias("signal_turnover"),
            pl.lit(1.0).alias("baseline_n_valid_ratio"),
            pl.lit(0.0).alias("intersection_risk_size_median_delta"),
            (pl.col("n_valid")/pl.col("n")).alias("coverage"),
            pl.lit(1.0).alias("retained_fraction"),
            pl.lit(None,dtype=pl.String).alias("exclusion_reason"),
        )
        frame = frame.with_columns(pl.col("daily_rank_ic").rolling_mean(c.rolling_window_trading_days)
            .over("horizon_days").alias("rolling_ic"))
        return CalculationResult(outputs=(TableOutput(artifact_type="factor_evaluation_panel",
            tables={"factor_evaluation_panel_v1": frame}, metadata={
                "ic": {"by_horizon": summaries, "decay_curve": [{"horizon_trading_days": h, **summaries[str(h)]} for h in c.reported_horizons]},
                "negative_controls": {"passed": passed, "variants": controls, "disposition_source": "evaluation_channel.variant_dispositions",
                    "within_date_cross_sectional_return_shuffle": cross_daily,
                    "whole_cross_section_time_axis_shuffle": {str(c.primary_horizon): controls["date_axis_shuffle"]},
                    "label_window_shift_peak_at_zero": controls["label_window_shift"]["by_horizon"][str(c.primary_horizon)],
                    "raw_signal_shift_peak_at_zero": controls["raw_signal_shift"]["by_horizon"][str(c.primary_horizon)],
                    "within_asset_time_shuffle_100_repetition_asset_demeaned_null": controls["within_asset_time_shuffle_demeaned"]},
                "risk_columns": risk, "risk_set_status": "candidate", "deployable": False,
                "universe_policy_authority": "published_L1_members_intersect_universe_keys",
                "legacy_d120_filter_reapplied": False,
                "data_quality": {"panel_rows": base.height, "daily_rows": frame.height,
                    "dates": len(market_dates), "holdout_touched": False,
                    "valid_daily_cross_sections": frame.filter(pl.col("horizon_days")==c.primary_horizon).height,
                    "median_cross_section": float(frame["n_valid"].median()),
                    "primary_daily_ic_autocorrelation": serial_autocorrelation(
                        frame.filter(pl.col("horizon_days") == c.primary_horizon)["daily_rank_ic"].to_list(), lags=(1,5,20))},
            }),), quality=QualityReport(status=QualityStatus.PASSED, checks=(QualityCheck(
                check_id="probe_evaluation_computed", passed=True, detail="Research acceptance recorded separately from artifact integrity", observed=frame.height),)))
