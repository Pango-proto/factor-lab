"""Independent audit of the two blocked M1 permutation diagnostics.

This service reads a pinned run, writes only diagnostic artifacts, and never
changes gate dispositions, thresholds, CURRENT pointers, or research attempts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from ..l2.estimation_panel import _residualize
from ..publishing.local import _write_immutable_parquet

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT / "config/permutation_gate_diagnostics_v1.json"
VARIANTS = ("within_asset_time_shuffle_demeaned", "date_axis_shuffle")


def normalized_ranks(values):
    """Independent rank implementation: Polars average ranks, then L2 norm."""
    ranks = pl.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()
    ranks = ranks - ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm > 0 else None


def rank_correlation(left, right):
    if len(left) < 3:
        return None
    a, b = normalized_ranks(left), normalized_ranks(right)
    return None if a is None or b is None else float(a @ b)


def draw_assignment(source, groups, rng, strategy):
    """Return donor-date indices; missing cells remain -1.

    Date-axis rejection sampling is implemented independently, with no call to
    the production derangement or permutation_means helpers.
    """
    n_dates, n_assets = source.shape
    assignment = np.full(source.shape, -1, dtype=np.int32)
    if strategy == VARIANTS[0]:
        for a, indices in enumerate(groups):
            positions = rng.permutation(len(indices))
            assignment[indices, a] = indices[positions]
    elif strategy == VARIANTS[1]:
        identity = np.arange(n_dates)
        for _ in range(1000):
            order = rng.permutation(n_dates)
            if not np.any(order == identity):
                break
        else:
            raise ValueError("DIAGNOSTIC_DERANGEMENT_REJECTION_EXHAUSTED")
        assignment[:] = np.broadcast_to(order[:, None], (n_dates, n_assets))
        assignment[~np.isfinite(source[order])] = -1
    else:
        raise ValueError("DIAGNOSTIC_VARIANT_UNSUPPORTED")
    return assignment


def zero_center_counterexamples():
    # Exhaustive enumeration, no random seeds or searched sample windows.
    signal = np.array([1., -1., 0.])
    histories = ([-1., -1., 2.], [-2., 1., 1.], [0., 0., 0.])
    draws = list(itertools.product(*histories))
    rank_mean = float(np.mean([rank_correlation(signal, np.array(y)) for y in draws]))
    covariance_mean = float(np.mean([np.mean(signal * np.array(y)) for y in draws]))
    persistent = np.array([-1., 0., 1.])
    return {
        "within_asset_arithmetic_demeaning": {
            "histories": histories, "signal": signal.tolist(), "enumerated_draws": len(draws),
            "asset_arithmetic_means": [float(np.mean(h)) for h in histories],
            "expected_unnormalized_covariance": covariance_mean,
            "expected_spearman": rank_mean, "exact_expected_spearman": "-1/9",
            "meaning": "Zero arithmetic donor means do not imply a zero Spearman permutation center.",
        },
        "date_axis_persistent_asset_order": {
            "signal_each_date": persistent.tolist(),
            "labels": [persistent.tolist(), (persistent + 10).tolist()],
            "only_derangement": [1, 0], "mean_spearman_after_swap": 1.0,
            "meaning": "Changing the date does not destroy persistent cross-asset rank structure.",
        },
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_fixed_inputs(lake, config):
    protected = {PROJECT / p: sha for p, sha in config["fixed_files"].items()}
    manifest_path = lake.root / config["source_manifest"]
    protected[manifest_path] = config["source_manifest_sha256"]
    for path, expected in protected.items():
        if file_sha256(path) != expected:
            raise ValueError(f"DIAGNOSTIC_PIN_MISMATCH {path}")
    manifest = _read_json(manifest_path)
    if manifest["run_id"] != config["source_run_id"] or manifest["holdout_touched"]:
        raise ValueError("DIAGNOSTIC_SOURCE_RUN_OR_HOLDOUT_MISMATCH")
    if any(manifest["definition"][k] != config[k] for k in ("start", "end", "label_end")):
        raise ValueError("DIAGNOSTIC_SAMPLE_CHANGED")
    if not config["start"] <= config["end"] < config["label_end"] < config["holdout_start"]:
        raise ValueError("DIAGNOSTIC_SAMPLE_BOUNDARY_INVALID")
    artifacts = {}
    for record in manifest["artifacts"]:
        path = lake.root / record["uri"]
        protected[path] = record["checksum"]
        if file_sha256(path) != record["checksum"]:
            raise ValueError("DIAGNOSTIC_ARTIFACT_METADATA_MISMATCH")
        metadata = _read_json(path)
        if metadata["quality_status"] != "passed":
            raise ValueError("DIAGNOSTIC_ARTIFACT_NOT_PASSED")
        artifacts[record["artifact_type"]] = metadata
        for table in metadata["tables"].values():
            tp = lake.root / table["uri"]
            if file_sha256(tp) != table["checksum"]:
                raise ValueError("DIAGNOSTIC_ARTIFACT_TABLE_MISMATCH")
            protected[tp] = table["checksum"]
    summary_record = manifest["outputs"]["summary"]
    summary_path = lake.root / summary_record["path"]
    if file_sha256(summary_path) != summary_record["sha256"]:
        raise ValueError("DIAGNOSTIC_SUMMARY_CHECKSUM_MISMATCH")
    protected[summary_path] = summary_record["sha256"]
    summary = _read_json(summary_path)
    evaluation_path = PROJECT / "config/evaluation_framework_v1.json"
    evaluation = _read_json(evaluation_path)
    if (evaluation["negative_controls"]["seed"] != config["seed"]
        or evaluation["negative_controls"]["time_shuffle_repetitions"] != config["repetitions"]
        or evaluation["sample"]["primary_horizon"] != config["primary_horizon"]
        or evaluation["evaluation_channel"]["signal_lookback_trading_days"] != config["lookback"]
        or evaluation["evaluation_channel"]["ic_neutralization_sides"] != "signal_only"):
        raise ValueError("DIAGNOSTIC_CONTRACT_CHANGED")
    return artifacts, summary, protected


def reconstruct_grids(lake, artifacts, summary, config, progress):
    """Rebuild the keyed primary panel without using the production context."""
    def scan(kind):
        record, = artifacts[kind]["tables"].values()
        return pl.scan_parquet(lake.root / record["uri"])
    low, high = config["start"], config["end"]
    window = pl.col("trade_date").is_between(pl.lit(low).str.to_date(), pl.lit(high).str.to_date())
    key = ["trade_date", "asset_id"]
    risk = artifacts["factor_evaluation_panel"]["metadata"]["risk_columns"]
    universe = scan("tradable_universe").filter(window).select(key)
    dates = universe.select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list()
    panel = (scan("risk_exposure_matrix").filter(window & pl.col("is_valid")).select(*key, *risk)
        .join(universe, on=key, how="inner", validate="1:1")
        .join(scan("probe_signal_panel").filter(window).select(*key, "signal_value"), on=key, validate="1:1")
        .join(scan("forward_labels").filter(window & (pl.col("horizon_id") == f"h{config['primary_horizon']}d"))
            .select(*key, "target_return"), on=key, how="left", validate="1:1")
        .sort(key).collect())
    if panel.height != summary["data_quality"]["panel_rows"]:
        raise ValueError("DIAGNOSTIC_PANEL_ROW_MISMATCH")
    # Production axes follow first appearance in the ordered panel, not an
    # alphabetic union: preserving that order is essential to replay RNG draws.
    assets = panel["asset_id"].unique(maintain_order=True).to_list()
    asset_index = {a:i for i,a in enumerate(assets)}
    date_index = {d:i for i,d in enumerate(dates)}
    signal = np.full((len(dates), len(assets)), np.nan)
    labels = np.full_like(signal, np.nan)
    saved_daily = (scan("factor_evaluation_panel").filter(pl.col("horizon_days") == config["primary_horizon"])
        .select("trade_date", "daily_rank_ic", "n_valid").collect())
    saved = {row["trade_date"]: row for row in saved_daily.to_dicts()}
    numerical = []
    for frame in panel.partition_by("trade_date", maintain_order=True):
        day = frame["trade_date"][0]; d = date_index[day]
        ai = np.array([asset_index[a] for a in frame["asset_id"].to_list()])
        raw = frame["signal_value"].to_numpy()
        labels[d, ai] = frame["target_return"].to_numpy()
        x = frame.select(risk).to_numpy()
        x = x[:, np.any(np.isfinite(x), axis=0)]
        valid = np.isfinite(raw) & np.all(np.isfinite(x), axis=1)
        if valid.sum() < max(3, x.shape[1]+2):
            continue
        design, values = x[valid], raw[valid]
        # Independent full-column least squares includes country and all
        # physical dummies; production drops constants before pseudoinversion.
        beta, _, rank, _ = np.linalg.lstsq(design, values, rcond=1e-10)
        residual = values - design @ beta
        production_residual = _residualize(values, design)[0]
        signal[d, ai[valid]] = residual
        joint = np.isfinite(signal[d]) & np.isfinite(labels[d])
        ic = rank_correlation(signal[d, joint], labels[d, joint])
        if day not in saved or ic is None:
            raise ValueError("DIAGNOSTIC_DAILY_SUPPORT_MISMATCH")
        numerical.append({"trade_date":day, "n_valid":int(joint.sum()), "saved_n_valid":saved[day]["n_valid"],
            "lstsq_rank":int(rank), "max_residual_absolute_difference":float(np.max(np.abs(residual-production_residual))),
            "independent_ic":ic, "saved_ic":saved[day]["daily_rank_ic"],
            "ic_absolute_difference":abs(ic-saved[day]["daily_rank_ic"])})
    numerical_frame = pl.DataFrame(numerical)
    passed = (numerical_frame.height == len(saved)
        and numerical_frame["ic_absolute_difference"].max() <= config["replay_absolute_tolerance"]
        and numerical_frame["max_residual_absolute_difference"].max() <= config["replay_absolute_tolerance"]
        and numerical_frame.filter(pl.col("n_valid") != pl.col("saved_n_valid")).is_empty())
    if not passed:
        raise ValueError("DIAGNOSTIC_INDEPENDENT_NEUTRALIZATION_OR_BASELINE_MISMATCH")
    progress(f"独立 OLS / 秩相关复算通过：{len(dates)} 日，{len(assets)} 只股票，{panel.height} 行")
    return signal, labels, dates, assets, numerical_frame


def replay_diagnostics(signal, labels, *, strategy, config, dates, assets, progress):
    finite_s = np.isfinite(signal)
    base_valid = finite_s & np.isfinite(labels)
    base_unit = np.zeros_like(signal)
    for d in range(len(dates)):
        valid = finite_s[d]
        ranks = normalized_ranks(signal[d, valid])
        if ranks is not None:
            base_unit[d, valid] = ranks
    counts = finite_s.sum(axis=0)
    asset_signal_mean = np.divide(base_unit.sum(axis=0), counts, out=np.zeros(len(assets)), where=counts>0)
    groups = [np.flatnonzero(np.isfinite(labels[:, a])) for a in range(len(assets))]
    source = labels.copy()
    source_means = np.full(len(assets), np.nan)
    for a, ix in enumerate(groups):
        if len(ix):
            source_means[a] = np.mean(source[ix,a])
            if strategy == VARIANTS[0]:
                source[ix,a] -= source_means[a]
    rng = np.random.default_rng(config["seed"])
    daily_records, repetition_records = [], []
    label_unit_sum = np.zeros(len(assets)); recipient_counts = np.zeros(len(assets), dtype=np.int64)
    asset_ic_sum = np.zeros(len(assets))
    lo = -(config["lookback"] + config["primary_horizon"] - 1)
    for repetition in range(config["repetitions"]):
        assignment = draw_assignment(source, groups, rng, strategy)
        # Every value is addressed by (source date, unchanged asset id).
        ystar = np.take_along_axis(source, np.maximum(assignment,0), axis=0)
        ystar[assignment < 0] = np.nan
        if strategy == VARIANTS[0] and not np.array_equal(np.isfinite(ystar), np.isfinite(labels)):
            raise ValueError("DIAGNOSTIC_TIME_SHUFFLE_MASK_CHANGED")
        days = []
        for d, day in enumerate(dates):
            valid = finite_s[d] & np.isfinite(ystar[d])
            indices = np.flatnonzero(valid)
            if len(indices)<3:
                continue
            su = normalized_ranks(signal[d, indices]); yu = normalized_ranks(ystar[d, indices])
            if su is None or yu is None:
                continue
            origins = assignment[d,indices]
            offsets = origins - d
            if strategy == VARIANTS[1] and np.any(offsets==0):
                raise ValueError("DIAGNOSTIC_DATE_DERANGEMENT_FIXED_POINT")
            overlap = (offsets>=lo) & (offsets<=-1)
            same_date = offsets==0
            cell_ic = su*yu
            persistent = float(asset_signal_mean[indices] @ yu)
            temporal = float((base_unit[d,indices]-asset_signal_mean[indices]) @ yu)
            support = float((su-base_unit[d,indices]) @ yu)
            centered_s = signal[d,indices]-np.mean(signal[d,indices])
            centered_y = ystar[d,indices]-np.mean(ystar[d,indices])
            row = {"variant":strategy,"repetition":repetition+1,"trade_date":day,"n_valid":len(indices),
                "ic":float(np.sum(cell_ic)),"persistent_asset_component":persistent,
                "time_varying_component":temporal,"support_reranking_component":support,
                "overlap_component":float(np.sum(cell_ic[overlap])),
                "same_date_component":float(np.sum(cell_ic[same_date])),
                "other_dates_component":float(np.sum(cell_ic[~(overlap|same_date)])),
                "overlap_cells":int(overlap.sum()),"same_date_cells":int(same_date.sum()),
                "baseline_cells":int(base_valid[d].sum()),
                "added_recipient_cells":int(np.sum(valid & ~base_valid[d])),
                "dropped_recipient_cells":int(np.sum(base_valid[d] & ~valid)),
                "donors_without_valid_signal":int(np.sum(~finite_s[origins,indices])),
                "unnormalized_covariance":float(np.mean(centered_s*centered_y))}
            days.append(row)
            label_unit_sum[indices] += yu
            recipient_counts[indices] += 1
            asset_ic_sum[indices] += cell_ic
        if len(days)<2:
            raise ValueError("DIAGNOSTIC_REPLAY_TOO_FEW_DATES")
        daily_records.extend(days)
        averaged = {k:float(np.mean([r[k] for r in days])) for k in (
            "ic","persistent_asset_component","time_varying_component","support_reranking_component",
            "overlap_component","same_date_component","other_dates_component","unnormalized_covariance")}
        averaged.update({k:int(sum(r[k] for r in days)) for k in (
            "n_valid","overlap_cells","same_date_cells","baseline_cells","added_recipient_cells",
            "dropped_recipient_cells","donors_without_valid_signal")})
        repetition_records.append({"variant":strategy,"repetition":repetition+1,"valid_dates":len(days),
            "assignment_sha256":hashlib.sha256(assignment.tobytes()).hexdigest(),**averaged})
        if (repetition+1)%10==0:
            progress(f"{strategy}：{repetition+1}/{config['repetitions']}，支持集与贡献拆分已记录")
    source_centered_means = [float(np.mean(source[ix,a])) if len(ix) else None for a,ix in enumerate(groups)]
    asset_table = pl.DataFrame({"variant":[strategy]*len(assets),"asset_id":assets,
        "signal_mean_normalized_rank":asset_signal_mean,"source_arithmetic_mean":source_means,
        "shuffled_source_arithmetic_mean":source_centered_means,
        "source_history_length":[len(ix) for ix in groups],"recipient_draws":recipient_counts,
        "mean_drawn_normalized_label_rank":np.divide(label_unit_sum,recipient_counts,
            out=np.full(len(assets),np.nan),where=recipient_counts>0),
        "total_normalized_ic_contribution":asset_ic_sum/(config["repetitions"]*len(dates)),
        "persistent_normalized_ic_contribution":asset_signal_mean*label_unit_sum/(config["repetitions"]*len(dates))})
    return pl.DataFrame(daily_records), pl.DataFrame(repetition_records), asset_table


def summarize_replay(repetitions, daily, assets, expected, tolerance):
    observed = repetitions["ic"].to_numpy()
    saved = np.asarray(expected, dtype=float)
    if observed.shape != saved.shape:
        raise ValueError("DIAGNOSTIC_REPETITION_COUNT_MISMATCH")
    difference = float(np.max(np.abs(observed-saved)))
    components = {}
    for field in ("ic","persistent_asset_component","time_varying_component","support_reranking_component",
                  "overlap_component","same_date_component","other_dates_component","unnormalized_covariance"):
        values = repetitions[field].to_numpy()
        components[field] = {"mean":float(values.mean()),"standard_deviation_across_repetitions":float(values.std(ddof=1)),
            "monte_carlo_standard_error":float(values.std(ddof=1)/np.sqrt(len(values))),
            "empirical_2_5_percentile":float(np.quantile(values,.025)),
            "empirical_97_5_percentile":float(np.quantile(values,.975))}
    count_names = ("n_valid","overlap_cells","same_date_cells","baseline_cells","added_recipient_cells",
                   "dropped_recipient_cells","donors_without_valid_signal")
    counts = {k:int(repetitions[k].sum()) for k in count_names}
    decomp_error = daily.select((pl.col("ic")-pl.col("persistent_asset_component")
        -pl.col("time_varying_component")-pl.col("support_reranking_component")).abs().max()).item()
    lag_error = daily.select((pl.col("ic")-pl.col("overlap_component")
        -pl.col("same_date_component")-pl.col("other_dates_component")).abs().max()).item()
    rank_means = assets.filter(pl.col("recipient_draws")>0)["mean_drawn_normalized_label_rank"].to_numpy()
    return {"replay_matches_saved":difference<=tolerance,"max_replay_absolute_difference":difference,
        "decomposition_max_absolute_error":decomp_error,"lag_decomposition_max_absolute_error":lag_error,
        "components":components,"support_counts_over_all_repetitions":counts,
        "overlap_cell_fraction":counts["overlap_cells"]/counts["n_valid"],
        "max_absolute_centered_donor_mean":float(assets["shuffled_source_arithmetic_mean"].abs().max()),
        "asset_mean_drawn_label_rank_sd":float(rank_means.std()),
        "interpretation_limit":"Contributions are exact algebraic accounting, not causal removal experiments; percentile ranges are permutation distributions, not confidence intervals for a population parameter."}


def diagnose_permutation_gates(lake: DataLake, *, config_path: Path=DEFAULT_CONFIG, progress=print):
    config = _read_json(config_path)
    if config.get("diagnostic_id") != "permutation_gate_diagnostics_v1":
        raise ValueError("DIAGNOSTIC_VERSION_UNSUPPORTED")
    artifacts, source_summary, protected = verify_fixed_inputs(lake,config)
    config_sha = file_sha256(config_path)
    definition = {"diagnostic_id":config["diagnostic_id"],"source_manifest_sha256":config["source_manifest_sha256"],
                  "config_sha256":config_sha,"diagnostic_code_hash":source_tree_hash()}
    run_id = "permutation_diagnostic_v1_" + json_hash(definition)[:16]
    out = lake.root/"diagnostics/permutation_gates_v1"/f"run_id={run_id}"
    manifest_path = out/"_MANIFEST.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for record in manifest["outputs"].values():
            if file_sha256(lake.root/record["path"]) != record["sha256"]:
                raise ValueError("DIAGNOSTIC_OUTPUT_CHECKSUM_MISMATCH")
        progress(f"固定输入与诊断产物校验通过，复用 {run_id}")
        return manifest_path
    signal, labels, dates, assets, numerical = reconstruct_grids(lake,artifacts,source_summary,config,progress)
    outputs = {}
    def save_table(name,frame):
        path = out/f"{name}.parquet";_write_immutable_parquet(path,frame)
        outputs[name] = lake.artifact_record(path)
    save_table("independent_neutralization_daily",numerical)
    lake.write_immutable_json(out/"fixed_axes.json",{"dates":[str(d) for d in dates],"assets":assets})
    outputs["fixed_axes"] = lake.artifact_record(out/"fixed_axes.json")
    results = {}
    for strategy in VARIANTS:
        daily,repetitions,asset_table = replay_diagnostics(signal,labels,strategy=strategy,config=config,
            dates=dates,assets=assets,progress=progress)
        expected = source_summary["negative_controls"]["variants"][strategy]["permutation_mean_ics"]
        results[strategy] = summarize_replay(repetitions,daily,asset_table,expected,config["replay_absolute_tolerance"])
        save_table(strategy+"_daily",daily);save_table(strategy+"_repetitions",repetitions);save_table(strategy+"_assets",asset_table)
    for path,sha in protected.items():
        if file_sha256(path) != sha:
            raise ValueError(f"DIAGNOSTIC_PROTECTED_INPUT_CHANGED {path}")
    if file_sha256(config_path)!=config_sha or source_tree_hash()!=definition["diagnostic_code_hash"]:
        raise ValueError("DIAGNOSTIC_IMPLEMENTATION_CHANGED_DURING_RUN")
    report = {"run_id":run_id,"source_run_id":config["source_run_id"],"diagnostic_only":True,
        "status":"completed" if all(v["replay_matches_saved"] for v in results.values()) else "implementation_mismatch",
        "production_gate_status_unchanged":"blocked","holdout_touched":False,
        "fixed_sample":{k:config[k] for k in ("start","end","label_end","holdout_start","primary_horizon","seed","repetitions","lookback")},
        "axes":{"market_dates":len(dates),"assets":len(assets),"finite_signal_cells":int(np.isfinite(signal).sum()),
            "finite_label_cells":int(np.isfinite(labels).sum()),"baseline_joint_cells":int((np.isfinite(signal)&np.isfinite(labels)).sum()),
            "label_donors_without_signal":int((np.isfinite(labels)&~np.isfinite(signal)).sum())},
        "independent_neutralization":{"daily_sections":numerical.height,
            "max_residual_difference":numerical["max_residual_absolute_difference"].max(),
            "max_daily_ic_difference":numerical["ic_absolute_difference"].max()},
        "variants":results,"counterexamples":zero_center_counterexamples(),"theory_sources":config["theory_sources"],
        "decision":"No production criteria or result changed. Any methodology revision requires a separate reviewable version proposal."}
    path = lake.write_immutable_json(out/"report.json",report);outputs["report"]=lake.artifact_record(path)
    lake.write_immutable_json(manifest_path,{"schema_version":1,"run_id":run_id,"status":report["status"],
        "definition":definition,"parent_run_ids":[config["source_run_id"]],
        "source_manifest":config["source_manifest"],"holdout_touched":False,"diagnostic_only":True,
        "protected_file_hashes":{str(p):sha for p,sha in protected.items()},"outputs":outputs})
    return manifest_path
