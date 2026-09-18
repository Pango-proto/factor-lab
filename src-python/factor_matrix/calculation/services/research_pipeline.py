"""Versioned M1 orchestration: explicit artifacts, sealed holdout, immutable runs."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Callable

import polars as pl

from ...storage import DataLake, file_sha256, json_hash, source_tree_hash
from ...factor_engine.contracts import FactorRole, FactorFamily
from ...factor_engine.plugins.risk._factory import risk_definition
from ...factor_engine.registry import FactorRegistry
from ...factor_engine.store import FactorRegistryStore
from ..components.legacy_adapters import adopt_risk_exposure_matrix, adopt_tradable_universe
from ..components.market import MaterializeRealizedReturns
from ..components.labels import BuildForwardLabels
from ..components.probe_signal import MaterializeProbeSignal
from ..components.probe_evaluation import EvaluateProbe
from ..core.contracts import CalculationRequest
from ..core.executor import CalculationExecutor
from ..core.policy import architecture_boundary_policy
from ..core.registry import CalculationRegistry
from ..l2.estimation_evaluation import EvaluationContract
from ..publishing.local import LocalArtifactStore

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT / "config/research_pipeline_v1.json"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_sample(dates, contract, requested_end=None):
    dates = sorted(set(d for d in dates if d < contract.holdout_start))
    if len(dates) <= contract.purge_gap_trading_days:
        raise ValueError("RESEARCH_PIPELINE_INSUFFICIENT_CALENDAR")
    safe = dates[-contract.purge_gap_trading_days - 1]
    end = requested_end or safe
    if end > safe or end < contract.development_start or end not in dates:
        raise ValueError(f"RESEARCH_PIPELINE_SAMPLE_BOUNDARY end={end} safe_end={safe}")
    feature_index = dates.index(end)
    label_end = dates[feature_index + max(contract.reported_horizons)]
    return end, label_end, safe


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def register_probe(lake):
    # The existing descriptor registry path explicitly assigns the probe role
    # and does not create research_attempt / alpha_assertion records.
    definition = risk_definition(
        definition_path=Path(__file__), factor_id="reversal_20d_v1", display_name="20 日反转研究探针",
        role=FactorRole.DESCRIPTOR, formula_expr="1-exp(rolling_sum(log1p(realized_return),20,min_samples=15))",
        source_tables=("realized_returns_v1",), source_fields=("realized_return",),
        params={"lookback_trading_days":20,"min_valid_observations":15}, pit_key="trade_date",
        hypothesis="Validate the estimation framework only; this is not a deployable Alpha assertion.",
        lookback_days=20, min_obs=15, family_root_id="reversal_framework_probe", variant_count=1,
        status_reason="Non-deployable registered framework probe",
    )
    definition = replace(definition, spec=replace(definition.spec, family=FactorFamily.ALPHA,
        proposed_date=date(2026,9,13), status_changed_date=date(2026,9,13),
        code_path=str(Path(__file__).relative_to(PROJECT))))
    FactorRegistryStore(lake.metadata / "factor_registry.sqlite").sync_definitions(FactorRegistry((definition,)))
    return definition.spec.feature_key


def run_research_pipeline(lake: DataLake, *, config_path: Path = DEFAULT_CONFIG,
                          end: date | None = None, progress: Callable[[str], None] | None = None) -> Path:
    config = _json(config_path)
    if config.get("pipeline_id") != "research_pipeline_v1":
        raise ValueError("RESEARCH_PIPELINE_VERSION_UNSUPPORTED")
    # Paths are relative to the pipeline config, never the process cwd.
    evaluation_path = config_path.parent / config["evaluation_config"]
    label_policy_path = config_path.parent / config["return_label_policy"]
    contract = EvaluationContract.from_json(evaluation_path)
    evaluation_json = _json(evaluation_path)
    if evaluation_json.get("status") != "active" or evaluation_json["sample"].get("holdout_status") != "sealed_unopened":
        raise ValueError("RESEARCH_PIPELINE_CONTRACT_NOT_ACTIVE_SEALED")
    pointer_path = lake.root / "gold/risk_exposure_matrix/_CURRENT.json"
    pointer = _json(pointer_path)
    l1_path = lake.root / pointer["manifest"]
    l1 = _json(l1_path)
    diagnosis_path = lake.root / pointer["diagnostics_manifest"]
    diagnosis = _json(diagnosis_path)
    if (l1["run_id"] != pointer["run_id"] or diagnosis.get("status") != "passed"
        or diagnosis.get("promotion_gate") != "passed"
        or diagnosis.get("risk_basis_id") != l1["risk_basis_id"]
        or diagnosis.get("history_run_id") != l1["run_id"]
        or diagnosis.get("history_manifest_sha256") != file_sha256(l1_path)
        or pointer.get("risk_basis_id") != l1["risk_basis_id"]):
        raise ValueError("RESEARCH_PIPELINE_L1_LINEAGE_OR_GATE_INVALID")
    universe_run = l1["tradable_universe_run_id"]
    universe_dir = lake.root / "gold/tradable_universe/artifact_version=1" / f"run_id={universe_run}"
    universe_meta = _json(universe_dir / "metadata.json")
    if universe_meta["run_id"] != universe_run or universe_meta["silver_version_id"] != l1["silver_version_id"]:
        raise ValueError("RESEARCH_PIPELINE_UNIVERSE_LINEAGE_INVALID")
    returns_path = lake.silver / "returns_daily/data.parquet"
    calendar = (pl.scan_parquet(returns_path).filter(pl.col("trade_date") < contract.holdout_start)
        .select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list())
    feature_end, label_end, safe_end = resolve_sample(calendar, contract, end)
    source_start = date.fromisoformat(l1["start"])
    if source_start >= contract.development_start:
        raise ValueError("RESEARCH_PIPELINE_SIGNAL_WARMUP_MISSING")
    sources = {
        "pipeline_config": config_path, "evaluation_config": evaluation_path, "label_policy": label_policy_path,
        "l1_pointer": pointer_path, "l1_manifest": l1_path, "l1_diagnostics": diagnosis_path,
        "l1_exposures": lake.root / l1["outputs"]["risk_exposure_matrix_v1"]["path"],
        "universe_metadata": universe_dir / "metadata.json", "universe": universe_dir / "tradable_universe.parquet",
        "returns": returns_path,
    }
    silver_pointer = lake.metadata / "silver_versions/_CURRENT"
    if silver_pointer.exists():
        sources["silver_pointer"] = silver_pointer
        sources["silver_manifest"] = silver_pointer.parent / silver_pointer.read_text().strip()
    hashes = {k:file_sha256(p) for k,p in sources.items()}
    if hashes["l1_exposures"] != l1["outputs"]["risk_exposure_matrix_v1"]["sha256"]:
        raise ValueError("RESEARCH_PIPELINE_EXPOSURE_CHECKSUM_MISMATCH")
    definition = {"pipeline_id":config["pipeline_id"], "source_hashes":hashes, "code_hash":source_tree_hash(),
                  "start":str(contract.development_start), "end":str(feature_end), "label_end":str(label_end)}
    run_id = "research_pipeline_v1_" + json_hash(definition)[:16]
    run_dir = lake.root / "gold/research_pipeline" / f"run_id={run_id}"
    manifest_path = run_dir / "_MANIFEST.json"
    if manifest_path.exists():
        previous = _json(manifest_path)
        for record in previous["outputs"].values():
            if file_sha256(lake.root / record["path"]) != record["sha256"]:
                raise ValueError("RESEARCH_PIPELINE_OUTPUT_CHECKSUM_MISMATCH")
        for record in previous["artifacts"]:
            meta_path = lake.root / record["uri"]
            if file_sha256(meta_path) != record["checksum"]:
                raise ValueError("RESEARCH_PIPELINE_ARTIFACT_CHECKSUM_MISMATCH")
            for table in _json(meta_path)["tables"].values():
                if file_sha256(lake.root / table["uri"]) != table["checksum"]:
                    raise ValueError("RESEARCH_PIPELINE_TABLE_CHECKSUM_MISMATCH")
        if progress:
            progress(f"已验证固定输入与产物校验和，复用 {run_id}")
        return manifest_path
    scope = "research:" + json_hash(definition)[:16]
    store = LocalArtifactStore(lake)
    plugins = (adopt_tradable_universe(lake), adopt_risk_exposure_matrix(lake),
               MaterializeRealizedReturns(lake, label_policy_path=label_policy_path), MaterializeProbeSignal(lake),
               BuildForwardLabels(lake, label_policy_path=label_policy_path), EvaluateProbe(evaluation_path, progress))
    executor = CalculationExecutor(CalculationRegistry(plugins), store, store, architecture_boundary_policy())
    refs = []
    def execute(operation, parameters, inputs=()):
        if progress:
            progress(operation)
        ref, = executor.execute(CalculationRequest(operation_id=operation, operation_version="1",
            as_of_date=feature_end, scope_key=scope, parameters=parameters, inputs=tuple(inputs)))
        refs.append(ref)
        return ref
    full = {"start_date":str(source_start), "end_date":str(label_end)}
    # The first source price has no preceding close, hence no return. Keep
    # that day in the universe for state transitions, but begin returns on
    # the next market session. Both precede the declared feature warmup.
    return_start = next(d for d in calendar if d > source_start)
    if len([d for d in calendar if return_start <= d < contract.development_start]) < 20:
        raise ValueError("RESEARCH_PIPELINE_INSUFFICIENT_WARMUP_RETURNS")
    feature = {"start_date":str(contract.development_start), "end_date":str(feature_end)}
    try:
        feature_key = register_probe(lake)
        u = execute("adopt_tradable_universe", {**full,"legacy_run_id":universe_run})
        x = execute("adopt_risk_exposure_matrix", feature)
        r = execute("materialize_realized_returns", {**full,"start_date":str(return_start)}, (u,))
        s = execute("materialize_probe_signal", feature, (r,u))
        y = execute("build_forward_labels", {"horizons":list(contract.reported_horizons),
            "max_suspended_fraction":config["max_suspended_fraction"]}, (r,u))
        e = execute("evaluate_research_probe", feature, (x,y,u,s))
        if any(file_sha256(p) != hashes[k] for k,p in sources.items()):
            raise ValueError("RESEARCH_PIPELINE_SOURCE_CHANGED_DURING_RUN")
        if source_tree_hash() != definition["code_hash"]:
            raise ValueError("RESEARCH_PIPELINE_CODE_CHANGED_DURING_RUN")
        metrics = _json(lake.root/e.uri)["metadata"]
        passed = metrics["negative_controls"]["passed"]
        summary = {"schema_version":1,"framework_id":contract.framework_id,"run_id":run_id,
            "config_sha256":hashes["evaluation_config"],"status":"passed" if passed else "blocked",
            "scope":{"start":str(contract.development_start),"end":str(feature_end),"safe_feature_end":str(safe_end),
                "label_end":str(label_end),"holdout_start":str(contract.holdout_start),"holdout_touched":False,
                "purge_gap_trading_days":contract.purge_gap_trading_days,"signal_id":contract.signal_id,
                "risk_basis_id":l1["risk_basis_id"],"risk_set_version":contract.risk_set_version,
                "risk_set_status":"candidate","deployable":False}, **metrics}
        summary_path = lake.write_immutable_json(run_dir/"summary.json", summary)
        manifest = {"schema_version":1,"run_id":run_id,"status":summary["status"],"execution_status":"completed",
            "definition":definition,"risk_basis_id":l1["risk_basis_id"],"risk_set_version":contract.risk_set_version,
            "sources":{key:{"path":str(path.resolve()),"sha256":hashes[key]} for key,path in sources.items()},
            "risk_set_status":"candidate","feature_key":feature_key,"holdout_touched":False,
            "parent_run_ids":[l1["run_id"],universe_run],
            "artifacts":[{"run_id":ref.key.run_id,"artifact_type":ref.key.artifact_type,"uri":ref.uri,"checksum":ref.checksum} for ref in refs],
            "outputs":{"summary":{"path":str(summary_path.relative_to(lake.root)),"sha256":file_sha256(summary_path)}}}
        lake.write_immutable_json(manifest_path, manifest)
        publication = {"run_id":run_id,"status":summary["status"],"manifest":str(manifest_path.relative_to(lake.root)),
            "manifest_sha256":file_sha256(manifest_path), "summary":manifest["outputs"]["summary"]}
        _atomic_json(lake.metadata/"research_pipeline_latest_attempt.json", publication)
        if passed:
            _atomic_json(lake.root/"gold/research_pipeline/_CURRENT.json", publication)
        return manifest_path
    except Exception as exc:
        lake.write_immutable_json(run_dir/f"failure_{json_hash(str(exc))[:12]}.json", {"run_id":run_id,"status":"failed",
            "error":f"{type(exc).__name__}: {exc}","definition":definition,
            "completed_stage_run_ids":[ref.key.run_id for ref in refs],"holdout_touched":False})
        raise
