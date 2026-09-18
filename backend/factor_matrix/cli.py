from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .benchmark_membership import BenchmarkMembershipPipeline
from .boards import board_quality, sync_board_membership
from .board_benchmarks import IndexDailyPipeline, build_board_benchmarks
from .quality import render_quality_report, run_daily_quality, run_history_quality
from .research import build_tradable_universe
from .factor_data import FactorDataPipeline
from .factor_quality import render_factor_data_quality, run_factor_data_quality
from .secrets import load_tushare_token, token_fingerprint
from .source import TushareClient
from .storage import DataLake, source_tree_hash
from .api import serve_strategy_api
from .classification import IndustryClassificationPipeline, industry_quality
from .cli_parser import build_parser
from .policy_config import load_market_domain_policy
from .concepts import (
    ConceptChangeScanner,
    ConceptSnapshotPipeline,
    concept_observation_manifest,
    concept_snapshot_passed,
)
from .incremental_market import IncrementalMarketPipeline
from .l0_production import l0_ready_passed, publish_l0_readiness
from .revisioned_silver import latest_silver_date, read_as_of_frame
from .diagnostics import (
    run_g0_history_diagnostics, run_listing_variant_sensitivity,
    run_new_listing_window_diagnostics,
)
from .calculation.l1 import (
    promote_l1_history, run_l1_history_diagnostics, run_l1_risk_exposure,
    run_l1_risk_exposure_history,
)
from .calendar_backfill import backfill_trade_calendar
from .calculation.l2 import (
    run_g3_risk_only, run_g3_risk_only_history, run_g4_attribution_diagnostics,
    run_g4_statistical_factor_spectrum, run_g3_run_reconciliation,
    run_g3_reconciliation, bind_g3_reconciliation_to_current,
    run_g3_canonicalization_attestation,
    bind_g3_attestation_to_current,
    run_g4_statistical_alpha_overlap,
    run_g4_style_loading_semantic_probe,
    run_g4_wls_weight_selection,
    freeze_and_publish_g4_wls_weight,
    run_g4_structural_weight_residual,
    run_g4_3,
    run_g4_3_v2,
    run_pit_contract_e2e,
)


def record_factor_quality_gate(lake: DataLake, report: dict) -> Path | None:
    candidates: list[tuple[str, Path, dict]] = []
    for path in lake.manifests.glob("factor_data_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("job") == "factor_data_sync":
            candidates.append((payload.get("completed_at", ""), path, payload))
    if not candidates:
        return None
    _, _, payload = max(candidates, key=lambda item: item[0])
    payload["quality_gate"] = {
        "status": report["status"],
        "report": f"metadata/quality_reports/factor_data_{report['as_of_date']}.json",
    }
    return lake.write_manifest(payload["run_id"], payload)


def snapshot_matches(path: Path, target: date, *, kind: str | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("as_of_date") != target.isoformat():
        return False
    if kind is None:
        return True
    if int(payload.get("schema_version", 0)) < 2:
        return False
    if kind == "research":
        board_counts = payload.get("counts", {}).get("by_board", {})
        return (
            set(board_counts) == {"MAIN", "CHINEXT", "STAR", "BSE"}
            and payload.get("lineage", {}).get("code_hash") == source_tree_hash()
        )
    if kind == "strategy":
        return (
            payload.get("strategy", {}).get("calculation_scope") == "board"
            and payload.get("lineage", {}).get("code_hash") == source_tree_hash()
        )
    raise ValueError(f"unsupported snapshot kind: {kind}")


def universe_snapshot_covers(lake: DataLake, target: date) -> bool:
    """Require a passed immutable PIT universe before a daily run is current."""
    for path in lake.manifests.glob("tradable_universe_v1_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("job") == "tradable_universe_v1"
                and payload.get("quality_gate", {}).get("status") == "passed"
                and date.fromisoformat(payload["start"]) <= target
                and date.fromisoformat(payload["end"]) >= target
            ):
                return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return False


def board_artifacts_cover(lake: DataLake, target: date) -> bool:
    required = {"board_membership_sync", "index_daily_sync", "board_benchmark"}
    found: set[str] = set()
    for path in lake.manifests.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            job = payload.get("job")
            if job not in required or payload.get("quality_gate", {}).get("status") != "passed":
                continue
            target_text = target.isoformat()
            if job == "board_membership_sync" and payload.get("config", {}).get("as_of_date") == target_text:
                found.add(job)
            elif payload.get("config", {}).get("end") == target_text:
                found.add(job)
        except (OSError, json.JSONDecodeError):
            continue
    return found == required


def china_market_date() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def main() -> int:
    args = build_parser().parse_args()
    if args.command == 'build-core6-current-v1':
        from .calculation.services.core6_pipeline import run_core6_current
        print(run_core6_current(project_root=args.project_root.resolve(), gate_path=args.data_gate.resolve(),
              gate_sha256=args.gate_sha256, output_root=args.output_root.resolve()))
        return 0
    if args.command == 'reconstruct-core6-paired-v1':
        from .calculation.services.core6_reconstruction import run_reconstruction
        print(run_reconstruction(project_root=args.project_root.resolve(), gate_path=args.data_gate.resolve(),
              gate_sha256=args.gate_sha256, approval_path=args.approval.resolve(),
              output_root=args.output_root.resolve(), workers=args.workers))
        return 0
    if args.command == 'build-core6-rolling-v1':
        from .calculation.services.core6_rolling import run_rolling
        print(run_rolling(parent_manifest=args.parent_manifest.resolve(),parent_sha256=args.parent_sha256,
                          output_root=args.output_root.resolve()))
        return 0
    if args.command == 'diagnose-core6-calibration-v1':
        from .calculation.services.core6_calibration import run_calibration
        print(run_calibration(project_root=args.project_root.resolve(), config_path=args.config.resolve(),
                              config_sha256=args.config_sha256, output_root=args.output_root.resolve()))
        return 0
    if args.command == 'audit-core6-independent-v1':
        from .calculation.services.core6_independent_audit import run_independent_audit
        print(run_independent_audit(project_root=args.project_root.resolve(), config_path=args.config.resolve(),
                                    config_sha256=args.config_sha256, output_root=args.output_root.resolve()))
        return 0
    if args.command == 'reassess-core6-admission-v2':
        from .calculation.services.core6_admission import run_admission
        print(run_admission(project_root=args.project_root.resolve(),policy_path=args.policy.resolve(),
                            policy_sha256=args.policy_sha256,output_root=args.output_root.resolve()))
        return 0
    if args.command == 'migrate-risk-acceptance-v2':
        from .calculation.services.risk_acceptance_migration import migrate_acceptance
        print(migrate_acceptance(registry_path=args.registry.resolve(),project_root=args.project_root.resolve(),
                                approval_path=args.approval.resolve(),approval_sha256=args.approval_sha256,
                                output_root=args.output_root.resolve()))
        return 0
    if args.command == 'build-risk-snapshot-v1':
        from .calculation.services.risk_snapshot import publish_request
        print(publish_request(args.input.resolve(), args.input_sha256, args.output_dir.resolve()))
        return 0
    if args.command == 'run-m3-risk-acceptance-v1':
        from .calculation.services.m3_fixture import publish_m3
        print(publish_m3(args.output_dir.resolve(), args.data_root.resolve(), args.config_root.resolve()))
        return 0
    if args.command == 'build-path-ensemble-v1':
        from .calculation.services.path_ensemble import publish_ensemble
        print(publish_ensemble(args.output_dir.resolve(), reference_path=args.reference.resolve(), workers=args.workers))
        return 0
    if args.command == 'build-backtest-preview-v3':
        from .calculation.services.backtest_preview_v3 import publish_preview
        print(publish_preview(args.output_dir.resolve()))
        return 0
    if args.command == 'build-backtest-preview-v2':
        from .calculation.services.backtest_preview_v2 import publish_preview
        print(publish_preview(args.output_dir.resolve()))
        return 0
    if args.command == 'run-equity-fixture-v2':
        from .calculation.services.equity_fixture import publish_equity_fixture
        print(publish_equity_fixture(args.output_dir.resolve(), render_charts=args.render_charts, font_path=args.font_path))
        return 0
    if args.command == 'build-backtest-preview':
        from .calculation.services.backtest_preview import publish_preview
        print(publish_preview(args.output_dir.resolve()))
        return 0
    lake = DataLake(args.data_root)
    if args.command == "profile-market-capacity":
        from .calculation.services.backtest_market import profile_market_capacity
        with lake.pipeline_lock():
            manifest=profile_market_capacity(lake,market_manifest=args.market_manifest,expected_sha256=args.manifest_sha256)
        print(manifest)
        return 0
    if args.command == "materialize-backtest-market":
        from .calculation.services.backtest_market import materialize_backtest_market
        with lake.pipeline_lock():
            manifest=materialize_backtest_market(lake,gate_path=args.data_gate,gate_sha256=args.gate_sha256,start=args.start,end=args.end)
        print(manifest)
        return 0
    if args.command == "complete-data-refresh":
        from .data_refresh import finish_data_refresh
        with lake.pipeline_lock():
            token = load_tushare_token()
            client = TushareClient(token)
            try:
                manifest = finish_data_refresh(client, lake, market_manifest=args.market_manifest,
                    credential_fingerprint=token_fingerprint(token), resume_manifest=args.resume_manifest, progress=lambda s: print(s, flush=True))
            finally:
                client.close()
                token = ""
        print(manifest)
        return 0 if json.loads(manifest.read_text())["status"] == "passed" else 1
    if args.command == "review-backtest-path":
        from .calculation.services.backtest_review import review_backtest
        with lake.pipeline_lock():
            manifest = review_backtest(lake, config_path=args.config)
        print(manifest)
        return 0
    if args.command == "catchup-market-facts":
        from .market_catchup import backfill_market
        with lake.pipeline_lock():
            token = load_tushare_token()
            client = TushareClient(token)
            try:
                manifest = backfill_market(client, lake, start=args.start, end=args.end,
                    credential_fingerprint=token_fingerprint(token), progress=lambda s: print(s, flush=True))
            finally:
                client.close()
                token = ""
        print(manifest)
        return 0 if json.loads(manifest.read_text())["status"] == "passed" else 1
    if args.command == "run-strategy-fixture":
        from .calculation.services.strategy_fixture import run_strategy_fixture
        with lake.pipeline_lock():
            manifest = run_strategy_fixture(lake, config_path=args.config)
        print(manifest)
        return 0
    if args.command == "run-accounting-fixture":
        from .calculation.services.accounting_fixture import run_accounting_fixture
        with lake.pipeline_lock():
            manifest = run_accounting_fixture(lake, config_path=args.config)
        print(manifest)
        return 0
    if args.command == 'calibrate-synthetic-temporal-null':
        from .calculation.services.synthetic_calibration import run_synthetic_calibration
        with lake.pipeline_lock():
            manifest = run_synthetic_calibration(lake,config_path=args.config,progress=lambda message:print(message,flush=True))
        print(manifest)
        return 0 if json.loads(manifest.read_text())['status']=='synthetic_framework_calibration_passed' else 1
    if args.command == "review-evaluation-v2":
        from .calculation.services.evaluation_review_v2 import publish_evaluation_review
        with lake.pipeline_lock():
            manifest = publish_evaluation_review(lake, config_path=args.config)
        print(manifest)
        return 1  # The review completes; temporal research validation is blocked.
    if args.command == "diagnose-probe-permutation-gates":
        from .calculation.services.permutation_diagnostics import diagnose_permutation_gates
        with lake.pipeline_lock():
            manifest = diagnose_permutation_gates(lake, config_path=args.config,
                progress=lambda message: print(message, flush=True))
        print(manifest)
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] == "completed" else 1
    if args.command == "run-research-pipeline":
        from .calculation.services.research_pipeline import run_research_pipeline
        with lake.pipeline_lock():
            manifest = run_research_pipeline(lake, config_path=args.config, end=args.end,
                                            progress=lambda message: print(message, flush=True))
        print(manifest)
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] == "passed" else 1
    if args.command == "init":
        catalog = lake.refresh_catalog()
        print(f"Initialized {lake.root}; catalog={catalog}")
        return 0
    if args.command == "sync-concepts":
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            manifest = ConceptSnapshotPipeline(
                client, lake, token_fingerprint(token)
            ).sync(
                args.date,
                force=args.force,
                request_delay=args.request_delay,
                workers=args.workers,
                progress=print,
            )
        finally:
            client.close()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        print(json.dumps({
            "manifest": str(manifest),
            "status": payload["status"],
            "output": payload.get("output"),
            "silver_version_id": payload.get("silver_version_id"),
            "quality_gate": payload.get("quality_gate"),
        }, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "passed" else 2
    if args.command == "serve-api":
        serve_strategy_api(
            args.host,
            args.port,
            data_root=args.data_root,
            research_snapshot_path=args.research_snapshot,
        )
        return 0
    if args.command == "quality-industry":
        report = industry_quality(
            lake, args.date, level=args.level, standard=args.standard
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.command == "sync-industry":
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            with lake.pipeline_lock():
                manifest = IndustryClassificationPipeline(
                    client, lake, token_fingerprint(token)
                ).sync(args.standard, args.request_delay)
            print(f"Industry classification sync complete; manifest={manifest}")
        finally:
            client.close()
            token = ""
        return 0
    if args.command == "sync-benchmarks":
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            with lake.pipeline_lock():
                manifest = BenchmarkMembershipPipeline(
                    client, lake, token_fingerprint(token)
                ).sync(args.start, args.end)
            print(f"Benchmark history sync complete; manifest={manifest}")
        finally:
            client.close()
            token = ""
        return 0
    if args.command == "sync-boards":
        with lake.pipeline_lock():
            manifest = sync_board_membership(lake, args.date)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "sync-board-indexes":
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            with lake.pipeline_lock():
                manifest = IndexDailyPipeline(
                    client, lake, token_fingerprint(token)
                ).sync(args.start, args.end)
            print(f"Board index daily sync complete; manifest={manifest}")
        finally:
            client.close()
            token = ""
        return 0
    if args.command == "board-benchmark-build":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            report = build_board_benchmarks(lake, args.start, args.end)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "quality-daily":
        report = run_daily_quality(lake, args.date)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_quality_report(report))
        return 0 if report["status"] == "passed" else 1
    if args.command == "quality-latest":
        latest_date = latest_silver_date(lake, "prices_daily", "trade_date")
        report = run_daily_quality(lake, latest_date)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_quality_report(report))
        return 0 if report["status"] == "passed" else 1
    if args.command == "quality-history":
        report = run_history_quality(lake)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_quality_report(report))
        return 0 if report["status"] == "passed" else 1
    if args.command == "diagnose-g0-history":
        with lake.immutable_base_guard():
            manifest = run_g0_history_diagnostics(lake, args.config)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "derive-new-listing-window":
        with lake.immutable_base_guard():
            manifest = run_new_listing_window_diagnostics(
                lake, args.config, args.external_facts
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "compare-listing-universe-variants":
        with lake.immutable_base_guard():
            manifest = run_listing_variant_sensitivity(
                lake,
                universe_metadata_path=args.universe_metadata,
                listing_window_manifest_path=args.listing_window_manifest,
                policy_path=args.new_listing_policy,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "build-l1-risk-exposure":
        with lake.pipeline_lock():
            manifest = run_l1_risk_exposure(
                lake,
                start=args.start,
                end=args.end,
                universe_metadata_path=args.universe_metadata,
                board_benchmark_summary_path=args.board_benchmark_summary,
                universe_variant=args.universe_variant,
                publish_current=args.publish_current,
                l1_config_path=args.l1_config,
                listing_policy_path=args.new_listing_policy,
                risk_candidates_path=args.risk_candidates,
                protocol_path=args.research_protocol,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "build-l1-risk-exposure-history":
        with lake.pipeline_lock():
            manifest = run_l1_risk_exposure_history(
                lake,
                start=args.start,
                end=args.end,
                universe_metadata_path=args.universe_metadata,
                board_benchmark_summary_path=args.board_benchmark_summary,
                listing_policy_path=args.new_listing_policy,
                risk_candidates_path=args.risk_candidates,
                protocol_path=args.research_protocol,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "diagnose-l1-history":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_l1_history_diagnostics(lake, args.history_manifest)
        print(manifest.read_text(encoding="utf-8"))
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] == "passed" else 1
    if args.command == "promote-l1-history":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            current = promote_l1_history(
                lake, args.history_manifest, args.diagnostics_manifest,
            )
        print(current.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g3-risk-only":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g3_risk_only(
                lake, start=args.start, end=args.end,
                publish_current=args.publish_current,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g3-risk-only-history":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g3_risk_only_history(
                lake, start=args.start, end=args.end,
                publish_current=args.publish_current,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-attribution-diagnostics":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_attribution_diagnostics(
                lake, config_path=args.config, external_facts_path=args.external_facts,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "reconcile-g3-runs":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g3_run_reconciliation(
                lake, old_manifest_path=args.old_manifest,
                new_manifest_path=args.new_manifest, config_path=args.config,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "reconcile-g3-current":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g3_reconciliation(lake)
            bind_g3_reconciliation_to_current(lake, manifest)
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "attest-g3-canonicalization":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g3_canonicalization_attestation(lake, config_path=args.config)
            bind_g3_attestation_to_current(lake, manifest)
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-statistical-spectrum":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_statistical_factor_spectrum(lake, config_path=args.config)
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-statistical-alpha-overlap":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_statistical_alpha_overlap(
                lake, spectrum_manifest_path=args.spectrum_manifest,
                config_path=args.config,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-style-semantic-probe":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_style_loading_semantic_probe(
                lake, config_path=args.config,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-wls-weight-selection":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_wls_weight_selection(lake, config_path=args.config)
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "freeze-g4-wls-weight":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = freeze_and_publish_g4_wls_weight(
                lake, candidate_manifest_path=args.candidate_manifest,
                config_path=args.config,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-structural-weight-residual":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_structural_weight_residual(
                lake, candidate_manifest_path=args.candidate_manifest,
                freeze_manifest_path=args.freeze_manifest,
                reconciliation_manifest_path=args.reconciliation_manifest,
            )
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "run-g4-3-stratified-residual-permutation":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_3(lake, config_path=args.config)
        print(manifest.read_text(encoding="utf-8"))
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] == "passed" else 1
    if args.command == "run-g4-3-v2-stratified-residual-permutation":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_g4_3_v2(lake, config_path=args.config)
        print(manifest.read_text(encoding="utf-8"))
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] in {"passed", "passed_draft"} else 1
    if args.command == "run-pit-contract-e2e":
        with lake.pipeline_lock(), lake.immutable_base_guard():
            manifest = run_pit_contract_e2e(lake, config_path=args.config)
        print(manifest.read_text(encoding="utf-8"))
        return 0 if json.loads(manifest.read_text(encoding="utf-8"))["status"] == "passed" else 1
    if args.command == "backfill-trade-calendar":
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            with lake.pipeline_lock():
                manifest = backfill_trade_calendar(
                    client, lake, start=args.start, end=args.end,
                    credential_fingerprint=token_fingerprint(token),
                )
        finally:
            client.close()
        print(manifest.read_text(encoding="utf-8"))
        return 0
    if args.command == "universe-build":
        domain_policy = load_market_domain_policy(args.universe_policy)
        with lake.pipeline_lock():
            report = build_tradable_universe(
                lake,
                args.start,
                args.end,
                domain_policy,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "daily-update":
        target = args.date or china_market_date()
        try:
            calendar = read_as_of_frame(lake, "trade_calendar")
        except FileNotFoundError:
            raise RuntimeError("DAILY_UPDATE_CALENDAR_MISSING run an initial market sync first")
        calendar_frame = calendar.filter(calendar["cal_date"] == target)
        if not calendar_frame.is_empty() and not bool(calendar_frame.get_column("is_open").max()):
            print(f"Daily update skipped: {target} is not a China trading day")
            return 0
        if (
            not args.force
            and l0_ready_passed(lake, target)
        ):
            print(f"Daily update already current: {target}")
            return 0
        token = ""
        client = None
        try:
            with lake.pipeline_lock():
                token = load_tushare_token()
                client = TushareClient(token)
                fingerprint = token_fingerprint(token)
                market_manifest = IncrementalMarketPipeline(
                    client, lake, fingerprint
                ).sync(target, force=args.force)
                market_payload = json.loads(market_manifest.read_text(encoding="utf-8"))
                if market_payload["status"] == "closed":
                    print(f"Daily update skipped: {target} is not a China trading day")
                    return 0
                if market_payload["status"] != "passed":
                    raise RuntimeError(
                        f"DAILY_UPDATE_MARKET_FAILED target={target} "
                        f"failed={market_payload.get('quality_gate', {}).get('failed_checks')}"
                    )
                board_manifest_payload = sync_board_membership(lake, target)
                board_manifest = lake.manifests / f"{board_manifest_payload['run_id']}.json"
                index_manifest = IndexDailyPipeline(client, lake, fingerprint).sync(target, target)
                risk_manifest = FactorDataPipeline(
                    client, lake, fingerprint
                ).sync_risk_free(target, target)
                existing_concept = concept_observation_manifest(lake, target)
                concept_manifest = existing_concept or ConceptChangeScanner(
                    client, lake, fingerprint
                ).sync(target)
                concept_payload = json.loads(concept_manifest.read_text(encoding="utf-8"))
                if concept_payload["status"] != "passed":
                    raise RuntimeError(
                        f"DAILY_UPDATE_CONCEPT_SCAN_FAILED target={target} "
                        f"reason={concept_payload.get('quality_gate', {}).get('reason')}"
                    )
                lake.refresh_catalog()
                readiness = publish_l0_readiness(
                    lake, target,
                    component_manifests=[
                        market_manifest, board_manifest, index_manifest,
                        risk_manifest, concept_manifest,
                    ],
                )
                readiness_payload = json.loads(readiness.read_text(encoding="utf-8"))
                if readiness_payload["status"] != "passed":
                    raise RuntimeError(
                        f"L0_READINESS_FAILED target={target} "
                        f"failed={readiness_payload['quality_gate']['failed_checks']}"
                    )
        finally:
            if client is not None:
                client.close()
            token = ""
        print(f"Daily update complete: {target}; unified L0 readiness passed")
        return 0

    if args.command == "quality-factor-data":
        import polars as pl

        check_date = args.date
        if check_date is None:
            check_date = latest_silver_date(lake, "prices_daily", "trade_date")
        report = run_factor_data_quality(lake, check_date)
        record_factor_quality_gate(lake, report)
        print(
            json.dumps(report, ensure_ascii=False, indent=2)
            if args.json
            else render_factor_data_quality(report)
        )
        return 0 if report["status"] == "passed" else 1

    if args.command == "sync-factor-data":
        if args.financial_end_year < args.financial_start_year:
            raise ValueError("financial end year must not be before start year")
        token = load_tushare_token()
        client = TushareClient(token)
        try:
            with lake.pipeline_lock():
                manifest = FactorDataPipeline(
                    client, lake, token_fingerprint(token)
                ).sync(
                    range(args.financial_start_year, args.financial_end_year + 1),
                    args.risk_start,
                    args.risk_end,
                    request_delay=args.request_delay,
                )
                factor_report = run_factor_data_quality(lake, args.risk_end)
                record_factor_quality_gate(lake, factor_report)
                if factor_report["status"] != "passed":
                    raise RuntimeError("FACTOR_DATA_QUALITY_FAILED")
            print(f"Factor data sync complete; manifest={manifest}")
        finally:
            client.close()
            token = ""
        return 0

    raise RuntimeError(f"UNHANDLED_COMMAND {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
