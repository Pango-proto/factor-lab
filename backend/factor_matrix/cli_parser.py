"""Command-line parser definitions; contains no pipeline execution logic."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factor-matrix")
    subparsers = parser.add_subparsers(dest="command", required=True)
    core6 = subparsers.add_parser('build-core6-current-v1', help='Pinned six-style current X and availability audit; no CURRENT promotion')
    core6.add_argument('--project-root', type=Path, default=Path('.'))
    core6.add_argument('--data-gate', type=Path, required=True)
    core6.add_argument('--gate-sha256', required=True)
    core6.add_argument('--output-root', type=Path, default=Path('data/diagnostics/core6_current_v1'))
    replay6 = subparsers.add_parser('reconstruct-core6-paired-v1', help='Approved historical reconstruction only; actual PIT admission remains blocked')
    replay6.add_argument('--project-root', type=Path, default=Path('.'))
    replay6.add_argument('--data-gate', type=Path, required=True)
    replay6.add_argument('--gate-sha256', required=True)
    replay6.add_argument('--approval', type=Path, default=Path('config/core6_reconstruction_approval_v1.json'))
    replay6.add_argument('--output-root', type=Path, default=Path('data/diagnostics/core6_reconstruction_v1'))
    replay6.add_argument('--workers', type=int, default=2)
    rolling6 = subparsers.add_parser('build-core6-rolling-v1', help='One-session expanding reconstructed F/Delta from a pinned paired G3 manifest')
    rolling6.add_argument('--parent-manifest', type=Path, required=True)
    rolling6.add_argument('--parent-sha256', required=True)
    rolling6.add_argument('--output-root', type=Path, default=Path('data/diagnostics/core6_rolling_v1'))
    calibration6 = subparsers.add_parser('diagnose-core6-calibration-v1', help='Fixed paired risk calibration diagnostics; never promote CURRENT')
    calibration6.add_argument('--project-root', type=Path, default=Path('.'))
    calibration6.add_argument('--config', type=Path, default=Path('config/core6_calibration_diagnostic_v1.json'))
    calibration6.add_argument('--config-sha256', required=True)
    calibration6.add_argument('--output-root', type=Path, default=Path('data/diagnostics/core6_calibration_v1'))
    audit6 = subparsers.add_parser('audit-core6-independent-v1', help='Read-only G2 metric, residual identity and missing-event audit')
    audit6.add_argument('--project-root', type=Path, default=Path('.'))
    audit6.add_argument('--config', type=Path, default=Path('config/core6_independent_audit_v1.json'))
    audit6.add_argument('--config-sha256', required=True)
    audit6.add_argument('--output-root', type=Path, default=Path('reports/core6_independent_audit_v1'))
    admission6 = subparsers.add_parser('reassess-core6-admission-v2', help='Approved native-strata retrospective G6 criteria; no production publication')
    admission6.add_argument('--project-root', type=Path, default=Path('.'))
    admission6.add_argument('--policy', type=Path, default=Path('config/core6_risk_admission_v2.json'))
    admission6.add_argument('--policy-sha256', required=True)
    admission6.add_argument('--output-root', type=Path, default=Path('data/diagnostics/core6_admission_v2'))
    migration6 = subparsers.add_parser('migrate-risk-acceptance-v2', help='Approved additive G6 state migration; preserve old rows and backup metadata')
    migration6.add_argument('--project-root', type=Path, default=Path('.'))
    migration6.add_argument('--registry', type=Path, default=Path('data/metadata/factor_registry.sqlite'))
    migration6.add_argument('--approval', type=Path, default=Path('config/core6_admission_v2_approval.json'))
    migration6.add_argument('--approval-sha256', required=True)
    migration6.add_argument('--output-root', type=Path, default=Path('reports/core6_admission_v2/migration'))
    risk = subparsers.add_parser('build-risk-snapshot-v1', help='Publish explicitly timestamped X/F/Delta and target admission')
    risk.add_argument('--input', type=Path, required=True)
    risk.add_argument('--input-sha256', required=True)
    risk.add_argument('--output-dir', type=Path, required=True)
    m3 = subparsers.add_parser('run-m3-risk-acceptance-v1', help='Publish risk engineering acceptance and read-only real-data readiness')
    m3.add_argument('--output-dir', type=Path, default=Path('public/demo-backtest/m3-v1'))
    m3.add_argument('--data-root', type=Path, default=Path('data'))
    m3.add_argument('--config-root', type=Path, default=Path('config'))
    ensemble = subparsers.add_parser('build-path-ensemble-v1', help='Run the frozen 1000-path synthetic execution ensemble')
    ensemble.add_argument('--output-dir', type=Path, default=Path('public/demo-backtest/ensemble-v1'))
    ensemble.add_argument('--reference', type=Path, default=Path('public/demo-backtest/v3/preview.json'))
    ensemble.add_argument('--workers', type=int, default=4)
    preview = subparsers.add_parser('build-backtest-preview', help='Publish fixed synthetic chart demonstration data only')
    preview.add_argument('--output-dir', type=Path, default=Path('public/demo-backtest/v1'))
    preview_v3 = subparsers.add_parser('build-backtest-preview-v3', help='Publish same-market strategy comparison and descriptive diagnostics')
    preview_v3.add_argument('--output-dir', type=Path, default=Path('public/demo-backtest/v3'))
    preview_v2 = subparsers.add_parser('build-backtest-preview-v2', help='Publish strict-ledger synthetic chart preview v2')
    preview_v2.add_argument('--output-dir', type=Path, default=Path('public/demo-backtest/v2-acceptance'))
    equity_fixture = subparsers.add_parser('run-equity-fixture-v2', help='Run normal A-share synthetic execution acceptance v2')
    equity_fixture.add_argument('--output-dir', type=Path, default=Path('reports/equity_execution_v2_20260917'))
    equity_fixture.add_argument('--render-charts', action='store_true')
    equity_fixture.add_argument('--font-path', type=Path)
    capacity = subparsers.add_parser("profile-market-capacity", help="Lagged ADV liquidity profile for explicitly published real bars")
    capacity.add_argument("--market-manifest", type=Path, required=True)
    capacity.add_argument("--manifest-sha256", required=True)
    capacity.add_argument("--data-root", type=Path, default=Path("data"))
    real_market = subparsers.add_parser("materialize-backtest-market", help="Publish real OHLCV/lagged liquidity behind an audited data gate")
    real_market.add_argument("--data-gate", type=Path, required=True)
    real_market.add_argument("--gate-sha256", required=True)
    real_market.add_argument("--start", type=iso_date, required=True)
    real_market.add_argument("--end", type=iso_date, required=True)
    real_market.add_argument("--data-root", type=Path, default=Path("data"))
    refresh = subparsers.add_parser("complete-data-refresh", help="Refresh financial/reference data and audit a passed market catchup")
    refresh.add_argument("--market-manifest", type=Path, required=True)
    refresh.add_argument("--resume-manifest", type=Path, help="Reuse verified successful components from this same-day refresh")
    refresh.add_argument("--data-root", type=Path, default=Path("data"))
    analytics = subparsers.add_parser("review-backtest-path", help="Pinned path metrics and execution counterfactuals; no research promotion")
    analytics.add_argument("--config", type=Path, default=Path("config/backtest_analytics_v1.json"))
    analytics.add_argument("--data-root", type=Path, default=Path("data"))
    catchup = subparsers.add_parser("catchup-market-facts", help="Chronologically append completed days of market facts without promoting research")
    catchup.add_argument("--start", type=iso_date, required=True)
    catchup.add_argument("--end", type=iso_date, required=True)
    catchup.add_argument("--data-root", type=Path, default=Path("data"))
    strategy_fixture = subparsers.add_parser("run-strategy-fixture", help="M2b v1 signal-to-strategy-to-ledger engineering baseline")
    strategy_fixture.add_argument("--data-root", type=Path, default=Path("data"))
    strategy_fixture.add_argument("--config", type=Path, default=Path("config/strategy_fixture_v1.json"))
    accounting = subparsers.add_parser("run-accounting-fixture", help="M2a v1 deterministic engineering ledger; no real research promotion")
    accounting.add_argument("--data-root", type=Path, default=Path("data"))
    accounting.add_argument("--config", type=Path, default=Path("config/accounting_fixture_v1.json"))
    synthetic = subparsers.add_parser('calibrate-synthetic-temporal-null', help='Execute the fixed, approved synthetic calibration; never promote research')
    synthetic.add_argument('--data-root', type=Path, default=Path('data'))
    synthetic.add_argument('--config', type=Path, default=Path('config/synthetic_temporal_calibration_v1.json'))
    review_v2 = subparsers.add_parser("review-evaluation-v2", help="Publish blocked v2 engineering and temporal review states")
    review_v2.add_argument("--data-root", type=Path, default=Path("data"))
    review_v2.add_argument("--config", type=Path, default=Path("config/evaluation_framework_v2.json"))
    permutation_diagnostics = subparsers.add_parser("diagnose-probe-permutation-gates", help="Independently audit pinned M1 permutation gates without changing acceptance")
    permutation_diagnostics.add_argument("--data-root", type=Path, default=Path("data"))
    permutation_diagnostics.add_argument("--config", type=Path, default=Path("config/permutation_gate_diagnostics_v1.json"))
    research_pipeline = subparsers.add_parser("run-research-pipeline", help="Run the versioned M1 probe pipeline before sealed holdout")
    research_pipeline.add_argument("--data-root", type=Path, default=Path("data"))
    research_pipeline.add_argument("--config", type=Path, default=Path("config/research_pipeline_v1.json"))
    research_pipeline.add_argument("--end", type=iso_date, help="Optional earlier feature cutoff; cannot cross the registered purge boundary")
    init = subparsers.add_parser("init", help="Initialize local data directories")
    init.add_argument("--data-root", type=Path, default=Path("data"))
    benchmark = subparsers.add_parser(
        "sync-benchmarks", help="Sync PIT CSI300/500/800 constituent weights"
    )
    benchmark.add_argument("--start", type=iso_date, required=True)
    benchmark.add_argument("--end", type=iso_date, required=True)
    benchmark.add_argument("--data-root", type=Path, default=Path("data"))
    boards = subparsers.add_parser(
        "sync-boards", help="Append board-membership changes derived from canonical daily state"
    )
    boards.add_argument("--date", type=iso_date, required=True)
    boards.add_argument("--data-root", type=Path, default=Path("data"))
    indexes = subparsers.add_parser(
        "sync-board-indexes", help="Sync official daily indexes used as board references"
    )
    indexes.add_argument("--start", type=iso_date, required=True)
    indexes.add_argument("--end", type=iso_date, required=True)
    indexes.add_argument("--data-root", type=Path, default=Path("data"))
    board_benchmark = subparsers.add_parser(
        "board-benchmark-build", help="Build four full-sample board total-return benchmarks"
    )
    board_benchmark.add_argument("--start", type=iso_date, required=True)
    board_benchmark.add_argument("--end", type=iso_date, required=True)
    board_benchmark.add_argument("--data-root", type=Path, default=Path("data"))
    quality = subparsers.add_parser("quality-daily", help="Run point-in-time checks for one trading day")
    quality.add_argument("--date", type=iso_date, required=True)
    quality.add_argument("--data-root", type=Path, default=Path("data"))
    quality.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    quality_latest = subparsers.add_parser(
        "quality-latest", help="Run the primary quality gate on the latest loaded trading day"
    )
    quality_latest.add_argument("--data-root", type=Path, default=Path("data"))
    quality_latest.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    quality_history = subparsers.add_parser(
        "quality-history", help="Run full-history structural and return checks"
    )
    quality_history.add_argument("--data-root", type=Path, default=Path("data"))
    quality_history.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    g0 = subparsers.add_parser(
        "diagnose-g0-history",
        help="Run versioned read-only full-history diagnostics before L1/L2",
    )
    g0.add_argument("--data-root", type=Path, default=Path("data"))
    g0.add_argument(
        "--config", type=Path, default=Path("config/g0_history_diagnostics_v1.json")
    )
    listing = subparsers.add_parser(
        "derive-new-listing-window",
        help="Derive board-by-regime listing-age volatility decay without L2 factor outputs",
    )
    listing.add_argument("--data-root", type=Path, default=Path("data"))
    listing.add_argument(
        "--config", type=Path,
        default=Path("config/new_listing_window_derivation_v1.json"),
    )
    listing.add_argument(
        "--external-facts", type=Path, default=Path("config/external_facts_v1.json")
    )
    sensitivity = subparsers.add_parser(
        "compare-listing-universe-variants",
        help="Persist a membership-only d120 versus derived sensitivity diagnostic",
    )
    sensitivity.add_argument("--data-root", type=Path, default=Path("data"))
    sensitivity.add_argument("--universe-metadata", type=Path, required=True)
    sensitivity.add_argument("--listing-window-manifest", type=Path, required=True)
    sensitivity.add_argument(
        "--new-listing-policy", type=Path,
        default=Path("config/new_listing_policy_v1.json"),
    )
    l1 = subparsers.add_parser(
        "build-l1-risk-exposure",
        help="Build role-specific risk exposures and exposure_quality_v1",
    )
    l1.add_argument("--start", type=iso_date, required=True)
    l1.add_argument("--end", type=iso_date, required=True)
    l1.add_argument("--data-root", type=Path, default=Path("data"))
    l1.add_argument("--universe-metadata", type=Path, required=True)
    l1.add_argument("--board-benchmark-summary", type=Path, required=True)
    l1.add_argument("--universe-variant", default="frozen_d0")
    l1.add_argument("--publish-current", action="store_true")
    l1.add_argument(
        "--l1-config", type=Path, default=Path("config/l1_risk_exposure_v1.json")
    )
    l1.add_argument(
        "--new-listing-policy", type=Path,
        default=Path("config/new_listing_policy_v1.json"),
    )
    l1.add_argument(
        "--risk-candidates", type=Path,
        default=Path("config/risk_factor_set_candidate_v1.json"),
    )
    l1.add_argument(
        "--research-protocol", type=Path,
        default=Path("config/research_protocol_v1.json"),
    )
    l1_history = subparsers.add_parser(
        "build-l1-risk-exposure-history",
        help="Build and consolidate month-partitioned full-history L1 exposures",
    )
    l1_history.add_argument("--start", type=iso_date, required=True)
    l1_history.add_argument("--end", type=iso_date, required=True)
    l1_history.add_argument("--data-root", type=Path, default=Path("data"))
    l1_history.add_argument("--universe-metadata", type=Path, required=True)
    l1_history.add_argument("--board-benchmark-summary", type=Path, required=True)
    l1_history.add_argument(
        "--new-listing-policy", type=Path,
        default=Path("config/new_listing_policy_v1.json"),
    )
    calendar_backfill = subparsers.add_parser(
        "backfill-trade-calendar",
        help="Append a bounded historical SSE calendar range to revisioned Silver",
    )
    calendar_backfill.add_argument("--start", type=iso_date, required=True)
    calendar_backfill.add_argument("--end", type=iso_date, required=True)
    calendar_backfill.add_argument("--data-root", type=Path, default=Path("data"))
    l1_history.add_argument(
        "--risk-candidates", type=Path,
        default=Path("config/risk_factor_set_candidate_v1.json"),
    )
    l1_history.add_argument(
        "--research-protocol", type=Path,
        default=Path("config/research_protocol_v1.json"),
    )
    l1_diagnostics = subparsers.add_parser(
        "diagnose-l1-history",
        help="Build immutable G2 history diagnostics before L1 promotion",
    )
    l1_diagnostics.add_argument("--data-root", type=Path, default=Path("data"))
    l1_diagnostics.add_argument("--history-manifest", type=Path, required=True)
    l1_promote = subparsers.add_parser(
        "promote-l1-history",
        help="Update L1 _CURRENT only after matching history diagnostics pass",
    )
    l1_promote.add_argument("--data-root", type=Path, default=Path("data"))
    l1_promote.add_argument("--history-manifest", type=Path, required=True)
    l1_promote.add_argument("--diagnostics-manifest", type=Path, required=True)
    g3 = subparsers.add_parser(
        "run-g3-risk-only",
        help="Run constrained L2b risk-only return decomposition from committed L1",
    )
    g3.add_argument("--start", type=iso_date, required=True, help="Realized-return date")
    g3.add_argument("--end", type=iso_date, required=True, help="Realized-return date")
    g3.add_argument("--data-root", type=Path, default=Path("data"))
    g3.add_argument("--publish-current", action="store_true")
    g3_history = subparsers.add_parser(
        "run-g3-risk-only-history",
        help="Build and consolidate month-partitioned formal G3 risk-only history",
    )
    g3_history.add_argument("--start", type=iso_date, required=True, help="Realized-return date")
    g3_history.add_argument("--end", type=iso_date, required=True, help="Realized-return date")
    g3_history.add_argument("--data-root", type=Path, default=Path("data"))
    g3_history.add_argument("--publish-current", action="store_true")
    g4_attribution = subparsers.add_parser(
        "run-g4-attribution-diagnostics",
        help="Build immutable G4.0 Huber, R-squared, grouping and WLS diagnostics",
    )
    g4_attribution.add_argument("--data-root", type=Path, default=Path("data"))
    g4_attribution.add_argument(
        "--config", type=Path, default=Path("config/g4_validation_protocol_v1.json")
    )
    g4_attribution.add_argument(
        "--external-facts", type=Path, default=Path("config/external_facts_v1.json")
    )
    reconcile_runs = subparsers.add_parser(
        "reconcile-g3-runs", help="Build immutable numeric reconciliation between two G3 history runs"
    )
    reconcile_runs.add_argument("--data-root", type=Path, default=Path("data"))
    reconcile_runs.add_argument("--old-manifest", type=Path, required=True)
    reconcile_runs.add_argument("--new-manifest", type=Path, required=True)
    reconcile_runs.add_argument(
        "--config", type=Path, default=Path("config/g3_run_reconciliation_v1.json")
    )
    reconcile_current = subparsers.add_parser(
        "reconcile-g3-current",
        help="Build immutable row-waterfall reconciliation for the active G3/L1 lineage",
    )
    reconcile_current.add_argument("--data-root", type=Path, default=Path("data"))
    attest = subparsers.add_parser(
        "attest-g3-canonicalization",
        help="Write the append-only G3 canonicalization attestation and bind it to _CURRENT",
    )
    attest.add_argument("--data-root", type=Path, default=Path("data"))
    attest.add_argument(
        "--config", type=Path, default=Path("config/g3_canonicalization_attestation_v1.json")
    )
    g4_spectrum = subparsers.add_parser(
        "run-g4-statistical-spectrum",
        help="Build the preregistered G4.1 rolling MP residual eigenspectrum",
    )
    g4_spectrum.add_argument("--data-root", type=Path, default=Path("data"))
    g4_spectrum.add_argument(
        "--config", type=Path, default=Path("config/g4_validation_protocol_v1.json")
    )
    g4_alpha_overlap = subparsers.add_parser(
        "run-g4-statistical-alpha-overlap",
        help="Historical exactly-once loading semantic probe; not an alpha test",
    )
    g4_alpha_overlap.add_argument("--data-root", type=Path, default=Path("data"))
    g4_alpha_overlap.add_argument("--spectrum-manifest", type=Path, required=True)
    g4_alpha_overlap.add_argument(
        "--config", type=Path,
        default=Path("config/g4_statistical_alpha_overlap_v1.json"),
    )
    g4_style_probe = subparsers.add_parser(
        "run-g4-style-semantic-probe",
        help="Run the frozen second-and-final exposure-only style semantic probe",
    )
    g4_style_probe.add_argument("--data-root", type=Path, default=Path("data"))
    g4_style_probe.add_argument(
        "--config", type=Path,
        default=Path("config/g4_style_loading_semantic_probe_v1.json"),
    )
    g4_weight = subparsers.add_parser(
        "run-g4-wls-weight-selection",
        help="Run preregistered PIT WLS candidate selection without publication authority",
    )
    g4_weight.add_argument("--data-root", type=Path, default=Path("data"))
    g4_weight.add_argument(
        "--config", type=Path, default=Path("config/g4_validation_protocol_v1.json")
    )
    g4_freeze = subparsers.add_parser(
        "freeze-g4-wls-weight",
        help="Freeze the selected PIT WLS scheme and publish a formal G3 current",
    )
    g4_freeze.add_argument("--data-root", type=Path, default=Path("data"))
    g4_freeze.add_argument("--candidate-manifest", type=Path, required=True)
    g4_freeze.add_argument(
        "--config", type=Path, default=Path("config/g4_validation_protocol_v1.json")
    )
    g4_residual = subparsers.add_parser(
        "run-g4-structural-weight-residual",
        help="Quantify residual heteroskedasticity and weight clipping after G4.1a freeze",
    )
    g4_residual.add_argument("--data-root", type=Path, default=Path("data"))
    g4_residual.add_argument("--candidate-manifest", type=Path, required=True)
    g4_residual.add_argument("--freeze-manifest", type=Path, required=True)
    g4_residual.add_argument("--reconciliation-manifest", type=Path, required=True)
    g4_3 = subparsers.add_parser(
        "run-g4-3-stratified-residual-permutation",
        help="Run read-only G4.3 residual group-label permutation diagnostics",
    )
    g4_3.add_argument("--data-root", type=Path, default=Path("data"))
    g4_3.add_argument(
        "--config", type=Path, default=Path("config/g4_3_residual_permutation_v1.json")
    )
    g4_3_v2 = subparsers.add_parser(
        "run-g4-3-v2-stratified-residual-permutation",
        help="Run the draft/formal G4.3 v2 daily-PIT conditional permutation diagnostic",
    )
    g4_3_v2.add_argument("--data-root", type=Path, default=Path("data"))
    g4_3_v2.add_argument(
        "--config", type=Path, default=Path("config/g4_3_residual_permutation_v2.json")
    )
    pit_e2e = subparsers.add_parser(
        "run-pit-contract-e2e",
        help="Run read-only PIT t+1/t+5 alignment canaries and cross-sectional placebo",
    )
    pit_e2e.add_argument("--data-root", type=Path, default=Path("data"))
    pit_e2e.add_argument(
        "--config", type=Path, default=Path("config/pit_contract_e2e_v1.json")
    )
    universe = subparsers.add_parser(
        "universe-build",
        help="Persist a point-in-time daily universe without endpoint conditioning",
    )
    universe.add_argument("--start", type=iso_date, required=True)
    universe.add_argument("--end", type=iso_date, required=True)
    universe.add_argument("--data-root", type=Path, default=Path("data"))
    universe.add_argument(
        "--universe-policy", type=Path,
        default=Path("config/tradable_universe_v1.json"),
    )
    factor_data = subparsers.add_parser(
        "sync-factor-data",
        help="Sync annual PIT financials and a lagged Shibor risk-free series",
    )
    factor_data.add_argument("--financial-start-year", type=int, required=True)
    factor_data.add_argument("--financial-end-year", type=int, required=True)
    factor_data.add_argument("--risk-start", type=iso_date, required=True)
    factor_data.add_argument("--risk-end", type=iso_date, required=True)
    factor_data.add_argument("--data-root", type=Path, default=Path("data"))
    factor_data.add_argument(
        "--request-delay", type=float, default=0.15, help="Delay between per-asset requests"
    )
    factor_quality = subparsers.add_parser(
        "quality-factor-data", help="Check financial PIT and risk-free lookahead guards"
    )
    factor_quality.add_argument("--date", type=iso_date, help="Defaults to latest market date")
    factor_quality.add_argument("--data-root", type=Path, default=Path("data"))
    factor_quality.add_argument("--json", action="store_true")
    daily = subparsers.add_parser(
        "daily-update",
        help="Idempotently sync one China trading day and publish validated snapshots",
    )
    daily.add_argument("--date", type=iso_date, help="China market date; defaults to Beijing today")
    daily.add_argument("--data-root", type=Path, default=Path("data"))
    daily.add_argument("--force", action="store_true", help="Refetch and rebuild even if current")
    concepts = subparsers.add_parser(
        "sync-concepts",
        help="Run an explicit low-frequency full concept reconciliation",
    )
    concepts.add_argument("--date", type=iso_date, required=True)
    concepts.add_argument("--data-root", type=Path, default=Path("data"))
    concepts.add_argument("--force", action="store_true")
    concepts.add_argument(
        "--request-delay", type=float, default=0.36,
        help="Minimum seconds between Tushare request starts (default stays below 200/min)",
    )
    concepts.add_argument("--workers", type=int, default=8)
    api = subparsers.add_parser("serve-api", help="Serve small computed strategy objects locally")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8765)
    api.add_argument("--data-root", type=Path, default=Path("data"))
    api.add_argument(
        "--research-snapshot", type=Path, default=Path("data/metadata/research-summary.json")
    )
    industry = subparsers.add_parser(
        "sync-industry", help="Run an explicit low-frequency revisioned Shenwan refresh"
    )
    industry.add_argument("--data-root", type=Path, default=Path("data"))
    industry.add_argument("--standard", choices=("SW2014", "SW2021"), required=True)
    industry.add_argument("--request-delay", type=float, default=0.1)
    industry_quality_parser = subparsers.add_parser(
        "quality-industry", help="Check point-in-time industry coverage on one date"
    )
    industry_quality_parser.add_argument("--date", type=iso_date, required=True)
    industry_quality_parser.add_argument(
        "--level", choices=("L1", "L2", "L3"), default="L2"
    )
    industry_quality_parser.add_argument(
        "--standard", choices=("SW2014", "SW2021"), required=True
    )
    industry_quality_parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser
