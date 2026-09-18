#!/usr/bin/env python3
"""CI guard for definition single-sourcing and frozen L2a choices."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


_SUBPROCESS_METHODS = {"run", "Popen", "call", "check_call", "check_output"}


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return f"{function.value.id}.{function.attr}"
    return None


def _string_literals(node: ast.AST) -> list[str]:
    return [
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def orchestration_violations(project: Path = PROJECT) -> list[str]:
    """Reject new script-to-script orchestration edges.

    Existing standalone scripts remain compatible. The invariant only blocks
    the dependency shape that makes a later scheduler migration unsafe:
    scripts invoking other repository scripts or shell entrypoints. Calling
    the installed ``factor-matrix`` CLI and local development tools remains
    allowed.
    """

    found: list[str] = []
    scripts_dir = project / "scripts"
    for path in sorted(scripts_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            found.append(f"{path.relative_to(project)}: syntax error: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else next((alias.name for alias in node.names), "")
                )
                if imported == "scripts" or imported.startswith("scripts."):
                    found.append(
                        f"{path.relative_to(project)}:{node.lineno}: imports another pipeline script"
                    )

            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            if call_name == "os.system":
                found.append(
                    f"{path.relative_to(project)}:{node.lineno}: os.system is not an approved pipeline boundary"
                )
            if call_name in {f"subprocess.{method}" for method in _SUBPROCESS_METHODS}:
                literals = _string_literals(node)
                if any(
                    re.search(r"(?:^|/)scripts/|(?:^|/)[^/]+\.sh$|python\s+scripts/", value)
                    for value in literals
                ):
                    found.append(
                        f"{path.relative_to(project)}:{node.lineno}: subprocess targets another pipeline script"
                    )

    for path in sorted(scripts_dir.glob("*.sh")):
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if re.search(r"(?:^|[;&|])\s*(?:bash|sh)\s+.*(?:scripts/|\.sh)", line):
                found.append(
                    f"{path.relative_to(project)}:{line_number}: shell script invokes another pipeline script"
                )
            elif re.search(r"(?:^|[;&|])\s*(?:[^\s]+/)?scripts/[^\s]+\.py", line):
                found.append(
                    f"{path.relative_to(project)}:{line_number}: shell script invokes another pipeline script"
                )
    return found


def violations(project: Path = PROJECT) -> list[str]:
    found: list[str] = orchestration_violations(project)
    calculation = project / "src-python" / "factor_matrix" / "calculation"
    forbidden_text = {
        "float_share *": "raw float market-cap reconstruction",
        'maximum_condition_number": 1000': "hard-coded condition threshold",
        'residual_sigma": 3.0': "hard-coded outlier threshold",
        "legacy_l1:": "legacy risk-basis fallback remains reachable",
        "legacy_g3:": "legacy risk-basis fallback remains reachable",
    }
    for path in calculation.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle, reason in forbidden_text.items():
            if needle in text:
                found.append(f"{path.relative_to(project)}: {reason}")

    canonical_text = (
        project / "src-python" / "factor_matrix" / "canonical_definitions.py"
    ).read_text(encoding="utf-8")
    for symbol in ("REGRESSION_BASE_WEIGHT", "EXPOSURE_ORTHOGONALIZATION_WEIGHT"):
        if symbol not in canonical_text:
            found.append(f"canonical weight semantic missing: {symbol}")
    l1_builder_text = (calculation / "l1" / "builder.py").read_text(encoding="utf-8")
    if "weights = [math.sqrt(value)" in l1_builder_text:
        found.append("L1 recreates the regression/orthogonalization weight from market cap")
    weight_contract = json.loads(
        (project / "config" / "weight_metric_contract_v1.json").read_text(encoding="utf-8")
    )
    if weight_contract.get("binding") != "same_pit_weight_artifact":
        found.append("formal exposure and regression metrics are not artifact-bound")

    runner_path = calculation / "l2" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_single_factor":
            parameters = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            if "horizon_days" in parameters:
                found.append("L2a run_single_factor accepts runtime horizon_days")

    protocol = json.loads((project / "config" / "research_protocol_v1.json").read_text())
    holdout = protocol["frozen_choices"]["holdout"]
    if not holdout["sealed"] or holdout["opened"]:
        found.append("holdout is not sealed and unopened")
    if "horizon_days" in protocol["targets"]:
        found.append("evaluation horizon appears in derivable targets")

    legacy_pipeline = project / "src-python" / "factor_matrix" / "pipeline.py"
    if legacy_pipeline.exists():
        found.append("legacy mutable MarketPipeline module still exists")
    legacy_risk_plugins = (
        project / "src-python" / "factor_matrix" / "factor_engine" / "plugins"
        / "risk_candidates.py"
    )
    if legacy_risk_plugins.exists():
        found.append("monolithic risk_candidates.py still exists")
    l1_config = json.loads(
        (project / "config" / "l1_risk_exposure_v1.json").read_text(encoding="utf-8")
    )
    if (
        l1_config.get("regression_universe") != "all_a_share"
        or l1_config.get("per_board_regression") != "forbidden"
    ):
        found.append("L1 regression universe is not frozen to all A shares")
    label_policy = json.loads(
        (project / "config" / "return_label_policy_v1.json").read_text(encoding="utf-8")
    )
    if any(
        label_policy["l2b"][event]["action"] != "exclude"
        for event in ("exchange_first_day", "resumption_day")
    ):
        found.append("L2b event-return exclusions are not frozen")

    store_text = (
        project / "src-python" / "factor_matrix" / "factor_engine" / "store.py"
    ).read_text(encoding="utf-8")
    required_registry_tokens = {
        "CREATE TABLE IF NOT EXISTS risk_set": "risk_set header table missing",
        "CREATE TABLE IF NOT EXISTS research_attempt": "research attempt ledger missing",
        "CREATE TABLE IF NOT EXISTS feature_assignment": "explicit feature assignment missing",
        "CREATE TABLE IF NOT EXISTS risk_set_expansion": "risk expansion lineage missing",
        "ALPHA_ASSERTION_SELF_NEUTRALIZATION_FORBIDDEN": "cross-role exclusion trigger missing",
        "FROZEN_RISK_SET_MEMBER_IMMUTABLE": "frozen member immutability missing",
    }
    for needle, reason in required_registry_tokens.items():
        if needle not in store_text:
            found.append(reason)
    if "UPDATE alpha_assertion SET variant_count" in store_text:
        found.append("alpha_assertion is still being used as the variant ledger")
    if "m.feature_version = NEW.feature_version" in store_text:
        found.append("self-neutralization scope incorrectly includes feature_version")
    research_log_text = (
        project / "src-python" / "factor_matrix" / "calculation" / "l2" / "research_log.py"
    ).read_text(encoding="utf-8")
    if "attempt_id" not in research_log_text:
        found.append("L2a research event is not bound to attempt_id")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "run_single_factor", "run_single_factor_with_neutralization_fidelity",
            "run_factor_evaluation_snapshot",
        }:
            parameters = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            if "attempt_id" not in parameters:
                found.append(f"{node.name} does not require attempt_id")

    # Resolution is descriptive only.  The FDR denominator must count the
    # append-only attempt ledger directly, including unsubmitted attempts.
    predictivity_path = calculation / "l2" / "predictivity.py"
    predictivity_tree = ast.parse(predictivity_path.read_text(encoding="utf-8"))
    for node in ast.walk(predictivity_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "fdr_denominator":
            body_text = ast.get_source_segment(predictivity_path.read_text(encoding="utf-8"), node) or ""
            if "research_attempt_resolution" in body_text:
                found.append("FDR denominator reads descriptive research_attempt_resolution")

    # The legacy parquet entry point may remain for fixtures, but its builder
    # must be an adapter into the one canonical categorical constraint path.
    design_path = calculation / "l2" / "design_matrix.py"
    design_tree = ast.parse(design_path.read_text(encoding="utf-8"))
    adapter = next(
        (node for node in ast.walk(design_tree)
         if isinstance(node, ast.FunctionDef)
         and node.name == "build_daily_industry_constrained_design"),
        None,
    )
    if adapter is None or not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_daily_categorical_constrained_design"
        for node in ast.walk(adapter)
    ):
        found.append("parquet industry design is not a thin adapter to canonical categorical constraints")

    expansion_path = project / "config" / "risk_set_expansion_v1.json"
    if not expansion_path.exists():
        found.append("risk_set_expansion manifest missing")
    else:
        expansion = json.loads(expansion_path.read_text(encoding="utf-8"))
        columns = [
            column
            for item in expansion.get("logical_members", ())
            for column in item.get("generated_columns", ())
        ]
        if len(columns) != expansion.get("expanded_factor_count"):
            found.append("risk_set_expansion K does not equal generated column count")
        if len(columns) != len(set(columns)):
            found.append("risk_set_expansion contains duplicate generated columns")

    g4_protocol = json.loads(
        (project / "config" / "g4_validation_protocol_v1.json").read_text(encoding="utf-8")
    )
    sample = g4_protocol.get("diagnostic_day_sample", {})
    if (
        sample.get("n_days") != 250
        or sample.get("seed") != 20260817
        or sample.get("paired") is not True
        or sample.get("all_candidates_share_sample") is not True
        or sample.get("sequential_early_stopping") is not False
        or sample.get("frozen_before_candidate_evaluation") is not True
    ):
        found.append("diagnostic day sample is not frozen as a paired selection sample")

    cli_text = (project / "src-python" / "factor_matrix" / "cli.py").read_text(encoding="utf-8")
    parser_text = (project / "src-python" / "factor_matrix" / "cli_parser.py").read_text(
        encoding="utf-8"
    )
    for command in ("sync-market", "backfill-market", "sync-price-limits", "matrix-build"):
        if command in cli_text or command in parser_text:
            found.append(f"legacy mutable command remains reachable: {command}")
    l0_modules = (
        "incremental_market.py", "board_benchmarks.py", "boards.py", "classification.py",
        "factor_data.py", "concepts.py", "benchmark_membership.py",
    )
    for name in l0_modules:
        path = project / "src-python" / "factor_matrix" / name
        text = path.read_text(encoding="utf-8")
        if any(
            needle in text
            for needle in ("self.lake.upsert(", "self.lake.replace(", "lake.upsert(", "lake.replace(")
        ):
            found.append(f"{path.relative_to(project)}: mutable Silver writer in L0 production")
    return found

FORBIDDEN_LEGACY_TABLE_REFERENCES = {
    "legacy_factor_registry",
    "risk_set_version",          # 旧表；新表是 risk_set + risk_set_member
    "feature_assignment_legacy",
}

def main() -> None:
    found = violations()
    if found:
        raise SystemExit("\n".join(found))
    print("architecture invariants passed")


if __name__ == "__main__":
    main()
