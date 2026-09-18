"""Expose only a completed evaluation under the current registered contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...storage import DataLake, file_sha256


DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "config/evaluation_framework_v1.json"


def evaluation_summary(lake: DataLake, config_path: Path | None = None) -> dict[str, Any]:
    from .evaluation_review_v2 import POINTER, evaluation_review_summary
    if config_path is None and (lake.metadata / POINTER).exists():
        return evaluation_review_summary(lake)
    path = config_path or DEFAULT_CONFIG
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get('framework_id') == 'estimation_evaluation_v2':
        return evaluation_review_summary(lake, config_path=path)
    contract_sha = file_sha256(path)
    source = lake.metadata / "evaluation-framework-summary.json"
    source_error = None
    latest_attempt = lake.metadata / "research_pipeline_latest_attempt.json"
    if latest_attempt.exists():
        try:
            publication = json.loads(latest_attempt.read_text(encoding="utf-8"))
            manifest_path = lake.root / publication["manifest"]
            if file_sha256(manifest_path) != publication["manifest_sha256"]:
                raise ValueError("manifest checksum mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = publication["summary"]
            if manifest["outputs"]["summary"] != record or manifest["run_id"] != publication["run_id"]:
                raise ValueError("publication lineage mismatch")
            source = lake.root / record["path"]
            if file_sha256(source) != record["sha256"]:
                raise ValueError("summary checksum mismatch")
        except (OSError, ValueError, KeyError, TypeError):
            source_error = "新流程的摘要或来源校验失败，结果已阻断。"
    latest = None
    excluded = None
    status = "registered_not_run"
    if source_error:
        status = "blocked"
        excluded = {"status": "blocked", "reason": source_error,
                    "source": "metadata/research_pipeline_latest_attempt.json"}
    elif source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        controls = payload.get("negative_controls")
        controls = controls if isinstance(controls, dict) else {}
        reason = None
        if payload.get("status") == "superseded":
            status, reason = "superseded", "旧评估结果已废弃，等待当前契约下的新结果。"
        elif payload.get("framework_id") != contract["framework_id"]:
            status, reason = "stale", "评估结果与当前框架版本不匹配。"
        elif payload.get("config_sha256") != contract_sha:
            status, reason = "stale", "评估结果未绑定当前配置，不能作为有效结果。"
        elif contract.get("status") != "active":
            status, reason = "blocked", "评估契约尚未启用。"
        elif payload.get("status") not in {"completed", "passed"} or controls.get("passed") is not True:
            status, reason = "blocked", "评估未完成或负对照未通过，主结果已阻断。"
            failures = [key for key, value in controls.get("variants", {}).items()
                        if isinstance(value, dict) and value.get("disposition") == "gate" and value.get("passed") is not True]
            if failures:
                reason += " 未通过：" + "、".join(failures) + "。"
        elif not all(isinstance(payload.get(key), dict) for key in ("scope", "ic", "data_quality")):
            status, reason = "blocked", "评估摘要不完整，不能展示主结果。"
        if reason is not None:
            excluded = {
                "status": payload.get("status"),
                "reason": reason,
                "superseded_by": payload.get("superseded_by"),
                "supersession_reason": payload.get("supersession_reason"),
                "source": str(source.relative_to(lake.root)),
            }
        else:
            status = "ready"
            latest = {
                "schema_version": payload.get("schema_version"),
                "framework_id": payload["framework_id"],
                "config_sha256": contract_sha,
                "status": payload["status"],
                "scope": payload["scope"],
                "data_quality": payload["data_quality"],
                "ic": {
                    "by_horizon": payload["ic"].get("by_horizon", {}),
                    "decay_curve": payload["ic"].get("decay_curve", []),
                },
                "negative_controls": controls,
                "backtest": payload.get("backtest"),
            }
    return {
        "schema_version": 1,
        "status": status,
        "framework_id": contract["framework_id"],
        "config_sha256": contract_sha,
        "contract": {
            key: contract.get(key, {}) for key in (
                "status", "component_declaration", "sample", "signal",
                "negative_controls", "evaluation_channel", "construction_gates",
                "inference", "output_contract", "holdout_capacity",
            )
        },
        "latest_run": latest,
        "excluded_run": excluded,
        "boundary": {
            "daily_ic_sent": False,
            "security_returns_sent": False,
            "factor_matrix_sent": False,
            "holdout_opened": contract.get("sample", {}).get("holdout_status") not in {
                None, "sealed_unopened",
            },
        },
    }
