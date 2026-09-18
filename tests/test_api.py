import json
import polars as pl
import pytest

from factor_matrix.api import (
    evaluation_framework_response,
    factor_catalog_response,
    pipeline_summary_response,
    risk_catalog_response,
    stock_scores_response,
)
from factor_matrix.storage import DataLake, file_sha256


def test_factor_and_risk_catalogs_keep_roles_and_values_separate() -> None:
    factors = factor_catalog_response(query="反转", page_size=10)
    risk = risk_catalog_response()
    assert factors["rows"] == []
    assert factors["total"] == 0
    assert factors["boundary"]["factor_values_sent"] is False
    assert all(row["role"] == "alpha_candidate" for row in factor_catalog_response()["rows"])
    assert {row["exposure_id"] for row in risk["rows"]} >= {"country", "size", "beta"}
    assert all(row["role"] != "alpha_candidate" for row in risk["rows"])
    style = risk_catalog_response(query="流动性", role="style_risk", page_size=5)
    assert style["total"] == 1
    assert style["rows"][0]["exposure_id"] == "liquidity"


def test_pipeline_summary_returns_counts_not_table_contents(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    lake.replace("prices_daily", pl.DataFrame({"asset_id": ["A", "B"]}))
    report_dir = lake.metadata / "quality_reports"
    report_dir.mkdir(parents=True)
    (report_dir / "full_history.json").write_text(json.dumps({
        "status": "passed", "summary": {"max_trade_date": "2026-08-13", "price_rows": 2}
    }))
    response = pipeline_summary_response(lake)
    prices = next(row for row in response["tables"] if row["id"] == "prices_daily")
    assert prices["rows"] == 2
    assert response["boundary"]["table_rows_sent"] is False


def test_score_api_requires_an_explicit_board(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    with pytest.raises(ValueError, match="BOARD_ID_REQUIRED"):
        stock_scores_response(lake, board_id="", limit=100)


def test_evaluation_framework_api_keeps_daily_series_server_side(tmp_path) -> None:
    lake = DataLake(tmp_path / "data")
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(json.dumps({
        "framework_id": "evaluation_test_v1",
        "status": "active",
        "component_declaration": {},
        "sample": {},
        "signal": {},
        "negative_controls": {},
        "construction_gates": {},
        "inference": {},
        "output_contract": {},
        "holdout_capacity": {},
    }))
    lake.metadata.mkdir(parents=True, exist_ok=True)
    (lake.metadata / "evaluation-framework-summary.json").write_text(json.dumps({
        "schema_version": 1,
        "framework_id": "evaluation_test_v1",
        "status": "completed",
        "config_sha256": file_sha256(config_path),
        "scope": {},
        "data_quality": {},
        "ic": {"by_horizon": {"1": {"mean": 0.01}}, "decay_curve": [], "daily_series": [1, 2]},
        "negative_controls": {"passed": True},
        "backtest": {},
    }))

    response = evaluation_framework_response(lake, config_path=config_path)

    assert response["status"] == "ready"
    assert "daily_series" not in response["latest_run"]["ic"]
    assert response["boundary"]["daily_ic_sent"] is False


@pytest.mark.parametrize(("overrides", "expected_status"), [
    ({"status": "superseded", "superseded_by": "new-channel"}, "superseded"),
    ({"framework_id": "old-framework"}, "stale"),
    ({"config_sha256": "old-config"}, "stale"),
    ({"config_sha256": None}, "stale"),
    ({"status": "blocked_negative_control"}, "blocked"),
    ({"negative_controls": {"passed": False}}, "blocked"),
    ({"negative_controls": {}}, "blocked"),
    ({"ic": None}, "blocked"),
])
def test_evaluation_api_does_not_expose_unusable_metrics(tmp_path, overrides, expected_status):
    lake = DataLake(tmp_path / "data")
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps({"framework_id": "test", "status": "active"}))
    payload = {
        "framework_id": "test", "status": "completed",
        "config_sha256": file_sha256(config),
        "scope": {}, "data_quality": {},
        "ic": {"by_horizon": {"5": {"mean": 999}}, "daily_series": [999]},
        "negative_controls": {"passed": True},
        "backtest": {"mean_turnover": 999},
        **overrides,
    }
    lake.metadata.mkdir(parents=True, exist_ok=True)
    summary = lake.metadata / "evaluation-framework-summary.json"
    summary.write_text(json.dumps(payload))
    before = summary.read_bytes()
    response = evaluation_framework_response(lake, config_path=config)
    assert response["status"] == expected_status
    assert response["latest_run"] is None
    assert response["excluded_run"]["reason"]
    assert "999" not in json.dumps(response)
    assert summary.read_bytes() == before  # inspection never rewrites history


def test_evaluation_api_distinguishes_no_run_from_unreadable_summary(tmp_path):
    lake = DataLake(tmp_path / "data")
    config = tmp_path / "evaluation.json"
    config.write_text(json.dumps({"framework_id": "test", "status": "active"}))
    response = evaluation_framework_response(lake, config_path=config)
    assert response["status"] == "registered_not_run"
    assert response["latest_run"] is None
    assert response["excluded_run"] is None
    lake.metadata.mkdir(parents=True, exist_ok=True)
    (lake.metadata / "evaluation-framework-summary.json").write_text("{partial")
    response = evaluation_framework_response(lake, config_path=config)
    assert response["status"] != "ready"
    assert response["latest_run"] is None
    assert response["excluded_run"] is not None
