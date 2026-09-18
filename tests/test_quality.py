from datetime import date

import polars as pl

from factor_matrix.quality import listed_as_of, render_quality_report, upgrade_legacy_report


def test_delist_date_is_first_non_listed_day() -> None:
    security = pl.DataFrame(
        {
            "asset_id": ["000627.SZ", "600000.SH"],
            "list_date": [date(1996, 11, 12), date(1999, 11, 10)],
            "delist_date": [date(2025, 9, 30), None],
        },
        schema_overrides={"delist_date": pl.Date},
    )
    result = listed_as_of(security, date(2025, 9, 30))
    assert result.get_column("asset_id").to_list() == ["600000.SH"]


def test_legacy_warning_is_rendered_as_review_not_error() -> None:
    report = upgrade_legacy_report(
        {
            "report_id": "daily_2026-08-06",
            "trade_date": "2026-08-06",
            "checks": [
                {
                    "check_id": "null_total_return",
                    "severity": "warning",
                    "value": 2,
                    "passed": True,
                    "detail": "legacy detail",
                }
            ],
        }
    )

    assert report["status"] == "passed"
    assert report["checks"][0]["result_label"] == "待复核"
    assert "待复核｜行情表原始空收益率：2" in render_quality_report(report)
    assert "severity" not in report["checks"][0]
