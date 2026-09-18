from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from .storage import DataLake, utc_now
from .revisioned_silver import read_as_of_frame


def run_factor_data_quality(lake: DataLake, as_of_date: date) -> dict[str, Any]:
    def load(table: str) -> pl.DataFrame:
        try:
            return read_as_of_frame(lake, table)
        except FileNotFoundError:
            return pl.DataFrame()

    financial = load("financial_pit")
    income = load("income_pit")
    cashflow = load("cashflow_pit")
    risk_free = load("risk_free_daily")

    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        name: str,
        value: int,
        passed: bool,
        detail: str,
        *,
        blocking: bool = True,
    ) -> None:
        needs_review = not blocking and value > 0
        checks.append(
            {
                "check_id": check_id,
                "name": name,
                "result": "review" if needs_review else ("passed" if passed else "failed"),
                "result_label": "待复核" if needs_review else ("通过" if passed else "失败"),
                "failure_policy": "block_factor_pipeline" if blocking else "review_only",
                "failure_policy_label": "失败时阻断因子管道" if blocking else "安全排除并提示",
                "observed_value": value,
                "passed": passed,
                "detail": detail,
            }
        )

    add(
        "financial_rows_nonzero",
        "点时财务非空",
        financial.height,
        financial.height > 0,
        "必须先构建版本化 financial_pit",
    )
    if not financial.is_empty():
        duplicate_revisions = financial.height - financial.unique("revision_id").height
        announcement_before_period = financial.filter(
            pl.col("announcement_date").is_not_null()
            & (pl.col("announcement_date") < pl.col("report_period"))
        ).height
        available_before_announcement = financial.filter(
            pl.col("available_at").is_not_null()
            & pl.col("announcement_date").is_not_null()
            & (pl.col("available_at") <= pl.col("announcement_date"))
        ).height
        unsafe_revisions = financial.filter(
            (pl.col("revision_policy") == "conservative_revision_first_seen")
            & pl.col("available_at").is_not_null()
            & (pl.col("available_at") <= pl.col("first_seen_at").dt.date())
        ).height
        formula_mismatch = financial.filter(
            pl.col("book_equity").is_not_null()
            & ((
                pl.col("book_equity")
                - (
                    pl.col("equity_parent")
                    - pl.col("preferred_equity")
                    + pl.col("deferred_tax")
                )
            ).abs() > 0.01)
        ).height
        unavailable_versions = financial.filter(pl.col("available_at").is_null()).height
        future_versions = financial.filter(pl.col("available_at") > as_of_date).height
        add("duplicate_financial_revisions", "财务版本 ID 唯一", duplicate_revisions, duplicate_revisions == 0, "同一 revision_id 不得重复")
        add("announcement_after_period", "公告日不早于报告期", announcement_before_period, announcement_before_period == 0, "报告期结束前不得出现正式公告")
        add("available_after_announcement", "可用日严格晚于公告日", available_before_announcement, available_before_announcement == 0, "统一使用公告后的下一交易日")
        add("revision_first_seen_guard", "未知修订不回填历史", unsafe_revisions, unsafe_revisions == 0, "无法证明修订时间时，以首次抓取后的交易日为最早可用日")
        add("book_equity_formula", "账面权益公式可重算", formula_mismatch, formula_mismatch == 0, "母公司权益 - 优先股 + 递延税调整")
        add("unavailable_financial_versions", "尚无可用日的财务版本", unavailable_versions, True, "缺公告日或缺下一交易日时不得进入矩阵", blocking=False)
        add("future_financial_versions", "截止日后才可用的财务版本", future_versions, True, "保留在 PIT 表中，但在截止日矩阵中安全排除", blocking=False)

    for table_id, label, statement in (
        ("income", "利润表", income),
        ("cashflow", "现金流量表", cashflow),
    ):
        add(
            f"{table_id}_rows_nonzero",
            f"PIT {label}非空",
            statement.height,
            statement.height > 0,
            f"正式价值因子要求版本化 {table_id}_pit",
        )
        if statement.is_empty():
            continue
        duplicates = statement.height - statement.unique("revision_id").height
        early_announcement = statement.filter(
            pl.col("announcement_date").is_not_null()
            & (pl.col("announcement_date") < pl.col("report_period"))
        ).height
        unsafe_availability = statement.filter(
            pl.col("available_at").is_not_null()
            & pl.col("announcement_date").is_not_null()
            & (pl.col("available_at") <= pl.col("announcement_date"))
        ).height
        unsafe_revision = statement.filter(
            (pl.col("revision_policy") == "conservative_revision_first_seen")
            & pl.col("available_at").is_not_null()
            & (pl.col("available_at") <= pl.col("first_seen_at").dt.date())
        ).height
        future = statement.filter(pl.col("available_at") > as_of_date).height
        add(f"duplicate_{table_id}_revisions", f"{label}版本 ID 唯一", duplicates, duplicates == 0, "同一 revision_id 不得重复")
        add(f"{table_id}_announcement_after_period", f"{label}公告日不早于报告期", early_announcement, early_announcement == 0, "报告期结束前不得进入正式公告版本")
        add(f"{table_id}_available_after_announcement", f"{label}可用日严格晚于公告日", unsafe_availability, unsafe_availability == 0, "统一使用公告后的下一交易日")
        add(f"{table_id}_revision_first_seen_guard", f"{label}未知修订不回填历史", unsafe_revision, unsafe_revision == 0, "无法验证修订公告时至少延迟到首次观察后的交易日")
        add(f"future_{table_id}_versions", f"截止日后才可用的{label}版本", future, True, "保留但不允许进入截止日因子", blocking=False)

    add(
        "risk_free_rows_nonzero",
        "无风险收益率非空",
        risk_free.height,
        risk_free.height > 0,
        "必须构建 risk_free_daily",
    )
    if not risk_free.is_empty():
        future_fixings = risk_free.filter(pl.col("rate_date") >= pl.col("trade_date")).height
        duplicate_rates = risk_free.height - risk_free.unique(["trade_date", "rate_id"]).height
        add("strictly_prior_risk_free", "利率 fixing 严格滞后", future_fixings, future_fixings == 0, "t 日收益只能使用 t 日之前公布的利率")
        add("risk_free_primary_key", "无风险收益率主键唯一", duplicate_rates, duplicate_rates == 0, "交易日与利率口径组合不得重复")

    blocking_failures = sum(
        not check["passed"] and check["failure_policy"] == "block_factor_pipeline"
        for check in checks
    )
    review_items = sum(check["result"] == "review" for check in checks)
    status = "passed" if blocking_failures == 0 else "failed"
    report = {
        "schema_version": 1,
        "report_id": f"factor_data_{as_of_date.isoformat()}",
        "as_of_date": as_of_date.isoformat(),
        "created_at": utc_now().isoformat(),
        "status": status,
        "conclusion": {
            "result_label": "通过" if status == "passed" else "失败",
            "headline": "因子数据可进入矩阵" if status == "passed" else "因子数据仍被阻断",
            "blocking_failures": blocking_failures,
            "review_items": review_items,
        },
        "counts": {
            "financial_versions": financial.height,
            "financial_assets": financial.get_column("asset_id").n_unique() if not financial.is_empty() else 0,
            "income_versions": income.height,
            "cashflow_versions": cashflow.height,
            "risk_free_days": risk_free.height,
        },
        "checks": checks,
    }
    output_dir = lake.metadata / "quality_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"factor_data_{as_of_date.isoformat()}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    markdown_temporary = markdown.with_suffix(".md.tmp")
    markdown_temporary.write_text(render_factor_data_quality(report), encoding="utf-8")
    os.replace(markdown_temporary, markdown)
    return report


def render_factor_data_quality(report: dict[str, Any]) -> str:
    lines = [
        f"# 因子数据质量：{report['conclusion']['result_label']}",
        "",
        f"- 截止日期：{report['as_of_date']}",
        f"- 结论：{report['conclusion']['headline']}",
        f"- 阻断问题：{report['conclusion']['blocking_failures']} 项",
        f"- 待复核提示：{report['conclusion']['review_items']} 项",
        f"- 财务版本：{report['counts']['financial_versions']:,}",
        f"- 财务股票：{report['counts']['financial_assets']:,}",
        f"- 无风险利率交易日：{report['counts']['risk_free_days']:,}",
        "",
        "| 结果 | 检查项 | 观测值 | 规则 |",
        "|---|---|---:|---|",
    ]
    for check in sorted(report["checks"], key=lambda item: item["passed"]):
        lines.append(
            f"| {check['result_label']} | {check['name']} | {check['observed_value']} | {check['detail']} |"
        )
    return "\n".join(lines) + "\n"
