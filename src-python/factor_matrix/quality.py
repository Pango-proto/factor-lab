from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from .storage import DataLake, open_duckdb, utc_now
from .external_facts import ExternalFacts


CHECK_LABELS = {
    "calendar_open": "交易日有效",
    "price_rows_nonzero": "日行情非空",
    "price_primary_key": "行情主键唯一",
    "valuation_primary_key": "估值主键唯一",
    "listed_coverage": "上市股票覆盖完整",
    "inferred_nontrading_state": "双源缺失推定不可交易",
    "security_mapping": "证券代码映射完整",
    "valuation_coverage": "行情均有估值",
    "listing_day_valuation_missing": "上市首日估值缺失",
    "valuation_without_price": "无行情的估值记录",
    "out_of_scope_price_rows": "非上市域历史行情",
    "ohlc_relation": "开高低收关系正确",
    "adjustment_factor": "复权因子有效",
    "volume_amount": "成交量额有效",
    "reported_return": "原始收益率一致",
    "null_total_return": "行情表原始空收益率",
    "extreme_total_return": "极端收益率记录",
    "market_cap": "市值数据有效",
    "return_primary_key": "收益率主键唯一",
    "suspension_zero_return": "停牌日收益率为零",
    "actionable_return_coverage": "停复牌收益率完整",
    "missing_open_dates": "交易日连续完整",
    "duplicate_prices": "历史行情主键唯一",
    "duplicate_valuations": "历史估值主键唯一",
    "missing_adjustment_factor": "历史复权因子完整",
    "invalid_ohlc": "历史开高低收有效",
    "price_valuation_mismatch": "历史行情估值匹配",
    "listed_coverage_missing": "历史上市股票覆盖完整",
    "deprecated_price_columns": "生产价格视图无前复权锚点字段",
    "old_bse_codes": "北交所代码已统一",
    "duplicate_returns": "历史收益率主键唯一",
    "nonzero_suspension_returns": "历史停牌收益率为零",
    "unexplained_null_returns": "历史空收益率均可解释",
}

CHECK_DETAILS = {
    "calendar_open": "目标日期必须是开市交易日",
    "price_rows_nonzero": "行情接口必须返回至少一条记录",
    "price_primary_key": "同一股票同一交易日不能出现重复行情",
    "valuation_primary_key": "同一股票同一交易日不能出现重复估值",
    "listed_coverage": "当日上市股票必须有行情或停牌记录",
    "security_mapping": "每条行情都必须能映射到证券主表",
    "valuation_coverage": "每条行情都必须有对应估值记录",
    "valuation_without_price": "记录有估值但没有行情的股票数量",
    "ohlc_relation": "最低价不得高于开盘价或收盘价，最高价不得低于两者",
    "adjustment_factor": "复权因子必须存在且大于零",
    "volume_amount": "成交量和成交额不得为负数",
    "reported_return": "按收盘价计算的涨跌幅应与供应商字段一致",
    "null_total_return": "行情表按相邻交易日计算，空值通常来自新股首日或停牌衔接；因子用收益表会另行处理停复牌",
    "extreme_total_return": "累计复权因子推导的总收益率绝对值超过 35%，建议人工抽查",
    "market_cap": "总市值和流通市值必须存在且大于零",
    "return_primary_key": "同一股票同一交易日不能出现重复收益率",
    "suspension_zero_return": "纯停牌日的收益率必须为零",
    "actionable_return_coverage": "停牌和复牌事件必须具有明确的收益率处理结果",
}


def _check(
    check_id: str,
    level: str,
    value: int | float,
    passed: bool,
    detail: str = "",
) -> dict[str, Any]:
    """Build an unambiguous check record for both people and programs."""
    needs_review = level != "error" and passed and value != 0
    return {
        "check_id": check_id,
        "name": CHECK_LABELS.get(check_id, check_id),
        "result": "review" if needs_review else ("passed" if passed else "failed"),
        "result_label": "待复核" if needs_review else ("通过" if passed else "失败"),
        "failure_policy": "block_pipeline" if level == "error" else "review_only",
        "failure_policy_label": "失败时阻断管道" if level == "error" else "仅提示人工复核",
        "observed_value": value,
        "passed": passed,
        "detail": CHECK_DETAILS.get(check_id, detail),
    }


def _add_conclusion(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks", [])
    blocking_failed = sum(
        not item["passed"] and item["failure_policy"] == "block_pipeline"
        for item in checks
    )
    review_items = sum(
        item["failure_policy"] == "review_only" and item["observed_value"] != 0
        for item in checks
    )
    passed = blocking_failed == 0
    report["status"] = "passed" if passed else "failed"
    report["conclusion"] = {
        "result": report["status"],
        "result_label": "通过" if passed else "失败",
        "headline": (
            "质量闸门通过，可以继续后续处理"
            if passed
            else f"质量闸门失败：发现 {blocking_failed} 项阻断问题"
        ),
        "blocking_failures": blocking_failed,
        "review_items": review_items,
    }
    return report


def render_quality_report(report: dict[str, Any]) -> str:
    """Render a short Chinese report with problems before successful checks."""
    conclusion = report["conclusion"]
    scope = report.get("trade_date")
    if not scope:
        summary = report.get("summary", {})
        scope = f"{summary.get('min_trade_date', '?')} 至 {summary.get('max_trade_date', '?')}"
    lines = [
        f"# 数据质量报告：{conclusion['result_label']}",
        "",
        f"- 检查范围：{scope}",
        f"- 总结：{conclusion['headline']}",
        f"- 阻断问题：{conclusion['blocking_failures']} 项",
        f"- 待复核提示：{conclusion['review_items']} 项",
    ]
    counts = report.get("counts")
    if counts:
        count_labels = {
            "listed": "上市",
            "prices": "行情",
            "valuations": "估值",
            "suspended": "停牌",
            "st": "ST",
            "returns": "收益率",
        }
        count_summary = "｜".join(
            f"{label} {counts[key]:,}"
            for key, label in count_labels.items()
            if key in counts
        )
        lines.extend(
            [
                "",
                "## 当日数据量",
                "",
                count_summary,
            ]
        )
    problems = [
        item
        for item in report["checks"]
        if not item["passed"]
        or (item["failure_policy"] == "review_only" and item["observed_value"] != 0)
    ]
    lines.extend(["", "## 异常与提示", ""])
    if not problems:
        lines.append("无。")
    else:
        sample_keys = {
            "listed_coverage": "missing_coverage",
            "security_mapping": "unmapped_prices",
            "valuation_coverage": "missing_valuation",
            "null_total_return": "null_total_return",
            "extreme_total_return": "extreme_total_return",
        }
        for item in problems:
            lines.append(
                f"- {item['result_label']}｜{item['name']}：{item['observed_value']}（{item['failure_policy_label']}）"
            )
            if item.get("detail"):
                lines.append(f"  - 说明：{item['detail']}")
            samples = report.get("samples", {}).get(sample_keys.get(item["check_id"], ""), [])
            if samples:
                lines.append(f"  - 样本：{', '.join(map(str, samples))}")
    lines.extend(["", "## 全部检查", "", "| 结果 | 检查项 | 观测值 | 失败处理 |", "|---|---|---:|---|"])
    for item in sorted(report["checks"], key=lambda row: row["passed"]):
        lines.append(
            f"| {item['result_label']} | {item['name']} | {item['observed_value']} | {item['failure_policy_label']} |"
        )
    return "\n".join(lines) + "\n"


def _write_report(lake: DataLake, filename: str, report: dict[str, Any]) -> None:
    output_dir = lake.metadata / "quality_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    markdown_tmp = markdown.with_suffix(".md.tmp")
    markdown_tmp.write_text(render_quality_report(report), encoding="utf-8")
    os.replace(markdown_tmp, markdown)


def upgrade_legacy_report(report: dict[str, Any]) -> dict[str, Any]:
    """Convert a saved schema-v1 report without rerunning expensive data scans."""
    report["schema_version"] = 2
    report["checks"] = [
        _check(
            item["check_id"],
            (
                "warning"
                if item.get("failure_policy") == "review_only"
                else item.get("severity", "error")
            ),
            item.get("observed_value", item.get("value", 0)),
            item["passed"],
            item.get("detail", ""),
        )
        for item in report.get("checks", [])
    ]
    return _add_conclusion(report)


def _assets(frame: pl.DataFrame) -> set[str]:
    if frame.is_empty() or "asset_id" not in frame.columns:
        return set()
    return set(frame.get_column("asset_id").drop_nulls().to_list())


def listed_as_of(security: pl.DataFrame, trade_date: date) -> pl.DataFrame:
    """Securities tradable on a date; delist_date is the first non-listed day."""
    return security.filter(
        (pl.col("list_date") <= trade_date)
        & (pl.col("delist_date").is_null() | (pl.col("delist_date") > trade_date))
    )


def run_daily_quality(lake: DataLake, trade_date: date) -> dict[str, Any]:
    catalog = lake.refresh_catalog()
    connection = open_duckdb(catalog, read_only=True)
    escaped_date = trade_date.isoformat()

    def query(sql: str) -> pl.DataFrame:
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
        return pl.DataFrame(rows, schema=columns, orient="row", infer_schema_length=None)

    try:
        prices = query(f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM prices_daily WHERE trade_date=DATE '{escaped_date}'")
        values = query(f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM valuation_daily WHERE trade_date=DATE '{escaped_date}'")
        security = query("SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM security_master")
        suspensions = query(
            f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM suspensions_daily WHERE trade_date=DATE '{escaped_date}'"
        )
        st = query(f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM stock_st_daily WHERE trade_date=DATE '{escaped_date}'")
        calendar = query(
            f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM trade_calendar WHERE cal_date=DATE '{escaped_date}'"
        )
        returns = query(f"SELECT * EXCLUDE(revision_at,first_seen_at,ingested_at) FROM returns_daily WHERE trade_date=DATE '{escaped_date}'")
        prior_price_assets = {
            row[0] for row in connection.execute(
                f"SELECT DISTINCT asset_id FROM prices_daily WHERE trade_date<DATE '{escaped_date}'"
            ).fetchall()
        }
    finally:
        connection.close()
    if "total_return" not in prices.columns:
        prices = prices.join(
            returns.select("trade_date", "asset_id", "total_return"),
            on=["trade_date", "asset_id"], how="left", validate="1:1",
        )

    facts = ExternalFacts.load(
        Path(__file__).resolve().parents[2] / "config" / "external_facts_v1.json"
    )
    if not facts.scope_active("regime_break", "BSE", trade_date):
        def without_pre_bse(frame: pl.DataFrame) -> pl.DataFrame:
            if frame.is_empty() or "asset_id" not in frame.columns:
                return frame
            return frame.filter(~pl.col("asset_id").str.ends_with(".BJ"))
        prices = without_pre_bse(prices)
        values = without_pre_bse(values)
        suspensions = without_pre_bse(suspensions)
        st = without_pre_bse(st)
        returns = without_pre_bse(returns)
        security = without_pre_bse(security)

    listed = listed_as_of(security, trade_date)
    listed_assets = _assets(listed)
    price_assets = _assets(prices)
    value_assets = _assets(values)
    suspended_assets = _assets(suspensions)

    checks: list[dict[str, Any]] = []

    def add(check_id: str, severity: str, value: int | float, passed: bool, detail: str) -> None:
        checks.append(_check(check_id, severity, value, passed, detail))

    is_open_rows = calendar.filter(pl.col("cal_date") == trade_date)
    is_open = (
        bool(is_open_rows.get_column("is_open")[0]) if is_open_rows.height == 1 else False
    )
    add("calendar_open", "error", int(is_open), is_open, "Date must be an open market day")
    add("price_rows_nonzero", "error", prices.height, prices.height > 0, "Daily endpoint must return rows")

    duplicate_prices = prices.height - prices.unique(["trade_date", "asset_id"]).height
    duplicate_values = values.height - values.unique(["trade_date", "asset_id"]).height
    add("price_primary_key", "error", duplicate_prices, duplicate_prices == 0, "Duplicate price keys")
    add("valuation_primary_key", "error", duplicate_values, duplicate_values == 0, "Duplicate valuation keys")

    missing_coverage = listed_assets - price_assets - suspended_assets
    unmapped_prices = price_assets - set(security.get_column("asset_id").to_list())
    inferred_nontrading = missing_coverage - value_assets
    inconsistent_missing = missing_coverage & value_assets
    add(
        "listed_coverage",
        "error",
        len(inconsistent_missing),
        len(inconsistent_missing) == 0,
        "A listed asset with valuation but without price/suspension is inconsistent",
    )
    add(
        "inferred_nontrading_state", "warning", len(inferred_nontrading), True,
        "No daily and no daily_basic record: retain as non-tradable even if suspend_d history is absent",
    )
    add(
        "security_mapping",
        "error",
        len(unmapped_prices),
        len(unmapped_prices) == 0,
        "Every price asset must map to security master",
    )

    in_scope_price_assets = price_assets & listed_assets
    missing_values = in_scope_price_assets - value_assets
    listing_day_assets = _assets(listed.filter(pl.col("list_date") == trade_date))
    listing_day_missing_values = missing_values & listing_day_assets
    unexplained_missing_values = missing_values - listing_day_assets
    extra_values = (value_assets & listed_assets) - price_assets
    add(
        "valuation_coverage", "error", len(unexplained_missing_values),
        len(unexplained_missing_values) == 0,
        "Non-listing-day prices must have valuation",
    )
    add(
        "listing_day_valuation_missing", "warning", len(listing_day_missing_values), True,
        "Listing-day daily_basic may be absent; asset remains non-tradable during listing-age exclusion",
    )
    add("valuation_without_price", "warning", len(extra_values), True, "Valuation rows without daily price")
    add(
        "out_of_scope_price_rows", "warning", len(price_assets - listed_assets), True,
        "Retained raw venue history before A-share listing; excluded from valuation coverage and tradability",
    )

    if prices.height:
        invalid_ohlc = prices.filter(
            (pl.col("low") > pl.min_horizontal("open", "close"))
            | (pl.col("high") < pl.max_horizontal("open", "close"))
            | (pl.col("low") > pl.col("high"))
        ).height
        missing_adj = prices.filter(pl.col("adj_factor").is_null() | (pl.col("adj_factor") <= 0)).height
        negative_flow = prices.filter((pl.col("volume") < 0) | (pl.col("amount") < 0)).height
        pct_mismatch = prices.filter(
            (
                ((pl.col("close") / pl.col("prev_close") - 1) * 100 - pl.col("pct_chg"))
                .abs()
                > 0.03
            )
            & pl.col("prev_close").is_not_null()
            & (pl.col("prev_close") > 0)
        ).height
        null_returns = prices.filter(pl.col("total_return").is_null()).height
        extreme_returns = prices.filter(pl.col("total_return").abs() > 0.35).height
        add("ohlc_relation", "error", invalid_ohlc, invalid_ohlc == 0, "OHLC ordering violations")
        add("adjustment_factor", "error", missing_adj, missing_adj == 0, "Missing/non-positive adj_factor")
        add("volume_amount", "error", negative_flow, negative_flow == 0, "Negative volume or amount")
        add("reported_return", "error", pct_mismatch, pct_mismatch == 0, "Raw close return differs from pct_chg")
        add("null_total_return", "warning", null_returns, True, "Usually IPO, gap, or suspended history")
        add("extreme_total_return", "warning", extreme_returns, True, "Absolute adjusted return above 35%")
        null_return_samples = (
            prices.filter(pl.col("total_return").is_null())
            .get_column("asset_id")
            .head(20)
            .to_list()
        )
        extreme_return_samples = (
            prices.filter(pl.col("total_return").abs() > 0.35)
            .get_column("asset_id")
            .head(20)
            .to_list()
        )
    else:
        null_return_samples = []
        extreme_return_samples = []

    if values.height:
        invalid_caps = values.filter(
            pl.col("total_mkt_cap").is_null()
            | pl.col("float_mkt_cap").is_null()
            | (pl.col("total_mkt_cap") <= 0)
            | (pl.col("float_mkt_cap") <= 0)
        ).height
        add("market_cap", "error", invalid_caps, invalid_caps == 0, "Missing/non-positive market caps")

    if not returns.is_empty():
        duplicate_returns = returns.height - returns.unique(["trade_date", "asset_id"]).height
        nonzero_suspension = returns.filter(
            pl.col("is_suspended")
            & pl.col("total_return").is_not_null()
            & (pl.col("total_return").abs() > 1e-12)
        ).height
        missing_actionable_returns = returns.filter(
            pl.col("total_return").is_null()
            & (pl.col("return_source") != "price")
            & pl.col("asset_id").is_in(list(prior_price_assets))
        ).height
        add("return_primary_key", "error", duplicate_returns, duplicate_returns == 0, "Duplicate return keys")
        add(
            "suspension_zero_return",
            "error",
            nonzero_suspension,
            nonzero_suspension == 0,
            "Pure suspension days must have zero return",
        )
        add(
            "actionable_return_coverage",
            "error",
            missing_actionable_returns,
            missing_actionable_returns == 0,
            "Suspension/resumption rows require explicit returns once a prior traded price exists",
        )

    report = {
        "schema_version": 2,
        "report_id": f"daily_{trade_date.isoformat()}",
        "trade_date": trade_date.isoformat(),
        "created_at": utc_now().isoformat(),
        "counts": {
            "listed": len(listed_assets),
            "prices": len(price_assets),
            "valuations": len(value_assets),
            "suspended": len(suspended_assets),
            "st": len(_assets(st)),
            "returns": returns.height,
        },
        "samples": {
            "missing_coverage": sorted(missing_coverage)[:20],
            "unmapped_prices": sorted(unmapped_prices)[:20],
            "missing_valuation": sorted(missing_values)[:20],
            "null_total_return": null_return_samples,
            "extreme_total_return": extreme_return_samples,
        },
        "checks": checks,
    }
    _add_conclusion(report)
    _write_report(lake, f"daily_{trade_date.isoformat()}.json", report)
    return report


def run_history_quality(lake: DataLake) -> dict[str, Any]:
    facts = ExternalFacts.load(
        Path(__file__).resolve().parents[2] / "config" / "external_facts_v1.json"
    )
    bse_start = next(
        fact.effective_from for fact in facts.facts
        if fact.fact_key == "regime_break" and fact.scope == "BSE"
    ).isoformat()
    catalog = lake.metadata / "catalog.duckdb"
    con = open_duckdb(catalog, read_only=True)
    try:
        summary = con.execute(
            "select min(trade_date),max(trade_date),count(*),count(distinct trade_date),count(distinct asset_id) from prices_daily"
        ).fetchone()
        checks_sql = {
            "missing_open_dates": "with bounds as (select min(trade_date) lo,max(trade_date) hi from prices_daily) select count(*) from trade_calendar c,bounds b where c.is_open=1 and c.cal_date between b.lo and b.hi and not exists(select 1 from prices_daily p where p.trade_date=c.cal_date)",
            "duplicate_prices": "select count(*) from (select trade_date,asset_id from prices_daily group by 1,2 having count(*)>1)",
            "duplicate_valuations": "select count(*) from (select trade_date,asset_id from valuation_daily group by 1,2 having count(*)>1)",
            "missing_adjustment_factor": "select count(*) from prices_daily where adj_factor is null or adj_factor<=0",
            "invalid_ohlc": "select count(*) from prices_daily where low>least(open,close) or high<greatest(open,close) or low>high",
            "price_valuation_mismatch": f"with securities as (select asset_id,min(list_date) list_date,max(delist_date) delist_date from security_master group by 1), in_scope_prices as (select p.trade_date,p.asset_id,s.list_date from prices_daily p join securities s on s.asset_id=p.asset_id and s.list_date<=p.trade_date and (s.delist_date is null or s.delist_date>p.trade_date) and (right(p.asset_id,3)<>'.BJ' or p.trade_date>=date '{bse_start}')) select count(*) from in_scope_prices p left join valuation_daily v using(trade_date,asset_id) where v.asset_id is null and p.trade_date<>p.list_date",
            "listed_coverage_missing": "with bounds as (select min(trade_date) lo,max(trade_date) hi from prices_daily), d as (select distinct cal_date from trade_calendar,bounds where is_open=1 and cal_date between lo and hi), securities as (select asset_id,min(list_date) list_date,max(delist_date) delist_date from security_master group by 1), expected as (select d.cal_date,s.asset_id from d join securities s on s.list_date<=d.cal_date and (s.delist_date is null or s.delist_date>d.cal_date)), covered as (select trade_date,asset_id from prices_daily union select trade_date,asset_id from suspensions_daily) select count(*) from expected e join valuation_daily v on v.trade_date=e.cal_date and v.asset_id=e.asset_id left join covered c on c.trade_date=e.cal_date and c.asset_id=e.asset_id where c.asset_id is null",
            "deprecated_price_columns": "select count(*) from pragma_table_info('prices_daily') where name in ('qfq_close','qfq_anchor_date')",
            "old_bse_codes": f"select count(*) from prices_daily where trade_date>=date '{bse_start}' and right(asset_id,3)='.BJ' and left(asset_id,1) in ('4','8')",
            "duplicate_returns": "select count(*) from (select trade_date,asset_id from returns_daily group by 1,2 having count(*)>1)",
            "nonzero_suspension_returns": "select count(*) from returns_daily where is_suspended and abs(coalesce(total_return,0))>1e-12",
            "unexplained_null_returns": "select count(*) from returns_daily r where r.total_return is null and exists(select 1 from returns_daily q where q.asset_id=r.asset_id and q.trade_date<r.trade_date and q.return_source in ('price','resumption'))",
        }
        checks = []
        warning_checks = {"price_valuation_mismatch"}
        for check_id, sql in checks_sql.items():
            value = int(con.execute(sql).fetchone()[0])
            checks.append(
                _check(check_id, "warning" if check_id in warning_checks else "error", value, value == 0)
            )
    finally:
        con.close()
    report = {
        "schema_version": 2,
        "report_id": "full_history",
        "created_at": utc_now().isoformat(),
        "summary": {
            "min_trade_date": summary[0].isoformat(),
            "max_trade_date": summary[1].isoformat(),
            "price_rows": summary[2],
            "trading_days": summary[3],
            "assets": summary[4],
        },
        "checks": checks,
    }
    _add_conclusion(report)
    _write_report(lake, "full_history.json", report)
    return report
