"""Reusable DuckDB query helpers for versioned history diagnostics."""

from __future__ import annotations

from typing import Any


def rows(connection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, params or [])
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def one(connection, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
    result = rows(connection, sql, params)
    if len(result) != 1:
        raise RuntimeError(f"G0_EXPECTED_ONE_ROW observed={len(result)}")
    return result[0]


def return_distribution(
    connection,
    relation: str,
    predicate: str,
    params: list[Any],
    warning_threshold: float,
    extreme_threshold: float,
) -> dict[str, Any]:
    return one(connection, f"""
        SELECT count(*) AS row_count,
               count(total_return) AS non_null_count,
               min(total_return) AS minimum,
               quantile_cont(total_return, 0.01) AS p01,
               quantile_cont(total_return, 0.50) AS median,
               quantile_cont(total_return, 0.95) AS p95,
               quantile_cont(total_return, 0.99) AS p99,
               max(total_return) AS maximum,
               count(*) FILTER (WHERE abs(total_return) > ?) AS absolute_warning_count,
               count(*) FILTER (WHERE abs(total_return) > ?) AS absolute_extreme_count
        FROM ({relation}) WHERE {predicate}
    """, [warning_threshold, extreme_threshold, *params])


def industry_summary(
    connection, state: str, industry: str, standard: str, level: str
) -> dict[str, Any]:
    column = f"{level.lower()}_code"
    if column not in {"l1_code", "l2_code"}:
        raise ValueError(f"G0_UNSUPPORTED_INDUSTRY_LEVEL level={level}")
    return one(connection, f"""
        WITH groups AS (
          SELECT state.trade_date, membership.{column} AS industry_code, count(*) AS members
          FROM ({state}) state
          JOIN ({industry}) membership
            ON membership.asset_id=state.asset_id
           AND membership.classification_standard=?
           AND membership.in_date<=state.trade_date
           AND (membership.out_date IS NULL OR membership.out_date>state.trade_date)
          WHERE state.in_a_share_scope AND state.board_id!='BSE'
            AND membership.{column} IS NOT NULL
          GROUP BY 1,2
        ), daily AS (
          SELECT trade_date, count(*) AS industry_count, min(members) AS minimum_members
          FROM groups GROUP BY 1
        )
        SELECT min(industry_count) AS minimum_industry_count,
               max(industry_count) AS maximum_industry_count,
               min(minimum_members) AS all_history_minimum_members,
               count(*) FILTER (WHERE minimum_members<5) AS dates_below_five,
               count(*) AS trading_dates
        FROM daily
    """, [standard])
