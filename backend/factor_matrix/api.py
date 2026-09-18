"""Small-object HTTP boundary for the local research frontend.

The browser may submit weights and constraints, but it never receives the
full security-by-factor matrix or any covariance matrix.
"""

from __future__ import annotations

import json
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import polars as pl

from .factor_engine import (
    FactorFamily, FactorRegistry, FactorRegistryStore, FactorRole, FactorStatus,
    FeatureRole,
)
from .risk_model import load_risk_factor_set
from .storage import DataLake, open_duckdb
from .risk_model.acceptance import registry_acceptance


def risk_acceptance_response(registry_path: Path) -> dict[str, Any]:
    """Read-only stage metadata; never translate old frozen into new approval."""
    if not registry_path.is_file():
        return {'schema_version':2,'status':'unavailable','rows':[],'production_allowed':False}
    rows=[]
    with sqlite3.connect(f'file:{registry_path.resolve()}?mode=ro',uri=True) as c:
        for version,status in c.execute('SELECT risk_set_version,status FROM risk_set ORDER BY risk_set_version').fetchall():
            try:
                state=registry_acceptance(c,version)
                rows.append({'risk_set_version':version,'registry_status':status,
                    'risk_basis_id':state.risk_basis_id,'basis_acceptance_status':state.basis_acceptance_status,
                    'covariance_acceptance_status':state.covariance_acceptance_status,
                    'pit_acceptance_status':state.pit_acceptance_status,
                    'basis_consumption_allowed':state.basis_consumption_allowed,
                    'risk_optimization_eligible':state.risk_optimization_eligible})
            except (ValueError,OSError,TypeError,KeyError):
                rows.append({'risk_set_version':version,'registry_status':status,
                    'evidence_status':'invalid','basis_consumption_allowed':False,'risk_optimization_eligible':False})
    return {'schema_version':2,'status':'ready','rows':rows,'production_allowed':False,
            'publication_status':'not_authorized_by_metadata_endpoint'}


def factor_catalog_response(
    *,
    query: str = "",
    feature_role: str = "bettable",
    role: str = "",
    family: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Search factor metadata without loading factor panels into the browser."""
    registry = FactorRegistry.discover()
    needle = query.strip().lower()[:80]
    rows = []
    # ``family`` is accepted only as a wire-level compatibility alias. It is
    # not returned and does not participate in persistence or identity.
    if family:
        feature_role = {"risk": "basis", "alpha": "bettable"}.get(family, family)
    selected_role = FactorRole(role) if role else None
    try:
        selected_feature_role = FeatureRole(feature_role)
    except ValueError:
        raise ValueError(f"FEATURE_ROLE_INVALID:{feature_role}")
    for plugin in registry.search_feature_role(selected_feature_role):
        if selected_role is not None and plugin.spec.role is not selected_role:
            continue
        spec = plugin.spec
        searchable = " ".join(
            (spec.factor_id, spec.display_name, spec.family.value, spec.formula_expr, spec.hypothesis)
        ).lower()
        if needle and needle not in searchable:
            continue
        rows.append({
            "factor_id": spec.factor_id,
            "version": spec.version,
            "label": spec.display_name,
            "feature_role": feature_role,
            "feature_key": spec.feature_key,
            "role": spec.role.value,
            "formula": spec.formula_expr,
            "description": spec.hypothesis,
            "source_tables": list(spec.source_tables),
            "source_fields": list(spec.source_fields),
            "pit_key": spec.pit_key,
            "family_root_id": spec.family_root_id,
            "winsorize_method": spec.winsorize_method,
            "standardize": spec.standardize,
            "weight_scheme": spec.weight_scheme,
            "neutralize_against": spec.neutralize_against,
            "orthogonalize_after": list(spec.orthogonalize_after),
            "coverage_min": spec.coverage_min,
            "research_status": spec.status.value,
            "deployment_status": "eligible" if spec.status is FactorStatus.VALIDATED else "blocked",
        })
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    start = (page - 1) * page_size
    return {
        "schema_version": 1,
        "status": "ready",
        "query": needle,
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "rows": rows[start : start + page_size],
        "boundary": {"factor_values_sent": False, "full_factor_matrix_sent": False},
    }


def risk_catalog_response(
    *, query: str = "", role: str = "", status: str = "",
    page: int = 1, page_size: int = 20,
) -> dict[str, Any]:
    """Risk-family view over the one physical factor definition registry."""
    registry = FactorRegistry.discover()
    needle = query.strip().lower()[:80]
    selected_role = FactorRole(role) if role else None
    selected_status = FactorStatus(status) if status else None
    rows = []
    for plugin in registry.search_feature_role(FeatureRole.BASIS, status=selected_status):
        if selected_role is not None and plugin.spec.role is not selected_role:
            continue
        spec = plugin.spec
        searchable = " ".join((
            spec.factor_id, spec.display_name, spec.family.value,
            spec.formula_expr, spec.hypothesis,
        )).lower()
        if needle and needle not in searchable:
            continue
        rows.append({
            "exposure_id": spec.factor_id,
            "version": spec.version,
            "label": spec.display_name,
            "feature_role": "basis",
            "role": spec.role.value,
            "feature_key": spec.feature_key,
            "formula": spec.formula_expr,
            "source_tables": list(spec.source_tables),
            "source_fields": list(spec.source_fields),
            "orthogonalize_after": list(spec.orthogonalize_after),
            "registry_status": spec.status.value,
            "production_status": "blocked_until_basis_covariance_pit_and_publication",
            "family_root_id": spec.family_root_id,
            "description": spec.hypothesis,
        })
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    start = (page - 1) * page_size
    return {
        "schema_version": 1,
        "status": "ready",
        "query": needle,
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "rows": rows[start : start + page_size],
        "boundary": {"security_exposures_sent": False, "covariance_sent": False},
    }


def pipeline_summary_response(lake: DataLake) -> dict[str, Any]:
    """Build a small, live catalog summary without exposing table contents."""
    tables = {
        "security_master": "证券主数据",
        "trade_calendar": "交易日历",
        "prices_daily": "行情与复权",
        "valuation_daily": "每日估值",
        "returns_daily": "投资收益率",
        "board_membership_history": "PIT 上市板块归属",
        "index_prices_daily": "板块官方参考指数",
        "financial_pit": "点时财务",
        "risk_free_daily": "无风险利率",
    }
    connection = open_duckdb(temp_directory=lake.root / "tmp" / "duckdb")
    rows: list[dict[str, Any]] = []
    try:
        for table, label in tables.items():
            path = lake.silver / table / "data.parquet"
            if not path.exists():
                rows.append({"id": table, "label": label, "status": "missing", "rows": 0})
                continue
            escaped = str(path).replace("'", "''")
            count = int(connection.execute(
                f"SELECT count(*) FROM read_parquet('{escaped}')"
            ).fetchone()[0])
            rows.append({"id": table, "label": label, "status": "complete", "rows": count})
    finally:
        connection.close()
    history_path = lake.metadata / "quality_reports" / "full_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    summary = history.get("summary", {})
    return {
        "schema_version": 1,
        "status": history.get("status", "not_ready"),
        "as_of_date": summary.get("max_trade_date"),
        "history": summary,
        "tables": rows,
        "boundary": {"base_data_sent": False, "table_rows_sent": False},
    }


def evaluation_framework_response(
    lake: DataLake, *, config_path: Path | None = None,
) -> dict[str, Any]:
    """Keep lifecycle validation in the Python service and return small objects."""
    from .calculation.services.evaluation_summary import evaluation_summary

    return evaluation_summary(lake, config_path)



def calculation_artifact_catalog_response(
    lake: DataLake, artifact_type: str = ""
) -> dict[str, Any]:
    """Expose calculation metadata, never full matrices or covariance tables."""
    rows: list[dict[str, Any]] = []
    for path in lake.root.glob(
        "gold/calculations/stage=*/operation=*/run_id=*/artifact=*/metadata.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if artifact_type and payload.get("artifact_type") != artifact_type:
            continue
        rows.append({
            key: payload.get(key)
            for key in (
                "artifact_type", "artifact_version", "run_id", "operation_id",
                "operation_version", "stage", "as_of_date", "scope_key",
                "quality_status", "created_at",
            )
        })
    for path in lake.root.glob("gold/*_v1/run_id=*/_MANIFEST.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        product = payload.get("product", "")
        normalized_type = product.removesuffix("_v1")
        if artifact_type and artifact_type not in {product, normalized_type}:
            continue
        identity = payload.get("identity", {})
        rows.append({
            "artifact_type": normalized_type,
            "artifact_version": "1",
            "run_id": payload.get("run_id"),
            "operation_id": "l2_stateless_pipeline",
            "operation_version": "1",
            "stage": "l2_return_decomposition",
            "as_of_date": (identity.get("date_range") or [None, None])[-1],
            "scope_key": identity.get("mode"),
            "quality_status": "passed",
            "created_at": payload.get("created_at"),
        })
    rows.sort(key=lambda row: (row.get("as_of_date") or "", row.get("created_at") or ""), reverse=True)
    return {
        "schema_version": 1,
        "status": "ready" if rows else "not_ready",
        "rows": rows,
        "boundary": {"table_values_sent": False, "covariance_sent": False},
    }


def stock_scores_response(
    lake: DataLake, *, board_id: str, limit: int
) -> dict[str, Any]:
    """Return one explicitly selected board from the latest score artifact."""
    if board_id not in {"MAIN", "CHINEXT", "STAR", "BSE"}:
        raise ValueError("BOARD_ID_REQUIRED")
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for path in lake.root.glob(
        "gold/calculations/stage=scoring/operation=*/run_id=*/artifact=stock_score/metadata.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("quality_status") != "passed":
            continue
        if payload.get("scope_key") not in {f"board:{board_id}", "market:board_scoped"}:
            continue
        candidates.append((payload.get("created_at", ""), path, payload))
    if not candidates:
        return {"schema_version": 1, "status": "not_ready", "board_id": board_id, "rows": []}
    _, metadata_path, metadata = max(candidates, key=lambda item: item[0])
    table_record = metadata.get("tables", {}).get("values")
    if not table_record:
        raise RuntimeError("STOCK_SCORE_VALUES_TABLE_MISSING")
    frame = pl.read_parquet(lake.root / table_record["uri"])
    if "board_id" not in frame.columns:
        raise RuntimeError("STOCK_SCORE_BOARD_COLUMN_MISSING")
    rank_column = "rank_signal" if "rank_signal" in frame.columns else "asset_id"
    rows = (
        frame.filter(pl.col("board_id") == board_id)
        .sort(rank_column)
        .head(min(max(limit, 1), 200))
        .to_dicts()
    )
    return {
        "schema_version": 1,
        "status": "ready",
        "board_id": board_id,
        "as_of_date": metadata.get("as_of_date"),
        "run_id": metadata.get("run_id"),
        "rows": rows,
        "boundary": {"implicit_cross_board_mix": False},
    }
class StrategySnapshotCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime_ns = -1
        self._payload: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        stat = self.path.stat()
        if self._payload is None or stat.st_mtime_ns != self._mtime_ns:
            self._payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._mtime_ns = stat.st_mtime_ns
        return self._payload


def make_handler(
    lake: DataLake,
    research_cache: StrategySnapshotCache,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FactorMatrixAPI/1"

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query_params = parse_qs(parsed.query)
            if path == '/api/risk-acceptance':
                try:
                    self._json(HTTPStatus.OK,risk_acceptance_response(lake.metadata/'factor_registry.sqlite'))
                except sqlite3.Error:
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE,{'status':'unavailable','production_allowed':False})
                return
            if path == "/api/factors":
                try:
                    response = factor_catalog_response(
                        query=query_params.get("q", [""])[0],
                        feature_role=query_params.get("feature_role", ["bettable"])[0] or "bettable",
                        family=query_params.get("family", [""])[0] or None,
                        role=query_params.get("role", [""])[0],
                        page=int(query_params.get("page", ["1"])[0]),
                        page_size=int(query_params.get("page_size", ["20"])[0]),
                    )
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.OK, response)
                return
            if path == "/api/risk-exposures":
                try:
                    response = risk_catalog_response(
                        query=query_params.get("q", [""])[0],
                        role=query_params.get("role", [""])[0],
                        status=query_params.get("status", [""])[0],
                        page=int(query_params.get("page", ["1"])[0]),
                        page_size=int(query_params.get("page_size", ["20"])[0]),
                    )
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.OK, response)
                return
            if path == "/api/pipeline/summary":
                self._json(HTTPStatus.OK, pipeline_summary_response(lake))
                return
            if path == "/api/evaluation/framework":
                self._json(HTTPStatus.OK, evaluation_framework_response(lake))
                return
            if path == "/api/calculation-artifacts":
                self._json(
                    HTTPStatus.OK,
                    calculation_artifact_catalog_response(
                        lake, query_params.get("type", [""])[0]
                    ),
                )
                return
            if path == "/api/scores":
                try:
                    response = stock_scores_response(
                        lake,
                        board_id=query_params.get("board_id", [""])[0],
                        limit=int(query_params.get("limit", ["100"])[0]),
                    )
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.OK, response)
                return
            if path == "/api/research/snapshot":
                try:
                    payload = research_cache.load()
                    payload["boundary"] = {
                        "base_data_sent": False,
                        "full_factor_matrix_sent": False,
                        "factor_covariance_sent": False,
                        "security_covariance_sent": False,
                    }
                    self._json(HTTPStatus.OK, payload)
                except FileNotFoundError:
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "not_ready"})
                return
            if path == "/api/health":
                try:
                    snapshot = research_cache.load()
                    self._json(HTTPStatus.OK, {"status": "ok", "as_of_date": snapshot["as_of_date"]})
                except Exception as exc:  # pragma: no cover - operating-system boundary
                    self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "error", "message": str(exc)})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"strategy-api {self.address_string()} {format % args}")

    return Handler


def serve_strategy_api(
    host: str = "127.0.0.1",
    port: int = 8765,
    data_root: Path = Path("data"),
    research_snapshot_path: Path = Path("data/metadata/research-summary.json"),
) -> None:
    research_cache = StrategySnapshotCache(research_snapshot_path.resolve())
    lake = DataLake(data_root)
    registry = FactorRegistry.discover()
    store = FactorRegistryStore(lake.metadata / "factor_registry.sqlite")
    store.sync_definitions(registry)
    risk_set_path = Path(__file__).resolve().parents[2] / "config" / "risk_factor_set_candidate_v1.json"
    if risk_set_path.exists():
        store.sync_risk_factor_set(load_risk_factor_set(risk_set_path, registry), registry)
    server = ThreadingHTTPServer((host, port), make_handler(lake, research_cache))
    print(f"Research API listening on http://{host}:{port}; snapshot={research_cache.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
