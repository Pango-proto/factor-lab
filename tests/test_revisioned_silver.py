from datetime import UTC, date, datetime, timedelta
import json

import polars as pl

from factor_matrix.revisioned_silver import (
    RevisionedSilverStore, SilverAsOfReader, SilverVersionLedger,
)
from factor_matrix.storage import DataLake, open_duckdb


def test_base_preserves_financial_first_observation_separate_from_freeze(tmp_path):
    lake=DataLake(tmp_path)
    base_root=lake.metadata/'base_manifests';base_root.mkdir(parents=True)
    p=lake.silver/'financial_pit'/'data.parquet';p.parent.mkdir(parents=True)
    pl.DataFrame({'revision_id':['fin_a'],'asset_id':['600000.SH'],
                  'first_seen_at':[datetime(2026,8,6,12,tzinfo=UTC)],
                  'available_at':[date(2026,8,7)]}).write_parquet(p)
    (base_root/'legacy_base_v1.json').write_text(json.dumps({'snapshot_id':'base','frozen_at':'2026-08-15T00:00:00+00:00',
        'artifacts':{'financial_pit':{'path':'silver/financial_pit/data.parquet'}}}))
    sql=SilverAsOfReader(lake).relation_sql('financial_pit',datetime(2026,8,16,tzinfo=UTC))
    with open_duckdb() as c:
        result=c.execute(f"SELECT cast(first_seen_at AS DATE),cast(revision_at AS DATE),available_at FROM ({sql})").fetchone()
    assert result==(date(2026,8,6),date(2026,8,15),date(2026,8,7))


def test_revisioned_partition_is_append_only_and_rejects_qfq(tmp_path):
    lake = DataLake(tmp_path)
    store = RevisionedSilverStore(lake)
    frame = pl.DataFrame({
        "trade_date": [date(2026, 8, 14)], "asset_id": ["000001.SZ"],
        "raw_close": [10.0], "adj_factor": [1.2],
    })
    seen = datetime(2026, 8, 14, 10, tzinfo=UTC)
    record = store.append("prices_daily", frame, effective_date=date(2026, 8, 14), first_seen_at=seen)
    stored = pl.read_parquet(tmp_path / record["path"])
    assert stored["revision_at"][0] == seen
    try:
        store.append(
            "prices_daily", frame.with_columns(pl.lit(10.0).alias("qfq_close")),
            effective_date=date(2026, 8, 15), first_seen_at=seen+timedelta(days=1),
        )
    except ValueError as exc:
        assert "DEPRECATED_PRICE_COLUMNS" in str(exc)
    else:
        raise AssertionError("qfq_close must be rejected")


def test_silver_version_chain_uses_parent_and_first_seen_semantics(tmp_path):
    lake = DataLake(tmp_path)
    ledger = SilverVersionLedger(lake)
    first = ledger.publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 14, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    second = ledger.publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 15, tzinfo=UTC),
        changes=[{"table": "prices_daily", "path": "x", "sha256": "y"}],
        quality_gate={"status": "passed"},
    )
    assert second["parent_version_id"] == first["version_id"]
    assert second["revision_at_semantics"] == "first_seen_at"
    assert ledger.current()["version_id"] == second["version_id"]


def test_as_of_reader_supports_delta_only_tables(tmp_path):
    lake = DataLake(tmp_path)
    base_root = lake.metadata / "base_manifests"
    base_root.mkdir(parents=True)
    (base_root / "legacy_base_v1.json").write_text(json.dumps({
        "snapshot_id": "base", "frozen_at": "2026-08-14T00:00:00+00:00",
        "artifacts": {},
    }))
    frame = pl.DataFrame({
        "snapshot_date": [date(2026, 8, 14)], "concept_id": ["885001.TI"],
        "asset_id": ["000001.SZ"],
    })
    seen = datetime(2026, 8, 15, tzinfo=UTC)
    RevisionedSilverStore(lake).append(
        "concept_membership_snapshot", frame,
        effective_date=date(2026, 8, 14), first_seen_at=seen,
    )
    sql = SilverAsOfReader(lake).relation_sql(
        "concept_membership_snapshot", seen + timedelta(seconds=1)
    )
    with open_duckdb() as connection:
        assert connection.execute(
            f"SELECT concept_id, asset_id FROM ({sql})"
        ).fetchone() == ("885001.TI", "000001.SZ")


def test_unpublished_delta_is_invisible_when_version_ledger_exists(tmp_path):
    lake = DataLake(tmp_path)
    base_root = lake.metadata / "base_manifests"
    base_root.mkdir(parents=True)
    base_path = lake.silver / "prices_daily" / "data.parquet"
    base_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": [date(2026, 8, 14)], "asset_id": ["000001.SZ"],
        "raw_close": [10.0], "adj_factor": [1.0],
    }).write_parquet(base_path)
    (base_root / "legacy_base_v1.json").write_text(json.dumps({
        "snapshot_id": "base", "frozen_at": "2026-08-14T00:00:00+00:00",
        "artifacts": {"prices_daily": {"path": "silver/prices_daily/data.parquet"}},
    }))
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 14, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    RevisionedSilverStore(lake).append(
        "prices_daily",
        pl.DataFrame({
            "trade_date": [date(2026, 8, 15)], "asset_id": ["000001.SZ"],
            "raw_close": [11.0], "adj_factor": [1.0],
        }),
        effective_date=date(2026, 8, 15),
        first_seen_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    sql = SilverAsOfReader(lake).relation_sql(
        "prices_daily", datetime(2026, 8, 16, tzinfo=UTC)
    )
    with open_duckdb() as connection:
        assert connection.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0] == 1
