from datetime import UTC, date, datetime
import json

import polars as pl

from factor_matrix.factor_data import (
    FactorDataPipeline,
    SHIBOR_FIELDS,
    normalize_financial_pit,
    normalize_shibor_daily,
    normalize_statement_pit,
)
from factor_matrix.source import TushareResponse
from factor_matrix.storage import DataLake
from factor_matrix.revisioned_silver import SilverVersionLedger


def test_financial_sync_normalizes_mixed_existing_and_new_observation_timezones():
    calendar=pl.DataFrame({'cal_date':[date(2026,8,7),date(2026,9,18)],'is_open':[1,1]})
    raw={'ts_code':'600000.SH','end_date':'20260630','ann_date':'20260806','f_ann_date':'20260806',
         'report_type':'1','comp_type':'1','update_flag':'0','total_hldr_eqy_exc_min_int':100.}
    old=normalize_financial_pit(pl.DataFrame([raw]),pl.DataFrame(),calendar,datetime(2026,8,6,12,tzinfo=UTC))
    old=old.with_columns(pl.col('first_seen_at').dt.convert_time_zone('Australia/Sydney'))
    current=normalize_financial_pit(pl.DataFrame([raw,{**raw,'ts_code':'000001.SZ'}]),pl.DataFrame(),calendar,
                                   datetime(2026,9,17,3,tzinfo=UTC),existing=old)
    assert current['first_seen_at'].dtype==pl.Datetime('us','UTC')
    assert current.filter(pl.col('asset_id')=='600000.SH')['first_seen_at'][0]==datetime(2026,8,6,12,tzinfo=UTC)


def test_full_financial_sync_resolves_canonical_mapping_after_module_split(tmp_path, monkeypatch):
    import factor_matrix.factor_data as module
    lake=DataLake(tmp_path/'data')
    mapping_path=lake.silver/'bse_code_mapping'/'data.parquet'
    mapping_path.parent.mkdir(parents=True)
    pl.DataFrame({'old_asset_id':['830001.BJ'],'asset_id':['920001.BJ']}).write_parquet(mapping_path)
    calendar=pl.DataFrame({'exchange':['SSE'],'cal_date':[date(2026,8,3)],'is_open':[1],'pretrade_date':[date(2026,7,31)]})
    monkeypatch.setattr(module,'read_as_of_frame',lambda lake,table,*args: calendar if table=='trade_calendar' else pl.DataFrame({'asset_id':['920001.BJ']}))
    monkeypatch.setattr(module,'publish_revisioned_batch',lambda *a,**kw: ([],{'version_id':'test'}))
    monkeypatch.setattr(module,'normalize_shibor_daily',lambda *a: pl.DataFrame({'trade_date':[date(2026,8,3)],'rate_date':[date(2026,7,31)]}))
    monkeypatch.setattr(lake,'refresh_catalog',lambda:None)
    pipeline=FactorDataPipeline(None,lake,'test')
    def fetch(api,params,fields):
        row={f:None for f in fields}
        if api=='trade_cal':row.update(exchange='SSE',cal_date='20260803',is_open=1,pretrade_date='20260731')
        return TushareResponse(api,params,fields,[[row[f] for f in fields]],{})
    monkeypatch.setattr(pipeline,'_fetch',fetch)
    mappings=[]
    def store_balance(frames,disclosures,calendar,started,code_mapping,**kw):
        mappings.append(code_mapping);return 1
    monkeypatch.setattr(pipeline,'_normalize_and_store_batch',store_balance)
    monkeypatch.setattr(pipeline,'_normalize_and_store_statement',lambda *a,**kw:1)
    path=pipeline.sync([2026],date(2026,8,1),date(2026,8,3))
    assert mappings==[{'830001.BJ':'920001.BJ'}]
    assert json.loads(path.read_text())['row_counts']['financial_pit_fetched']==1


def test_financial_pit_delays_unverifiable_revision_until_first_seen() -> None:
    ingested_at = datetime(2026, 8, 6, 9, tzinfo=UTC)
    base = {
        "ts_code": "600000.SH",
        "ann_date": "20250419",
        "f_ann_date": "20250419",
        "end_date": "20241231",
        "report_type": "1",
        "comp_type": "1",
        "end_type": "4",
        "total_hldr_eqy_exc_min_int": 100.0,
        "total_hldr_eqy_inc_min_int": 110.0,
        "oth_eqt_tools_p_shr": 5.0,
        "defer_tax_liab": 2.0,
        "total_assets": 500.0,
    }
    balances = pl.DataFrame(
        [
            {**base, "update_flag": "0"},
            {**base, "update_flag": "1"},
            {
                **base,
                "ts_code": "000001.SZ",
                "total_hldr_eqy_exc_min_int": 200.0,
                "update_flag": "1",
            },
        ]
    )
    disclosures = pl.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "end_date": ["20241231", "20241231"],
            "actual_date": ["20250420", "20250420"],
        }
    )
    calendar = pl.DataFrame(
        {
            "cal_date": [date(2025, 4, 21), date(2026, 8, 6), date(2026, 8, 7)],
            "is_open": [1, 1, 1],
        }
    )

    result = normalize_financial_pit(
        balances, disclosures, calendar, ingested_at
    ).sort("asset_id")

    assert result.height == 2
    sz = result.filter(pl.col("asset_id") == "000001.SZ").row(0, named=True)
    sh = result.filter(pl.col("asset_id") == "600000.SH").row(0, named=True)
    assert sz["available_at"] == date(2026, 8, 7)
    assert sz["revision_policy"] == "conservative_revision_first_seen"
    assert sh["available_at"] == date(2025, 4, 21)
    assert sh["revision_policy"] == "confirmed_original_announcement"
    assert sh["book_equity"] == 97.0


def test_shibor_uses_strictly_prior_fixing() -> None:
    shibor = pl.DataFrame(
        {
            "date": ["20260804", "20260805"],
            "3m": [1.4, 1.5],
        }
    )
    calendar = pl.DataFrame(
        {
            "cal_date": [date(2026, 8, 5), date(2026, 8, 6)],
            "is_open": [1, 1],
        }
    )

    result = normalize_shibor_daily(
        shibor, calendar, datetime(2026, 8, 6, tzinfo=UTC)
    ).sort("trade_date")

    assert result.get_column("rate_date").to_list() == [
        date(2026, 8, 4),
        date(2026, 8, 5),
    ]
    assert result.filter(pl.col("rate_date") >= pl.col("trade_date")).height == 0


def test_income_revision_uses_same_conservative_pit_clock() -> None:
    ingested_at = datetime(2026, 8, 6, 9, tzinfo=UTC)
    calendar = pl.DataFrame(
        {"cal_date": [date(2025, 4, 21), date(2026, 8, 7)], "is_open": [1, 1]}
    )
    income = pl.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ"],
            "ann_date": ["20250420", "20250420"],
            "f_ann_date": ["20250420", "20250420"],
            "end_date": ["20241231", "20241231"],
            "report_type": ["1", "1"],
            "comp_type": ["1", "1"],
            "n_income_attr_p": [100.0, 200.0],
            "revenue": [500.0, 600.0],
            "total_revenue": [500.0, 600.0],
            "update_flag": ["0", "1"],
        }
    )

    result = normalize_statement_pit(
        income,
        calendar,
        ingested_at,
        statement="income",
        value_fields=("n_income_attr_p", "revenue", "total_revenue"),
    )

    original = result.filter(pl.col("asset_id") == "600000.SH").row(0, named=True)
    revision = result.filter(pl.col("asset_id") == "000001.SZ").row(0, named=True)
    assert original["available_at"] == date(2025, 4, 21)
    assert revision["available_at"] == date(2026, 8, 7)
    assert revision["revision_policy"] == "conservative_revision_first_seen"


def test_complete_adjustment_history_distinguishes_unchanged_and_adjusted_latest() -> None:
    ingested_at = datetime(2026, 8, 6, 9, tzinfo=UTC)
    calendar = pl.DataFrame(
        {
            "cal_date": [date(2025, 4, 21), date(2026, 8, 7)],
            "is_open": [1, 1],
        }
    )
    rows = pl.DataFrame(
        {
            "ts_code": ["UNCHANGED.SH", "ADJUSTED.SH", "ADJUSTED.SH"],
            "ann_date": ["20250420", "20250420", "20250420"],
            "f_ann_date": ["20250420", "20250420", "20250420"],
            "end_date": ["20241231", "20241231", "20241231"],
            "report_type": ["1", "1", "5"],
            "comp_type": ["1", "1", "1"],
            "n_income_attr_p": [10.0, 22.0, 20.0],
            "revenue": [100.0, 110.0, 100.0],
            "total_revenue": [100.0, 110.0, 100.0],
            "update_flag": ["1", "1", "0"],
        }
    )

    result = normalize_statement_pit(
        rows,
        calendar,
        ingested_at,
        statement="income",
        value_fields=("n_income_attr_p", "revenue", "total_revenue"),
        adjustment_history_complete=True,
    )

    unchanged = result.filter(pl.col("asset_id") == "UNCHANGED.SH").row(0, named=True)
    adjusted_latest = result.filter(
        (pl.col("asset_id") == "ADJUSTED.SH") & (pl.col("report_type") == "1")
    ).row(0, named=True)
    adjusted_original = result.filter(
        (pl.col("asset_id") == "ADJUSTED.SH") & (pl.col("report_type") == "5")
    ).row(0, named=True)
    assert unchanged["available_at"] == date(2025, 4, 21)
    assert unchanged["revision_policy"] == "confirmed_unchanged_latest_no_adjustment_record"
    assert adjusted_latest["available_at"] == date(2026, 8, 7)
    assert adjusted_original["available_at"] == date(2025, 4, 21)


def test_daily_risk_free_sync_requires_prior_fixing_for_target(tmp_path) -> None:
    class Client:
        def query(self, api_name, params, fields):
            assert api_name == "shibor"
            item = ["20260806", 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
            raw = {"code": 0, "data": {"fields": SHIBOR_FIELDS, "items": [item]}}
            return TushareResponse(api_name, params, SHIBOR_FIELDS, [item], raw)

    lake = DataLake(tmp_path / "data")
    lake.replace(
        "trade_calendar",
        pl.DataFrame({
            "exchange": ["SSE"], "cal_date": [date(2026, 8, 7)], "is_open": [1]
        }),
    )
    SilverVersionLedger(lake).publish(
        base_id="legacy_base_v1", base_snapshot_sha256="base",
        observed_at=datetime(2026, 8, 7, tzinfo=UTC), changes=[],
        quality_gate={"status": "passed"},
    )
    manifest = FactorDataPipeline(Client(), lake, "fingerprint").sync_risk_free(
        date(2026, 8, 7), date(2026, 8, 7)
    )
    payload = json.loads(manifest.read_text())
    row = pl.read_parquet(lake.root / payload["changes"][0]["path"]).row(0, named=True)
    assert row["trade_date"] == date(2026, 8, 7)
    assert row["rate_date"] == date(2026, 8, 6)
