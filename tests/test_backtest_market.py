import json
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from factor_matrix.calculation.services.accounting import AccountingEngine
from factor_matrix.calculation.services.backtest_market import normalize_backtest_bars,checked_data_gate
from factor_matrix.storage import DataLake,file_sha256


def frames():
    dates=[date(2001,1,d) for d in range(1,5)]
    prices=pl.DataFrame(dict(trade_date=dates,asset_id=['600000.SH']*4,volume=[1.,2.,99.,4.],amount=[10.,20.,990.,40.]))
    state=pl.DataFrame(dict(trade_date=dates,asset_id=['600000.SH']*4,is_suspended=[False]*4,source_day_complete=[True]*4))
    return prices,state


def test_real_adapter_explicit_units_and_lagged_adv_excludes_same_day_volume():
    prices,state=frames()
    result=normalize_backtest_bars(prices,state,adv_window=2)
    assert result['volume_shares'].to_list()==[100,200,9900,400]
    assert result['amount_cny'].to_list()==[10000,20000,990000,40000]
    assert result['adv_shares'].to_list()==[None,None,150,5050]
    assert result['adv_last_observation_date'][2]==date(2001,1,2)


def test_only_confirmed_suspension_may_have_zero_imputed_liquidity():
    prices,state=frames()
    prices=prices.filter(pl.col('trade_date')!=date(2001,1,2))
    missing=normalize_backtest_bars(prices,state,adv_window=2)
    assert missing['adv_shares'][2] is None
    state=state.with_columns((pl.col('trade_date')==date(2001,1,2)).alias('is_suspended'))
    halted=normalize_backtest_bars(prices,state,adv_window=2)
    assert halted['adv_shares'][2]==50


def test_duplicate_prices_are_rejected_before_join():
    prices,state=frames()
    with pytest.raises(ValueError,match='DUPLICATE_KEYS'):
        normalize_backtest_bars(pl.concat([prices,prices.head(1)]),state)


def test_closed_real_data_gate_is_not_opened_by_available_files(tmp_path):
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps(dict(job='complete_data_refresh',status='failed',market_input_allowed=False)))
    with pytest.raises(ValueError,match='REAL_DATA_GATE_CLOSED'):
        checked_data_gate(DataLake(tmp_path),gate,expected_sha256=file_sha256(gate))


def test_successful_gate_still_rejects_stale_data_version(tmp_path,monkeypatch):
    import factor_matrix.calculation.services.backtest_market as module
    class Ledger:
        def __init__(self,lake):pass
        def current(self):return {'version_id':'new'}
    monkeypatch.setattr(module,'SilverVersionLedger',Ledger)
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps(dict(job='complete_data_refresh',status='passed',market_input_allowed=True,silver_version_id='old')))
    with pytest.raises(ValueError,match='STALE_VERSION'):
        checked_data_gate(DataLake(tmp_path),gate,expected_sha256=file_sha256(gate))


def test_real_equity_requires_explicit_scope_and_rule_policy():
    event=dict(kind='submit',event_id='s',order_id='o',asset_id='600000.SH',side='buy',quantity=100,
               limit_price='10.00',time='2001-01-01T10:00:00+00:00',decision_time='2001-01-01T10:00:00+00:00',
               earliest_execution='2001-01-02T10:00:00+00:00')
    with pytest.raises(ValueError):AccountingEngine('10000',lot_size=100).apply(event)
    with pytest.raises(ValueError, match='REAL_EQUITY_REQUIRES_RULE_AWARE_ENGINE'):
        AccountingEngine('10000',lot_size=100,asset_scope=('600000.SH',))
    from factor_matrix.calculation.services.equity_execution import EquityAccountingEngine, clock
    engine=EquityAccountingEngine('10000', instruments={'600000.SH':'MAIN'}, sessions=['2026-09-01','2026-09-02','2026-09-03'])
    event.update(time=clock('2026-09-01'),decision_time=clock('2026-09-01'),signal_available_at=clock('2026-09-01'),earliest_execution=clock('2026-09-02','open'))
    assert engine.apply(event)=='accepted'
    assert engine.reserved_cash==Decimal('1005.01')
    with pytest.raises(ValueError,match='UNSUPPORTED_EQUITY_SCOPE'):
        AccountingEngine('10000',lot_size=100,asset_scope=('920001.BJ',))
