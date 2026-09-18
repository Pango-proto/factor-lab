import json
from decimal import Decimal
from pathlib import Path

from factor_matrix.storage import DataLake,file_sha256
from factor_matrix.calculation.services.strategy_fixture import run_strategy_fixture
from factor_matrix.calculation.services.backtest_review import review_backtest


def test_fixed_replay_audited_path_and_unavailable_ope_are_separate(tmp_path):
    lake=DataLake(tmp_path/'data')
    parent=run_strategy_fixture(lake,config_path=Path('config/strategy_fixture_v1.json'))
    config=json.loads(Path('config/backtest_analytics_v1.json').read_text())
    config['source']={'path':str(parent.relative_to(lake.root)),'sha256':file_sha256(parent)}
    cp=tmp_path/'analytics.json';cp.write_text(json.dumps(config))
    manifest=review_backtest(lake,config_path=cp)
    result=json.loads(manifest.read_text())
    assert result['status']=='engineering_passed'
    assert result['research_status']=='not_eligible'
    assert result['strict_ope_status']=='not_estimable'
    tables={k:json.loads((lake.root/r['path']).read_text()) for k,r in result['outputs'].items()}
    assert Decimal(tables['path_functionals']['max_drawdown'])==Decimal('0.062')
    assert len(tables['counterfactuals'])==2
    assert tables['counterfactuals'][0]['reconciliation']['status']=='passed'
    assert tables['counterfactuals'][1]['reconciliation']['status']=='passed'
    assert tables['strict_ope']['estimate'] is None
    assert all(r['one_session_capacity_shares'] is None for r in tables['capacity']['rows'])
    checksum=file_sha256(manifest)
    assert review_backtest(lake,config_path=cp)==manifest
    assert file_sha256(manifest)==checksum
