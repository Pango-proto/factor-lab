"""Same-market strategy comparison; synthetic stress cases, never a path distribution."""
import math
import random
from decimal import Decimal as D
from pathlib import Path
import numpy as np
from ...storage import DataLake, json_hash, source_tree_hash
from .accounting import money, ZERO
from .backtest_path import path_functionals_v3
from .equity_execution import EquityAccountingEngine, POLICY, clock, next_weekdays
from .equity_fixture import build_equity_fixture, execute, mark, submit
from .execution_invariants import audit_execution_path
from .preview_statistics import describe_returns, price_diagnostics
from .strategy_fixture import plain

CONTRACT = {'id':'backtest_chart_preview_v3','seed':1729,'sessions':252,'initial_cash':'100000',
    'purpose':'synthetic_strategy_comparison_only','comparison':'three_strategies_one_common_market',
    'flow_timing':'period_end','dividend_tax':'not_modelled','execution_policy':POLICY,
    'scenarios':['steady','stress','recovery'],'rebalance_sessions':21,
    'benchmark':'buy_max_affordable_at_first_open_then_hold_cash_remainder_no_terminal_liquidation',
    'stress_execution_events':{'64':'open_gap_plus_3_percent','127':'halt_carry_close'},
    'statistics':'descriptive_no_research_gates;rf0;252_annualization;sample_std_ddof1;gross_one_way_turnover',
    'calendar':'ordinal_synthetic_weekdays_not_exchange_calendar'}
STRATEGIES=[('a','固定 80%','#346779'),('b','趋势 80/20%','#bd8396'),('c','固定 40%','#8a7199'),('benchmark','买入持有基准','#89908a')]


def build_market(scenario, *, seed=None):
    rng=random.Random(CONTRACT['seed'] if seed is None else seed)
    bars=[{'day':0,'close':D('20'),'open':D('20'),'reference_close':D('20'),'halted':False}]
    for day in range(1,253):
        noise=[rng.gauss(0,1) for _ in range(4)]
        prior=bars[-1]['close']
        volatility=.019 if 87<=day<=138 else .007
        drift=.00065
        if scenario=='stress' and 95<=day<=115:drift-=.010
        if scenario=='recovery':
            if 60<=day<=85:drift-=.009
            if 135<=day<=180:drift+=.004
        lower,upper=money(prior*D('.91')),money(prior*D('1.09'))
        close=min(upper,max(lower,money(str(float(prior)*math.exp(drift+volatility*(.72*noise[0]+.69*noise[1]))))))
        opening=min(upper,max(lower,money(str(float(prior)*math.exp(.002*noise[1])))))
        halted=scenario=='stress' and day==127
        if scenario=='stress' and day==64:opening=money(prior*D('1.03'))
        if halted:opening=close=prior
        bars.append({'day':day,'close':close,'open':opening,'reference_close':prior,'halted':halted})
    return bars


def target_exposure(sid, history):
    if sid=='a':return D('.8')
    if sid=='c':return D('.4')
    if sid=='benchmark':return D('1')
    if len(history)<20:return D('.2')
    return D('.8') if history[-1]>sum(history[-20:])/20 else D('.2')


def run_strategy(sid,name,color,bars,days, *, include_histogram=True):
    asset='FIXTURE.SH'
    engine=EquityAccountingEngine('100000',instruments={asset:'MAIN'},sessions=days,flow_timing=CONTRACT['flow_timing'])
    mark(engine,days[0],{asset:'20'})
    target=D(0);targets=[target];pending=[];unfilled=[]
    for day,bar in enumerate(bars[1:],1):
        rebalance=day==1 if sid=='benchmark' else (day-1)%21==0
        oid=str(day)
        if rebalance:
            target=target_exposure(sid,[b['close'] for b in bars[:day]])
            requested=int(engine.tables['daily_nav'][-1]['nav']*target/bar['reference_close']/100)*100
            change=requested-engine.total(asset)
            if change:
                side='buy' if change>0 else 'sell'
                limit=money(bar['reference_close']*(D('1.02') if side=='buy' else D('.98')))
                qty=engine.affordable_quantity(asset,engine.cash-engine.reserved_cash,limit,change) if change>0 else -change
                if qty:
                    status=submit(engine,days[day-1],oid,asset,side,qty,limit)
                    pending.append({'day':day,'decision_day':day-1,'side':side,'quantity':qty,'limit_price':float(limit),'status':status})
        if oid in engine.orders:
            result=execute(engine,days[day],oid,bar['open'],bar['reference_close'],halted=bar['halted'])
            if result not in ('filled','partial_fill'):
                unfilled.append({'day':day,'side':engine.orders[oid].side,'quantity':engine.orders[oid].quantity,
                    'price':float(bar['open']),'limit_price':float(engine.orders[oid].limit_price),'reason':result})
        mark(engine,days[day],{asset:bar['close']},opens={asset:bar['open']},halted=(asset,) if bar['halted'] else ())
        targets.append(target)
    checks=audit_execution_path(engine)
    navs=engine.tables['daily_nav']
    metrics=path_functionals_v3(navs[1:],initial_nav='100000',initial_time=clock(days[0]),flow_timing=CONTRACT['flow_timing'])
    rows=[]
    for i,account in enumerate(navs):
        m=metrics['curve'][i-1] if i else None
        shares=sum(p['quantity'] for p in engine.tables['positions'] if p['date']==days[i])
        rows.append({'day':i,'nav':float(account['nav']),'wealth':float(m['unit_wealth']) if m else 1,
            'return':float(m['period_return']) if m else 0,'drawdown':-float(m['drawdown']) if m else 0,
            'price':float(bars[i]['close']),'shares':shares,'cash':float(account['cash']),
            'market_value':float(account['market_value']),'receivables':float(account['receivables']),
            'cash_stack_top':float(account['market_value']+account['cash']),'nav_stack_top':float(account['nav']),
            'exposure':float(account['market_value']/account['nav']),'target_exposure':float(targets[i])})
    trades=[]
    for f in engine.tables['fills']:
        fees={r['component']:float(r['amount']) for r in engine.tables['fees'] if r['fill_id']==f['fill_id']}
        trades.append({'day':days.index(f['time'][:10]),'side':f['side'],'quantity':f['quantity'],'price':float(f['price']),
            'decision_time':engine.orders[f['order_id']].decision_time,'fill_time':f['time'],**fees})
    stats=describe_returns([r['return'] for r in rows[1:]], include_histogram=include_histogram)
    mdd=float(metrics['max_drawdown']);total=float(metrics['total_return']);cagr=(1+total)**(252/len(rows[1:]))-1
    summary={'total_return':total,'final_nav':rows[-1]['nav'],'max_drawdown':mdd,'fills':len(trades),
        'unfilled':len(unfilled),'annualized_volatility':stats['annualized_volatility'],'sharpe':stats['sharpe'],
        'calmar':cagr/mdd if mdd else None,'total_fees':float(sum((f['amount'] for f in engine.tables['fees']),ZERO)),
        'turnover':sum(t['quantity']*t['price']/rows[t['day']-1]['nav'] for t in trades),
        'average_exposure':sum(r['exposure'] for r in rows[1:])/252,
        'underwater_sessions':metrics['longest_underwater_observations'],'terminal_underwater_sessions':metrics['terminal_underwater_observations'],
        'right_censored':metrics['right_censored']}
    return {'id':sid,'name':name,'color':color,'rows':rows,'trades':trades,'unfilled':unfilled,'orders':pending,'statistics':stats,'summary':summary,'invariants':checks}


def acceptance():
    fixture=build_equity_fixture()
    # Preserve the published 10-day hand ledger; test a funded limit-buy branch.
    days=next_weekdays('2026-09-08T00:00:00+00:00',3)
    engine=EquityAccountingEngine('2000',instruments={'FIXTURE.SH':'MAIN'},sessions=days)
    mark(engine,days[0],{'FIXTURE.SH':'9'})
    submit(engine,days[0],'upper-limit','FIXTURE.SH','buy',100,'9.90')
    result=execute(engine,days[1],'upper-limit','9.90','9')
    if result!='price_limit_blocked' or engine.cash!=D('2000') or engine.tables['fills']:
        raise AssertionError('UPPER_LIMIT_BUY_BRANCH')
    values={}
    for timing in ('period_end','period_start'):
        values[timing]=path_functionals_v3([{'time':clock(days[1]),'nav':'12576.79','external_flow':'2000'}],
            initial_nav='9676.79',initial_time=clock(days[0]),flow_timing=timing)['total_return']
        values[timing]=D('.967679')*(1+values[timing])-1
    if values['period_end']!=D('.057679') or abs(values['period_start']-D('.042263804556732'))>D('1e-12'):
        raise AssertionError('FLOW_CONVENTION_HAND_CHECK')
    fixture['flow_timing']='period_end'
    fixture['dividend_tax']='not_modelled'
    fixture['flow_comparison']=plain(values)
    fixture['checks'] += [{'name':'upper_limit_buy_funded_branch','actual':result,'expected':'price_limit_blocked','status':'passed'},
        *[{'name':'flow_'+k,'actual':str(v),'status':'passed'} for k,v in values.items()]]
    return fixture


def build_preview_data():
    days=next_weekdays('2026-01-01T00:00:00+00:00',254)
    scenarios=[]
    for scenario,name in [('steady','常态行情'),('stress','下跌与执行压力'),('recovery','下跌后恢复')]:
        bars=build_market(scenario)
        paths=[run_strategy(*strategy,bars,days) for strategy in STRATEGIES]
        returns=[[r['return'] for r in p['rows'][1:]] for p in paths]
        scenarios.append({'id':scenario,'name':name,'series':paths,'correlation':np.corrcoef(returns).tolist(),
            'market':plain(bars),'price_diagnostics':price_diagnostics([float(b['close']) for b in bars]),
            'high_volatility':{'from':87,'to':138,'label':'预设高波动 D87–D138'},
            'invariants':{'status':'passed','strategy_sessions':252*3,'benchmark_sessions':252}})
    return {'schema_version':3,'contract':CONTRACT,'scenarios':scenarios,'acceptance_fixture':acceptance(),
            'real_market_data':False,'research_status':'not_eligible','risk_basis_id':None,'risk_set_version':None,'parent_run_ids':[]}


def publish_preview(output_dir:Path):
    lake=DataLake(output_dir)
    payload=build_preview_data()
    path=lake.write_immutable_json(output_dir/'preview.json',payload)
    return lake.write_immutable_json(output_dir/'_MANIFEST.json',{'schema_version':3,
        'run_id':CONTRACT['id']+'_'+json_hash(CONTRACT)[:16],'status':'visual_demo_only','definition':CONTRACT,
        'code_hash':source_tree_hash(),'real_market_data':False,'research_promoted':False,
        'risk_basis_id':None,'risk_set_version':None,'parent_run_ids':[],
        'outputs':{'preview':lake.artifact_record(path)}})
