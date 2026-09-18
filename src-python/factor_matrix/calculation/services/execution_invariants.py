"""Reusable fail-closed audit of a normal-equity path with no corporate actions."""
from decimal import Decimal
from .accounting import instant, money
from .equity_fixture import audit_engine


def audit_execution_path(engine):
    audit_engine(engine)
    def require(condition, reason):
        if not condition:
            raise ValueError('INVARIANT_'+reason)
    require(not engine.tables['corporate_actions'], 'USE_CORPORATE_ACTION_AUDITOR')
    for r in engine.tables['daily_nav']:
        require(r['cash'] >= 0, 'NONNEGATIVE_CASH')
        require(r['nav'] == r['cash']+r['market_value']+r['receivables'], 'NAV_IDENTITY')
    accepted = [o for o in engine.orders.values() if o.status != 'rejected']
    for o in accepted:
        if o.side == 'buy':
            require(o.quantity>=200 if engine.instruments[o.asset_id]=='STAR' else o.quantity%100==0, 'BUY_ORDER_UNIT')
    available_lots = []
    for f in engine.tables['fills']:
        e = engine.seen[f['event_id']]
        o = engine.orders[f['order_id']]
        require(not e['halted'], 'NO_HALTED_FILL')
        require(instant(f['time']) >= instant(o.earliest_execution)>instant(o.decision_time), 'DECISION_BEFORE_FILL')
        limit = Decimal('.10' if engine.instruments[f['asset_id']]=='MAIN' else '.20')
        reference = Decimal(e['reference_close'])
        lower,upper = money(reference*(1-limit)),money(reference*(1+limit))
        require(lower<=f['price']<=upper, 'DAILY_LIMIT_RANGE')
        require(not(f['side']=='buy' and f['price']==upper or f['side']=='sell' and f['price']==lower), 'NO_ADVERSE_LIMIT_FILL')
        if f['side']=='buy':
            require(instant(f['sellable_at']).date()>instant(f['time']).date(), 'T1_LOCK')
            available_lots.append([f['asset_id'],f['quantity'],f['sellable_at']])
        else:
            qty=f['quantity']
            for lot in available_lots:
                if lot[0]==f['asset_id'] and instant(lot[2])<=instant(f['time']):
                    taken=min(qty,lot[1]);qty-=taken;lot[1]-=taken
            require(qty==0, 'T1_SELLABILITY')
    return {'status':'passed','sessions':len(engine.tables['daily_nav'])-1,
            'checks':['NAV_identity','nonnegative_cash','buy_order_units','T1','adverse_limit','halt','ledger_reconciliation']}
