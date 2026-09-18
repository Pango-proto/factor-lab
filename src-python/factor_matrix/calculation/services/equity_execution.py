"""Normal A-share execution v2; explicit calendar/board, exact-money accounting.

A scoped engineering policy, not a broker certification. ST, IPO/no-limit days,
BSE, historical schedules before 2023-08-28, intraday/VWAP and dividend tax are
unsupported. Daily locked-limit blocking is a conservative fill assumption.
Legacy fixture v1 remains reproducible through AccountingEngine.
"""
from datetime import timedelta
from decimal import Decimal

from .accounting import AccountingEngine, ZERO, commission, decimal, instant, money

POLICY = {
    'id': 'normal_a_share_execution_v2', 'rules_checked_as_of': '2026-09-17',
    'supported_fee_dates_from': '2023-08-28',
    'scope': 'normal_MAIN_CHINEXT_STAR_only_no_ST_IPO_or_limit_exemptions',
    'commission_rate': '0.0001854', 'minimum_commission': '5.00',
    'stamp_duty_sell': '0.0005', 'transfer_fee_both': '0.00001',
    'fee_rounding': 'cumulative_per_order_component_half_up_cent',
    'broker_rounding_verified': False, 'commission_includes_exchange_fees_assumed': True,
    'execution': 'previous_session_close_decision_next_session_open_limit_order',
    'limit_fill': 'conservative_reject_adverse_side_at_limit_open',
    'price_basis': 'raw_execution_and_marks_explicit_corporate_actions_no_double_adjustment',
    'sources': [
        'https://one.sse.com.cn/onething/gptz/',
        'https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml',
        'https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf',
    ],
}


def next_weekdays(start: str, count: int) -> list[str]:
    """Synthetic calendar only. Never use this in place of an exchange calendar."""
    day = instant(start).date()
    days = []
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days


def clock(day: str, phase='close') -> str:
    return day + ('T01:30:00+00:00' if phase == 'open' else 'T07:00:00+00:00')


def fees_for(notional: Decimal, side: str) -> dict[str, Decimal]:
    if side not in ('buy', 'sell') or notional < 0:
        raise ValueError('INVALID_FEE_INPUT')
    return {'commission': commission(notional),
            'stamp_duty': money(notional * Decimal(POLICY['stamp_duty_sell'])) if side == 'sell' else ZERO,
            'transfer_fee': money(notional * Decimal(POLICY['transfer_fee_both']))}


class EquityAccountingEngine(AccountingEngine):
    def __init__(self, initial_cash: str, *, instruments: dict[str, str], sessions: list[str], flow_timing="period_end"):
        if not sessions or sorted(set(sessions)) != sessions or sessions[0] < POLICY['supported_fee_dates_from']:
            raise ValueError('UNSUPPORTED_OR_UNORDERED_CALENDAR')
        for day in sessions:
            instant(clock(day))
        if any(board not in ('MAIN', 'CHINEXT', 'STAR') for board in instruments.values()):
            raise ValueError('UNSUPPORTED_BOARD_OR_SPECIAL_STATUS')
        super().__init__(initial_cash, lot_size=100, asset_scope=tuple(instruments))
        self.instruments = dict(instruments)
        self.sessions = list(sessions)
        if flow_timing not in ("period_end", "period_start"):
            raise ValueError("INVALID_FLOW_TIMING")
        self.flow_timing = flow_timing
        self.references = {}
        self.pending_flow = ZERO
        self.flow_time = None

    def _next_day(self, day):
        if day not in self.sessions or self.sessions.index(day) + 1 >= len(self.sessions):
            raise ValueError('NEXT_SESSION_NOT_AVAILABLE')
        return self.sessions[self.sessions.index(day)+1]

    def _quantity_reason(self, o, e):
        held = self.total(o.asset_id, o.earliest_execution) - self.reserved_shares(o.asset_id)
        if self.instruments[o.asset_id] == 'STAR':
            valid = o.quantity >= 200 if o.side == 'buy' else o.quantity >= 200 or o.quantity == held
        else:
            valid = o.quantity % 100 == 0 if o.side == 'buy' else (o.quantity % 100 == 0 or o.quantity % 100 == held % 100)
        return None if valid else 'quantity_constraint'

    def _submission_sellable_time(self, order, event):
        return order.earliest_execution

    def _fill_quantity(self, order, capacity):
        # Order units constrain submitted instructions, not exchange partial fills.
        return min(order.remaining, capacity)

    def buy_reserve(self, order):
        if not order.remaining:
            return ZERO
        value = money(order.limit_price * order.remaining)
        return value + sum(fees_for(order.notional+value, 'buy').values(), ZERO) - sum(fees_for(order.notional, 'buy').values(), ZERO)

    def _fee_deltas(self, order, value, event):
        before, after = fees_for(order.notional, order.side), fees_for(order.notional+value, order.side)
        return {key: after[key]-before[key] for key in after}

    def _submit(self, e):
        day = instant(e['time']).date().isoformat()
        if e['time'] != clock(day) or e['earliest_execution'] != clock(self._next_day(day), 'open'):
            raise ValueError('REQUIRES_CLOSE_DECISION_NEXT_SESSION_OPEN')
        if instant(e['signal_available_at']) > instant(e['decision_time']):
            raise ValueError('FUTURE_SIGNAL')
        return super()._submit(e)

    def validate_bar(self, asset, bar):
        if bar.get('normal_session') is not True:
            raise ValueError('SPECIAL_SESSION_REQUIRES_SEPARATE_RULE_POLICY')
        ref = decimal(bar['reference_close'])
        if ref <= 0 or ref != money(ref):
            raise ValueError('INVALID_REFERENCE_CLOSE')
        limit = Decimal('.10' if self.instruments[asset] == 'MAIN' else '.20')
        lower, upper = money(ref*(1-limit)), money(ref*(1+limit))
        values = [decimal(bar[key]) for key in ('open', 'high', 'low', 'close')]
        op, high, low, close = values
        if any(v <= 0 or v != money(v) or not lower <= v <= upper for v in values):
            raise ValueError('BAR_OUTSIDE_PRICE_LIMIT')
        if not low <= min(op, close) <= max(op, close) <= high:
            raise ValueError('INVALID_OHLC')
        return lower, upper

    def _execute(self, e):
        o = self.orders[e['order_id']]
        day = instant(e['time']).date().isoformat()
        if e['time'] != clock(day, 'open'):
            raise ValueError('OPEN_EXECUTION_REQUIRED')
        if e['sellable_at'] != clock(self._next_day(day), 'open'):
            raise ValueError('T_PLUS_ONE_SETTLEMENT_REQUIRED')
        if e['time'] != o.earliest_execution and o.status == 'open':
            return 'expired_session_order'
        # Only information at open is used here: never inspect the day's high,
        # low or close to decide if an opening fill is possible.
        ref = decimal(e['reference_close'])
        if ref <= 0 or ref != money(ref) or e.get('normal_session') is not True:
            raise ValueError('INVALID_EXECUTION_REFERENCE')
        if o.asset_id in self.references and ref != self.references[o.asset_id]:
            raise ValueError('REFERENCE_CLOSE_MISMATCH')
        limit = Decimal('.10' if self.instruments[o.asset_id] == 'MAIN' else '.20')
        down, up = money(ref*(1-limit)), money(ref*(1+limit))
        price = decimal(e['price'])
        if not down <= price <= up:
            raise ValueError('EXECUTION_OUTSIDE_PRICE_LIMIT')
        if e['halted']:
            return super()._execute(e)
        if (o.side == 'buy' and price == up) or (o.side == 'sell' and price == down):
            return super()._execute({**e, 'side_blocked': True})
        return super()._execute(e)

    def _external_flow(self, e):
        day = instant(e['time']).date().isoformat()
        expected_time = clock(day) if self.flow_timing == 'period_end' else day+'T01:29:00+00:00'
        if e['time'] != expected_time or day not in self.sessions or any(r['date'] == day for r in self.tables['daily_nav']):
            raise ValueError('EXTERNAL_FLOW_REQUIRES_UNMARKED_SESSION_END')
        amount = money(e['amount'])
        if amount == 0 or amount != decimal(e['amount']):
            raise ValueError('INVALID_EXTERNAL_FLOW')
        if self.cash + amount < self.reserved_cash:
            return 'insufficient_cash_for_withdrawal'
        self._cash(e, 'external_flow', amount, e['event_id'])
        self.pending_flow += amount
        self.flow_time = e['time']
        return 'external_flow_booked'

    def _mark(self, e):
        day = instant(e['time']).date().isoformat()
        if day not in self.sessions or e['time'] != clock(day):
            raise ValueError('SESSION_CLOSE_MARK_REQUIRED')
        if self.flow_time is not None and instant(self.flow_time).date() != instant(e['time']).date():
            raise ValueError('EXTERNAL_FLOW_MUST_PRECEDE_SAME_SESSION_MARK')
        for order in self.orders.values():
            if order.status == 'open' and instant(order.earliest_execution).date().isoformat() <= day:
                order.status, order.reason = 'expired', 'unfilled_session_order'
        for asset, q in e['quotes'].items():
            if q.get('carry_reason') == 'synthetic_halt_last_mark':
                if q.get('halted') is not True or asset not in self.references or decimal(q['price']) != self.references[asset]:
                    raise ValueError('HALT_CARRY_REQUIRES_PRIOR_MARK')
                continue
            bar = q['bar']
            self.validate_bar(asset, bar)
            if decimal(q['price']) != decimal(bar['close']):
                raise ValueError('CLOSE_MARK_MISMATCH')
            if asset in self.references and decimal(bar['reference_close']) != self.references[asset]:
                raise ValueError('REFERENCE_CLOSE_MISMATCH')
            self.references[asset] = decimal(q['price'])
        result = super()._mark(e)
        self.tables['daily_nav'][-1]['external_flow'] = self.pending_flow
        self.pending_flow, self.flow_time = ZERO, None
        return result

    def _split(self, e):
        result = super()._split(e)
        asset = e['asset_id']
        if asset in self.references:
            self.references[asset] = money(self.references[asset]*Decimal(e['denominator'])/Decimal(e['numerator']))
        return result

    def _dividend_record(self, e):
        result = super()._dividend_record(e)
        asset = e['asset_id']
        if asset in self.references:
            self.references[asset] = money(self.references[asset]-decimal(e['cash_per_share']))
            if self.references[asset] <= 0:
                raise ValueError('INVALID_EX_DIVIDEND_REFERENCE')
        return result

    def affordable_quantity(self, asset, budget, price, requested):
        """Largest legal buy fitting all fees; pure before-execution sizing."""
        price, budget = decimal(price), decimal(budget)
        if price <= 0 or requested < 0 or budget < 0:
            raise ValueError('INVALID_SIZING_INPUT')
        step, minimum = (1, 200) if self.instruments[asset] == 'STAR' else (100, 100)
        low, high = 0, requested // step
        while low < high:
            mid = (low+high+1)//2
            value = money(price*mid*step)
            if value+sum(fees_for(value, 'buy').values(), ZERO) <= budget:
                low = mid
            else:
                high = mid-1
        return low*step if low*step >= minimum else 0
