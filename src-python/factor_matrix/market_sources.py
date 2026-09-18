"""Single source of truth for Tushare market-ingestion contracts."""

SECURITY_FIELDS = [
    "ts_code", "symbol", "name", "area", "industry", "market", "exchange",
    "curr_type", "list_status", "list_date", "delist_date",
]

REQUIRED_SOURCE_FIELDS = {
    "stock_basic": {"ts_code", "list_status", "list_date", "delist_date"},
    "trade_cal": {"exchange", "cal_date", "is_open", "pretrade_date"},
    "daily": {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"},
    "adj_factor": {"ts_code", "trade_date", "adj_factor"},
    "daily_basic": {"ts_code", "trade_date", "total_share", "float_share", "free_share", "total_mv", "circ_mv"},
    "suspend_d": {"ts_code", "trade_date", "suspend_type"},
    "stock_st": {"ts_code", "trade_date", "type"},
    "namechange": {"ts_code", "name", "start_date", "end_date"},
    "stk_limit": {"ts_code", "trade_date", "up_limit", "down_limit"},
    "bse_mapping": {"name", "o_code", "n_code", "list_date"},
}

PRICE_LIMIT_FIELDS = ["ts_code", "trade_date", "pre_close", "up_limit", "down_limit"]
