from datetime import date

import pytest

from factor_matrix.market_catchup import backfill_market, open_dates
from factor_matrix.source import TushareResponse
from factor_matrix.storage import DataLake


def response(rows):
    return TushareResponse("trade_cal", {}, ["cal_date", "is_open"], rows, {})


def test_calendar_returns_open_dates_in_chronological_order():
    assert open_dates(response([["20260916", 1], ["20260915", 0], ["20260914", 1]]),
                      date(2026, 9, 14), date(2026, 9, 16)) == [date(2026, 9, 14), date(2026, 9, 16)]


@pytest.mark.parametrize("rows", [[], [["20260915", 1]], [["20260916", 1], ["20260916", 1]],
                                   [["20260915", 1], ["20260916", 3]]])
def test_incomplete_or_ambiguous_calendar_is_not_silently_skipped(rows):
    with pytest.raises(ValueError):
        open_dates(response(rows), date(2026, 9, 15), date(2026, 9, 16))


@pytest.mark.parametrize("start,end", [(date(2026, 9, 17), date(2026, 9, 17)),
                                       (date(2026, 9, 16), date(2026, 9, 18)),
                                       (date(2026, 9, 16), date(2026, 9, 15))])
def test_no_unfinished_day_or_invalid_range_before_network(tmp_path, start, end):
    with pytest.raises(ValueError, match="COMPLETED_PRIOR_DAYS"):
        backfill_market(None, DataLake(tmp_path), start=start, end=end, credential_fingerprint="fixture",
                        today=date(2026, 9, 17))
