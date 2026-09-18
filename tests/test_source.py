import http.client
import json

from factor_matrix.source import TushareClient


class FakeResponse:
    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        if self.error:
            raise self.error
        return self.payload


def test_tushare_retries_incomplete_chunk_without_retrying_business_errors(monkeypatch):
    calls = []
    responses = [
        FakeResponse(error=http.client.IncompleteRead(b"partial")),
        FakeResponse(json.dumps({
            "code": 0, "data": {"fields": ["trade_date"], "items": [["20210104"]]},
        }).encode()),
    ]

    def fake_open(*_, **__):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    client = TushareClient("secret", retries=2, retry_delay=0)
    result = client.query("daily", {"trade_date": "20210104"})
    assert len(calls) == 2
    assert result.items == [["20210104"]]
