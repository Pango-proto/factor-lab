from __future__ import annotations

import json
import http.client
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class TushareError(RuntimeError):
    pass


@dataclass(frozen=True)
class TushareResponse:
    api_name: str
    params: dict[str, Any]
    fields: list[str]
    items: list[list[Any]]
    raw: dict[str, Any]


class TushareClient:
    def __init__(
        self,
        token: str,
        *,
        url: str = "https://api.tushare.pro",
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._token = token
        self._url = url
        self._retries = retries
        self._retry_delay = retry_delay

    def query(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> TushareResponse:
        safe_params = params or {}
        payload = json.dumps(
            {
                "api_name": api_name,
                "token": self._token,
                "params": safe_params,
                "fields": ",".join(fields or []),
            },
            separators=(",", ":"),
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self._retries):
            request = urllib.request.Request(
                self._url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "factor-matrix/0.1",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                if raw.get("code") != 0:
                    raise TushareError(
                        f"{api_name} failed with code={raw.get('code')}: {raw.get('msg', '')}"
                    )
                data = raw.get("data") or {}
                return TushareResponse(
                    api_name=api_name,
                    params=safe_params,
                    fields=data.get("fields") or [],
                    items=data.get("items") or [],
                    raw=raw,
                )
            except urllib.error.HTTPError as exc:
                if not 500 <= exc.code < 600:
                    raise TushareError(
                        f"{api_name} HTTP failure status={exc.code}"
                    ) from exc
                last_error = exc
                if attempt + 1 < self._retries:
                    time.sleep(self._retry_delay * (2**attempt))
            except (
                urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected,
                json.JSONDecodeError, UnicodeDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self._retries:
                    time.sleep(self._retry_delay * (2**attempt))
        raise TushareError(f"{api_name} network failure: {last_error}")

    def close(self) -> None:
        self._token = ""
