#!/usr/bin/env python3
"""Verify Tushare endpoint access without persisting or printing the token."""

from __future__ import annotations

import getpass
import hashlib
import json
import sys
import urllib.error
import urllib.request


API_URL = "https://api.tushare.pro"

CHECKS = {
    "required.security": {
        "stock_basic": {"list_status": "L"},
        "bak_basic": {"trade_date": "20250731"},
        "trade_cal": {"exchange": "SSE", "start_date": "20250701", "end_date": "20250710"},
        "stock_st": {"trade_date": "20250731"},
        "suspend_d": {"trade_date": "20250731"},
    },
    "required.market": {
        "daily": {"trade_date": "20250731"},
        "adj_factor": {"trade_date": "20250731"},
        "daily_basic": {"trade_date": "20250731"},
    },
    "required.financial": {
        "balancesheet": {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20251231"},
        "disclosure_date": {"end_date": "20241231"},
    },
    "required.classification": {
        "index_classify": {"level": "L1", "src": "SW2021"},
        "index_member_all": {"l1_code": "801010.SI", "is_pub": "1"},
    },
    "required.risk_free": {
        "shibor": {"start_date": "20250701", "end_date": "20250710"},
        "repo_daily": {"trade_date": "20250731"},
    },
    "optional_validation": {
        "income": {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20251231"},
        "cashflow": {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20251231"},
        "fina_indicator": {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20251231"},
        "dividend": {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20251231"},
    },
}


def request_api(api_name: str, token: str, params: dict[str, str]) -> dict:
    payload = json.dumps(
        {"api_name": api_name, "token": token, "params": params, "fields": ""}
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "factor-matrix-audit/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    token = getpass.getpass("Tushare token (hidden, never saved): ").strip()
    if not token:
        print("Token is required.", file=sys.stderr)
        return 2

    fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    print(json.dumps({"token_fingerprint_sha256": fingerprint, "results": []}))

    required_failed = False
    for group, checks in CHECKS.items():
        for api_name, params in checks.items():
            result = {"group": group, "api": api_name}
            try:
                response = request_api(api_name, token, params)
                code = response.get("code")
                data = response.get("data") or {}
                fields = data.get("fields") or []
                items = data.get("items") or []
                result.update(
                    status="available" if code == 0 else "denied",
                    code=code,
                    rows=len(items),
                    fields=fields,
                    message=response.get("msg") or "",
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                result.update(status="network_error", code=None, rows=0, fields=[], message=str(exc))
            if group.startswith("required") and result["status"] != "available":
                required_failed = True
            print(json.dumps(result, ensure_ascii=False))

    token = ""  # Minimize lifetime of the plaintext reference.
    return 1 if required_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
