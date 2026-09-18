from __future__ import annotations

import getpass
import hashlib
import os
import shutil
import subprocess


KEYCHAIN_SERVICE = "factor-matrix-tushare"


def _load_from_local_module() -> str:
    """Load the optional developer-local token module.

    The module is intentionally gitignored and must never be committed.
    """
    try:
        from ._local_secrets import TUSHARE_TOKEN
    except (ImportError, AttributeError):
        return ""
    return str(TUSHARE_TOKEN).strip()


def _load_from_macos_keychain() -> str:
    if not shutil.which("security"):
        return ""
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_tushare_token() -> str:
    """Load a token without persisting it or including it in logs."""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        token = _load_from_local_module()
    if not token:
        token = _load_from_macos_keychain()
    if not token:
        token = getpass.getpass("Tushare token (hidden, never saved): ").strip()
    if not token:
        raise RuntimeError("Tushare token is required")
    return token


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
