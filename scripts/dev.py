"""Run the private Python API and Vite as one local development service."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = os.environ.copy()
    source_root = str(ROOT / "src-python")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH", "")) if item
    )
    if not environment.get("UV_CACHE_DIR"):
        uv_cache_dir = ROOT / ".uv-cache"
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        environment["UV_CACHE_DIR"] = str(uv_cache_dir)
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "factor_matrix.cli",
            "serve-api",
        ],
        cwd=ROOT,
        env=environment,
    )
    vite = subprocess.Popen(
        [str(ROOT / "node_modules" / ".bin" / "vite")], cwd=ROOT, env=environment,
    )
    processes = [api, vite]

    def stop(*_: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return next((process.returncode for process in processes if process.returncode), 0) or 0


if __name__ == "__main__":
    raise SystemExit(main())
