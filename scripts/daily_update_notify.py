"""Run the daily update and surface one useful macOS notification."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / ".venv" / "bin" / "factor-matrix"


def notification_kind(returncode: int, stdout: str, local_hour: int) -> str | None:
    if returncode == 0 and "Daily update complete:" in stdout:
        return "success"
    if returncode != 0 and local_hour >= 23:
        return "failure"
    return None


def notify(title: str, message: str) -> None:
    script = (
        "on run argv\n"
        "display notification (item 2 of argv) with title (item 1 of argv)\n"
        "end run"
    )
    subprocess.run(
        ["/usr/bin/osascript", "-e", script, title, message],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    command = [
        str(CLI),
        "daily-update",
        "--data-root",
        str(PROJECT_ROOT / "data"),
        *sys.argv[1:],
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    kind = notification_kind(result.returncode, result.stdout, datetime.now().hour)
    if kind == "success":
        completed = next(
            (line for line in result.stdout.splitlines() if line.startswith("Daily update complete:")),
            "今日 L0 行情事实与状态快照已发布",
        )
        notify("A 股数据更新完成", completed.replace("Daily update complete: ", "交易日 "))
    elif kind == "failure":
        notify("A 股数据更新失败", "三次盘后尝试仍未完成，请查看数据管道错误日志")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
