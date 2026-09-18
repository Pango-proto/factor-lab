import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "daily_update_notify.py"
SPEC = importlib.util.spec_from_file_location("daily_update_notify", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_notifies_once_when_new_snapshot_is_published() -> None:
    assert MODULE.notification_kind(0, "Daily update complete: 2026-08-07", 21) == "success"


def test_does_not_notify_for_idempotent_or_closed_day_skip() -> None:
    assert MODULE.notification_kind(0, "Daily update already current: 2026-08-07", 22) is None
    assert MODULE.notification_kind(0, "Daily update skipped: closed", 21) is None


def test_failure_notification_waits_until_final_attempt() -> None:
    assert MODULE.notification_kind(1, "", 21) is None
    assert MODULE.notification_kind(1, "", 22) is None
    assert MODULE.notification_kind(1, "", 23) == "failure"


def test_scheduled_command_matches_current_cli(monkeypatch):
    from types import SimpleNamespace
    from factor_matrix.cli_parser import build_parser
    commands = []
    monkeypatch.setattr(MODULE.sys, "argv", ["daily_update_notify.py", "--date", "2026-09-16"])
    monkeypatch.setattr(MODULE.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert MODULE.main() == 0
    args = build_parser().parse_args(commands[0][1:])
    assert args.command == "daily-update" and args.date.isoformat() == "2026-09-16"
