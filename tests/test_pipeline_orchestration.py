from pathlib import Path

from scripts.check_architecture_invariants import orchestration_violations


def test_current_pipeline_has_no_script_to_script_edges() -> None:
    project = Path(__file__).resolve().parents[1]
    assert orchestration_violations(project) == []


def test_script_to_script_subprocess_is_rejected(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "bad.py").write_text(
        "import subprocess\nsubprocess.run(['python', 'scripts/other.py'], check=True)\n",
        encoding="utf-8",
    )
    violations = orchestration_violations(tmp_path)
    assert "subprocess targets another pipeline script" in violations[0]


def test_shell_to_shell_pipeline_edge_is_rejected(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "bad.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nbash scripts/other.sh\n",
        encoding="utf-8",
    )
    violations = orchestration_violations(tmp_path)
    assert "shell script invokes another pipeline script" in violations[0]
