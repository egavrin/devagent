import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_dev_agent.testing.coverage_gate import CoverageGate, CoverageResult, check_coverage


def _write_coverage(tmp_path: Path, coverage_payload: dict) -> None:
    (tmp_path / "coverage.json").write_text(json.dumps(coverage_payload))


def test_run_coverage_parses_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".coveragerc").write_text("")

    coverage_payload = {
        "totals": {"percent_covered": 82.5},
        "files": {
            "ai_dev_agent/module_a.py": {"summary": {"percent_covered": 95.0}},
            "ai_dev_agent/module_b.py": {"summary": {"percent_covered": 70.0}},
        },
    }
    _write_coverage(tmp_path, coverage_payload)

    recorded = {}

    def fake_run(cmd, capture_output, text, cwd):
        recorded["cmd"] = cmd
        return SimpleNamespace(stdout="coverage ok")

    # Pretend pytest-xdist is available so we exercise the parallel branch
    monkeypatch.setitem(sys.modules, "pytest_xdist", object())
    monkeypatch.setattr("ai_dev_agent.testing.coverage_gate.subprocess.run", fake_run)

    gate = CoverageGate(threshold=75.0)
    result = gate.run_coverage()

    assert result.passed is True
    assert result.total_coverage == pytest.approx(82.5)
    assert result.uncovered_files == ["ai_dev_agent/module_b.py: 70.0%"]
    assert recorded["cmd"][-1] == "tests/"
    assert "-n" in recorded["cmd"] and "auto" in recorded["cmd"]


def test_get_file_coverage_normalizes_windows_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    gate = CoverageGate(threshold=80.0)

    win_path = str(tmp_path / "ai_dev_agent" / "module_c.py").replace("/", "\\")
    coverage_payload = {
        "files": {
            win_path: {
                "summary": {"percent_covered": 75.0},
            }
        }
    }

    files = gate._get_file_coverage(coverage_payload)

    # Expect normalized relative path even when coverage.json contains Windows style paths
    assert files == {"ai_dev_agent/module_c.py": 75.0}


def test_enforce_failure_reports_and_respects_exit(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    gate = CoverageGate(threshold=90.0)

    result = CoverageResult(
        total_coverage=80.0,
        passed=False,
        threshold=90.0,
        uncovered_files=["ai_dev_agent/foo.py: 75.0%"],
        report="FAIL",
        details={"ai_dev_agent/foo.py": 75.0},
    )

    monkeypatch.setattr(CoverageGate, "run_coverage", lambda self: result)

    # exit_on_fail=False returns False without raising SystemExit
    assert gate.enforce(exit_on_fail=False) is False
    out = capsys.readouterr().out
    assert "Coverage check failed" in out
    assert "ai_dev_agent/foo.py: 75.0%" in out


def test_check_coverage_uses_gate(monkeypatch):
    monkeypatch.setattr(
        "ai_dev_agent.testing.coverage_gate.CoverageGate.run_coverage",
        lambda self: CoverageResult(
            total_coverage=99.0,
            passed=True,
            threshold=self.threshold,
            uncovered_files=[],
            report="ok",
            details={},
        ),
    )

    assert check_coverage(threshold=85.0) is True
