import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_dev_agent.testing.coverage_report import CoverageReporter, generate_report


def _write_coverage(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "coverage.json").write_text(json.dumps(payload))


def test_generate_terminal_report_and_summary(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    def fake_run(cmd, cwd, capture_output=False, text=False, check=False):
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "coverage"]
        return SimpleNamespace(stdout="ok")

    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.subprocess.run", fake_run)

    # No coverage file yet -> summary returns error
    summary = reporter.get_coverage_summary()
    assert summary["error"].startswith("No coverage")

    payload = {
        "totals": {
            "percent_covered": 91.2,
            "percent_covered_display": "91.20%",
            "num_statements": 10,
            "covered_lines": 9,
            "missing_lines": 1,
            "excluded_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "missing_branches": 0,
        },
        "files": {
            str(tmp_path / "ai_dev_agent" / "alpha.py").replace("/", "\\"): {
                "summary": {
                    "percent_covered": 50.0,
                    "num_statements": 4,
                    "missing_lines": 2,
                }
            }
        },
    }
    _write_coverage(tmp_path, payload)

    # Terminal report uses subprocess; fake run returns text
    result = reporter.generate_terminal_report()
    assert result == "ok"

    summary = reporter.get_coverage_summary()
    assert summary["total_coverage"] == pytest.approx(91.2)
    assert summary["files"] == 1


def test_file_coverage_categories_and_path_normalization(tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    payload = {
        "files": {
            str(tmp_path / "ai_dev_agent" / "alpha.py").replace("/", "\\"): {
                "summary": {"percent_covered": 97.0, "num_statements": 10, "missing_lines": 0}
            },
            "beta.py": {
                "summary": {"percent_covered": 85.0, "num_statements": 5, "missing_lines": 1}
            },
            "gamma.py": {
                "summary": {"percent_covered": 65.0, "num_statements": 5, "missing_lines": 2}
            },
            "delta.py": {
                "summary": {"percent_covered": 55.0, "num_statements": 5, "missing_lines": 3}
            },
        }
    }
    _write_coverage(tmp_path, {"totals": {}, **payload})

    coverage = reporter.get_file_coverage(threshold=90.0)

    assert [item["path"] for item in coverage["excellent"]] == ["ai_dev_agent/alpha.py"]
    assert [item["path"] for item in coverage["good"]] == ["beta.py"]
    assert [item["path"] for item in coverage["fair"]] == ["gamma.py"]
    assert [item["path"] for item in coverage["poor"]] == ["delta.py"]


def test_trend_persistence_and_analysis(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    payload = {
        "totals": {
            "percent_covered": 80.0,
            "percent_covered_display": "80.0%",
            "num_statements": 100,
            "covered_lines": 80,
            "missing_lines": 20,
            "excluded_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "missing_branches": 0,
        },
        "files": {},
    }
    _write_coverage(tmp_path, payload)

    monkeypatch.setattr(
        "ai_dev_agent.testing.coverage_report.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="deadbeef"),
    )

    reporter.save_coverage_trend()
    reporter.save_coverage_trend()

    trends = reporter.load_coverage_trends()
    assert len(trends) == 2
    analysis = reporter.get_coverage_trend_analysis()
    assert analysis["data_points"] == 2
    assert "history" in analysis


def test_generate_reports(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reporter = CoverageReporter(project_root=tmp_path)
    payload = {
        "totals": {
            "percent_covered": 96.0,
            "percent_covered_display": "96.0%",
            "num_statements": 100,
            "covered_lines": 96,
            "missing_lines": 4,
            "excluded_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "missing_branches": 0,
        },
        "files": {},
    }
    _write_coverage(tmp_path, payload)

    badge = reporter.generate_badge_data()
    assert badge["message"] == "96.0%"
    assert badge["color"] == "brightgreen"

    markdown = reporter.generate_markdown_report()
    assert "## Overall Coverage" in markdown

    out_path = tmp_path / "report.json"
    json_report = generate_report(format="json", output_file=out_path)
    assert json.loads(json_report)["total_coverage"] == pytest.approx(96.0)
    assert out_path.exists()


def test_generate_html_report_success(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    htmlcov_dir = tmp_path / "htmlcov"
    htmlcov_dir.mkdir()
    (htmlcov_dir / "index.html").write_text("<html></html>")

    calls = []

    def fake_run(cmd, cwd, check):
        calls.append((tuple(cmd), cwd, check))

    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.subprocess.run", fake_run)

    report_path = reporter.generate_html_report()
    assert report_path == htmlcov_dir / "index.html"
    assert calls and calls[0][0][:3] == (sys.executable, "-m", "coverage")


def test_generate_html_report_failure(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    def fake_run(cmd, cwd, check):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        reporter.generate_html_report()


def test_terminal_report_failure(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)

    def fake_run(cmd, cwd, capture_output, text, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="boom")

    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.subprocess.run", fake_run)
    result = reporter.generate_terminal_report()
    assert "non-zero exit status" in result


def test_file_coverage_error_when_missing(tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)
    assert reporter.get_file_coverage() == {"error": "No coverage data available"}


def test_save_trend_handles_missing_summary(monkeypatch, tmp_path, caplog):
    reporter = CoverageReporter(project_root=tmp_path)
    caplog.set_level("WARNING")

    reporter.save_coverage_trend()
    assert "Cannot save trend" in caplog.text


def test_save_trend_handles_git_failure(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)
    payload = {
        "totals": {
            "percent_covered": 90.0,
            "percent_covered_display": "90.0",
            "num_statements": 10,
            "covered_lines": 9,
            "missing_lines": 1,
            "excluded_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "missing_branches": 0,
        },
        "files": {},
    }
    _write_coverage(tmp_path, payload)

    def failing_git(*args, **kwargs):
        raise RuntimeError("git not available")

    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.subprocess.run", failing_git)

    reporter.save_coverage_trend()
    trends = reporter.load_coverage_trends()
    assert len(trends) == 1
    assert trends[0].commit_hash is None


def test_load_trends_handles_invalid_json(tmp_path, caplog):
    reporter = CoverageReporter(project_root=tmp_path)
    reporter.trends_file.write_text("{not json")
    caplog.set_level("ERROR")

    assert reporter.load_coverage_trends() == []
    assert "Failed to load trends" in caplog.text


def test_generate_markdown_with_low_coverage(monkeypatch, tmp_path):
    reporter = CoverageReporter(project_root=tmp_path)
    payload = {
        "totals": {
            "percent_covered": 40.0,
            "percent_covered_display": "40.0%",
            "num_statements": 50,
            "covered_lines": 20,
            "missing_lines": 30,
            "excluded_lines": 0,
            "num_branches": 0,
            "covered_branches": 0,
            "missing_branches": 0,
        },
        "files": {
            "foo.py": {
                "summary": {"percent_covered": 40.0, "num_statements": 5, "missing_lines": 3}
            }
        },
    }
    _write_coverage(tmp_path, payload)

    markdown = reporter.generate_markdown_report()
    assert "🔴 Needs Improvement" in markdown


def test_generate_report_html_and_unknown(tmp_path, monkeypatch):
    reporter = CoverageReporter(project_root=tmp_path)
    monkeypatch.setattr("ai_dev_agent.testing.coverage_report.CoverageReporter", lambda: reporter)

    def fake_html():
        return tmp_path / "htmlcov" / "index.html"

    monkeypatch.setattr(reporter, "generate_html_report", fake_html)

    result = generate_report(format="html")
    assert result.endswith("htmlcov/index.html")

    with pytest.raises(ValueError):
        generate_report(format="yaml")
