from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ai_dev_agent.engine.metrics.coverage import compute_patch_coverage


@pytest.fixture()
def coverage_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    module = repo / "module.py"
    module.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"], cwd=repo, check=True, capture_output=True, env=env
    )
    return repo


def test_patch_coverage_reports_ratio(coverage_repo: Path) -> None:
    module = coverage_repo / "module.py"
    module.write_text("def greet():\n    message = 'hi'\n    return message\n", encoding="utf-8")
    coverage_xml = coverage_repo / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage>
  <packages>
    <package>
      <classes>
        <class filename="module.py">
          <lines>
            <line number="2" hits="1"/>
            <line number="3" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""".strip(),
        encoding="utf-8",
    )
    result = compute_patch_coverage(coverage_repo, coverage_xml=coverage_xml)
    assert result is not None
    assert result.total_lines == 2
    assert result.covered_lines == 2
    assert result.ratio == pytest.approx(1.0)


def test_patch_coverage_handles_uncovered_lines(coverage_repo: Path) -> None:
    module = coverage_repo / "module.py"
    module.write_text("def add(x, y):\n    return x + y\n", encoding="utf-8")
    coverage_xml = coverage_repo / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage>
  <packages>
    <package>
      <classes>
        <class filename="module.py">
          <lines>
            <line number="2" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""".strip(),
        encoding="utf-8",
    )
    result = compute_patch_coverage(coverage_repo, coverage_xml=coverage_xml)
    assert result is not None
    assert result.total_lines >= 1
    assert result.covered_lines == 0
    assert result.ratio == pytest.approx(0.0)
