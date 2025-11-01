from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ai_dev_agent.engine.metrics.diff import compute_diff_metrics


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    (repo_root / "README.md").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=env,
    )
    return repo_root


def test_compute_diff_metrics_counts_lines(git_repo: Path) -> None:
    target = git_repo / "README.md"
    target.write_text("initial\nupdated\nextra\n", encoding="utf-8")
    metrics = compute_diff_metrics(git_repo)
    assert metrics.total_lines == 2
    assert metrics.file_count == 1
    assert metrics.concentration == pytest.approx(2.0)
    assert metrics.files == ["README.md"]


def test_compute_diff_metrics_includes_untracked_files(git_repo: Path) -> None:
    new_file = git_repo / "new.py"
    new_file.write_text("print('hi')\n", encoding="utf-8")
    metrics = compute_diff_metrics(git_repo, include_untracked=True)
    assert "new.py" in metrics.files
    assert metrics.total_lines >= 1
