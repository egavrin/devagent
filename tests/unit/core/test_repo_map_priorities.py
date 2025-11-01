"""Focused tests for RepoMap heuristics and filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_dev_agent.core.repo_map import RepoMap


@pytest.fixture
def repo(tmp_path) -> RepoMap:
    return RepoMap(tmp_path)


def test_should_skip_generated_file_and_directory(repo, tmp_path):
    generated_file = tmp_path / "build" / "app_generated.py"
    generated_file.parent.mkdir()
    generated_file.touch()

    skip_file, reason_file = repo._should_skip_file(generated_file)
    skip_dir, reason_dir = repo._should_skip_file(tmp_path / "dist" / "bundle.js")

    assert skip_file is True
    assert "generated" in reason_file
    assert skip_dir is True
    assert "generated directory" in reason_dir


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/test_example.py", 0.1),
        ("templates/email.html", 0.2),
        ("examples/demo_app.py", 0.3),
        ("benchmarks/perf.py", 0.4),
        ("src/core/service.py", 1.5),
        ("plugins/auth/provider.py", 1.2),
        ("lib/app/main.py", 1.3),
    ],
)
def test_priority_multiplier_matches_conventions(repo, path, expected):
    assert repo._get_file_priority_multiplier(path) == pytest.approx(expected)


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("i", True),
        ("data", True),
        ("RESULT", True),
        ("42", True),
        ("ValidName", False),
    ],
)
def test_is_noisy_symbol(repo, symbol, expected):
    assert repo._is_noisy_symbol(symbol) is expected


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("USER_ID", True),
        ("user_id", True),
        ("UserId", True),
        ("f", False),
        ("", False),
    ],
)
def test_is_well_named_symbol(repo, symbol, expected):
    assert repo._is_well_named_symbol(symbol) is expected
