"""Additional tests covering review execution helpers and fallback paths."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from ai_dev_agent.core.utils.config import Settings

review_module = importlib.import_module("ai_dev_agent.cli.review")


def _make_hunk(
    *,
    header: str = "@@ -1,1 +1,1 @@",
    context: list[tuple[int, str]] | None = None,
    added: list[tuple[int, str]] | None = None,
    removed: list[tuple[int, str]] | None = None,
) -> dict[str, object]:
    return {
        "header": header,
        "context_lines": [
            {"line_number": lineno, "content": text} for lineno, text in (context or [])
        ],
        "added_lines": [{"line_number": lineno, "content": text} for lineno, text in (added or [])],
        "removed_lines": [
            {"line_number": lineno, "content": text} for lineno, text in (removed or [])
        ],
    }


@pytest.fixture(autouse=True)
def _clear_patch_cache():
    review_module._PATCH_CACHE.clear()
    yield
    review_module._PATCH_CACHE.clear()


def test_format_patch_dataset_handles_empty_and_filter_messages() -> None:
    """Ensure dataset formatter covers empty patches and filter branches."""
    empty = review_module.format_patch_dataset({})
    assert "No files with additions" in empty

    parsed_patch = {
        "files": [
            {
                "path": "src/example.py",
                "language": "python",
                "change_type": "modified",
                "_chunk_index": 0,
                "_chunk_total": 2,
                "hunks": [],
            }
        ]
    }

    no_match = review_module.format_patch_dataset(parsed_patch, filter_pattern=r"unmatched")
    assert "No files matching pattern" in no_match

    invalid_pattern = review_module.format_patch_dataset(parsed_patch, filter_pattern=r"[invalid")
    assert "FILE: src/example.py (segment 1/2)" in invalid_pattern


def test_split_large_file_entries_respects_limits() -> None:
    """Large files should be split into indexed chunks honoring limits."""
    file_entry = {
        "path": "src/component.py",
        "language": "python",
        "change_type": "modified",
        "hunks": [
            _make_hunk(added=[(10, "print('1')")]),
            _make_hunk(added=[(20, "print('2')")]),
            _make_hunk(added=[(30, "print('3')")]),
        ],
    }

    split_entries = review_module._split_large_file_entries(
        [file_entry],
        max_hunks_per_group=1,
        max_lines_per_group=1,
    )

    assert len(split_entries) == 3
    indices = {(entry["_chunk_index"], entry["_chunk_total"]) for entry in split_entries}
    assert indices == {(0, 3), (1, 3), (2, 3)}
    assert all(len(entry["hunks"]) == 1 for entry in split_entries)


def test_chunk_patch_files_applies_overlap() -> None:
    """Chunker should apply overlap when requested."""
    base_entries = [
        {
            "path": f"src/module_{idx}.py",
            "language": "python",
            "change_type": "modified",
            "hunks": [
                _make_hunk(
                    context=[(1, "def fn():")],
                    added=[(idx * 10 + 1, f"print('{idx}')")],
                )
            ],
        }
        for idx in range(3)
    ]

    chunks = review_module._chunk_patch_files(
        base_entries,
        max_files_per_chunk=2,
        max_lines_per_chunk=5,
        overlap_lines=2,
    )

    assert len(chunks) == 2
    first_paths = [entry["path"] for entry in chunks[0]]
    second_paths = [entry["path"] for entry in chunks[1]]
    assert first_paths == ["src/module_0.py", "src/module_1.py"]
    # Overlap should pull the last file from the previous chunk into the next
    assert second_paths[0] == "src/module_1.py"
    assert "src/module_2.py" in second_paths


def test_dynamic_limits_shrink_for_large_inputs() -> None:
    """Dynamic chunk limits should shrink when combined context grows."""
    massive_rule = "A" * 600_000  # ~150k tokens estimation
    entries = [
        {
            "path": "src/big.py",
            "hunks": [
                _make_hunk(
                    context=[(1, "def big():")],
                    added=[(2, "return 'x'"), (3, "return 'y'")],
                )
            ],
        }
    ]

    line_limit = review_module._compute_dynamic_line_limit(0, massive_rule, entries)
    file_limit = review_module._compute_dynamic_file_limit(0, massive_rule, entries)

    assert line_limit == 400  # Tightened from default
    assert file_limit == 10  # Shrinks default chunk size


def test_refine_chunks_for_token_budget_splits_large_sets() -> None:
    """Token budget refinement should split oversized chunks recursively."""
    entries = [
        {
            "path": f"src/segment_{idx}.py",
            "language": "python",
            "hunks": [
                _make_hunk(
                    header="@@ -0,0 +1,1 @@",
                    added=[(1, "value = 'x' * 100")],
                    context=[(0, "# context line" * 5)],
                )
            ],
        }
        for idx in range(2)
    ]

    refined = review_module._refine_chunks_for_token_budget(
        [entries], rule_text="rule", filter_pattern=None, token_budget=10
    )

    assert len(refined) == 2
    assert all(len(chunk) == 1 for chunk in refined)


def test_run_review_handles_invalid_model_output(monkeypatch, tmp_path) -> None:
    """Model returning invalid JSON should fall back to empty summary."""
    settings = Settings()
    settings.api_key = "dummy-key"
    settings.workspace_root = tmp_path

    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_file = tmp_path / "changes.patch"
    patch_file.write_text("fake patch content")
    rule_file = tmp_path / "rule.md"
    rule_file.write_text(
        "# Rule\n\n## Applies To\n[invalid\n\n## Description\nMissing bracket should trigger fallback."
    )

    parsed_patch = {
        "files": [
            {
                "path": "src/app.py",
                "language": "python",
                "change_type": "modified",
                "hunks": [_make_hunk(added=[(5, "print('hello')")])],
            }
        ]
    }

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _path: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda value: value)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    call_count = {"value": 0}

    def fake_execute(*_args, **_kwargs):
        call_count["value"] += 1
        return {"result": {"status": "ok"}, "final_json": "not-a-dict"}

    monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_file),
        rule_file=str(rule_file),
        json_output=False,
        settings=settings,
    )

    assert call_count["value"] == 1
    assert result["summary"]["total_violations"] == 0
    assert result["summary"]["files_reviewed"] == 1
    assert result["summary"]["rule_name"] == "rule"
    assert result["violations"] == []
    assert ctx.obj.get("_session_id") is None


def test_run_review_deduplicates_overlapping_chunks(monkeypatch, tmp_path) -> None:
    """Overlapping chunk processing should deduplicate identical violations."""
    settings = Settings()
    settings.api_key = "dummy-key"
    settings.workspace_root = tmp_path
    settings.review_max_hunks_per_chunk = 1
    settings.review_max_lines_per_chunk = 10
    settings.review_max_files_per_chunk = 3
    settings.review_chunk_overlap_lines = 10
    settings.review_token_budget = 20

    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_file = tmp_path / "big.patch"
    patch_file.write_text("placeholder")
    rule_file = tmp_path / "rule.md"
    rule_file.write_text(
        "# Rule\n\n## Applies To\nregex:src/module_a\\.py\n\n## Description\nEnsure safe logging."
    )

    parsed_patch = {
        "files": [
            {
                "path": "src/module_a.py",
                "language": "python",
                "change_type": "modified",
                "hunks": [
                    _make_hunk(added=[(10, "print('log A')")]),
                    _make_hunk(added=[(20, "print('log B')")]),
                ],
            },
            {
                "path": "src/module_b.py",
                "language": "python",
                "change_type": "modified",
                "hunks": [_make_hunk(added=[(5, "print('ignore')")])],
            },
        ]
    }

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _path: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda _value: "")
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    responses = [
        {
            "violations": [
                {
                    "file": "src/module_a.py",
                    "line": 10,
                    "message": "Use structured logging",
                    "code_snippet": "print('log A')",
                    "severity": "critical",
                }
            ],
            "summary": {"total_violations": 3, "files_reviewed": 5, "rule_name": ""},
        },
        {
            "violations": [
                {
                    "file": "src/module_a.py",
                    "line": 10,
                    "message": "Use structured logging",
                    "code_snippet": "print('log A')",
                    "severity": "critical",
                },
                {
                    "file": "src/module_a.py",
                    "line": 20,
                    "message": "Avoid bare print",
                    "code_snippet": "print('log B')",
                    "severity": "major",
                },
            ],
            "summary": {"total_violations": 2, "files_reviewed": 2, "rule_name": "Custom"},
        },
    ]

    def fake_execute(*_args, **_kwargs):
        payload = responses.pop(0)
        return {"result": {"status": "ok"}, "final_json": payload}

    monkeypatch.setattr(review_module, "_execute_react_assistant", fake_execute)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_file),
        rule_file=str(rule_file),
        json_output=True,
        settings=settings,
    )

    violations = result["violations"]
    assert len(violations) == 2  # Duplicate removed
    assert {v["line"] for v in violations} == {10, 20}
    assert all(v["severity"] in {"error", "warning"} for v in violations)
    assert result["summary"]["total_violations"] == 2
    assert result["summary"]["files_reviewed"] == 1
    assert result["summary"]["rule_name"] == "Custom"

    # JSON mode should not reset silent_mode flag to False
    assert ctx.obj.get("silent_mode") is True


def test_normalize_applies_to_pattern_variants() -> None:
    """Normalization should handle regex prefixes, globbing, and special tokens."""
    assert review_module._normalize_applies_to_pattern("") is None
    assert review_module._normalize_applies_to_pattern("   ") is None

    globbed = review_module._normalize_applies_to_pattern("src/*.py,docs/*.md")
    assert globbed is not None and "src" in globbed and "docs" in globbed
    assert globbed.count("$") >= 1

    regex_direct = review_module._normalize_applies_to_pattern("regex:^foo$")
    assert regex_direct == "^foo$"

    special_chars = review_module._normalize_applies_to_pattern("src/(.*).py")
    assert special_chars == "src/(.*).py"

    no_payload = review_module._normalize_applies_to_pattern("regex:")
    assert no_payload is None


def test_extract_applies_to_pattern_variants() -> None:
    """Rule parsing should support multiple metadata shapes."""
    rule_content = "# Rule\n\n## Applies To\n*.py"
    extracted = review_module.extract_applies_to_pattern(rule_content)
    assert extracted is not None and "\\.py" in extracted

    rule_content = "Applies To: regex:^src/.*\\.py$"
    extracted = review_module.extract_applies_to_pattern(rule_content)
    assert extracted == "^src/.*\\.py$"

    rule_content = "scope: lib/*"
    extracted = review_module.extract_applies_to_pattern(rule_content)
    assert "lib" in extracted

    rule_content = "# Rule\n\n## Applies To\nregex:"
    extracted = review_module.extract_applies_to_pattern(rule_content)
    assert extracted == "regex:"

    assert review_module.extract_applies_to_pattern("# Rule\n\nNo scope defined") is None


def test_parse_patch_file_handles_stat_failure(monkeypatch, tmp_path) -> None:
    """When stat fails, parsing should still succeed without caching."""
    patch_path = tmp_path / "example.patch"
    patch_path.write_text("diff --git a/foo.py b/foo.py\n")

    class DummyParser:
        def __init__(self, content: str, include_context: bool = False):
            self.content = content

        def parse(self):
            return {"files": []}

    monkeypatch.setattr(review_module, "PatchParser", DummyParser)

    def failing_stat(_self):
        raise OSError("stat failed")

    monkeypatch.setattr(Path, "stat", failing_stat)

    parsed = review_module.parse_patch_file(patch_path)
    assert parsed == {"files": []}


def test_parse_patch_file_uses_cache_and_eviction(monkeypatch, tmp_path) -> None:
    """Subsequent parses should use the cache and respect size limits."""
    patch_path = tmp_path / "cached.patch"
    patch_path.write_text("diff --git a/foo.py b/foo.py\n")

    calls: list[str] = []

    class DummyParser:
        def __init__(self, content: str, include_context: bool = False):
            calls.append(content)

        def parse(self):
            return {"files": []}

    monkeypatch.setattr(review_module, "PatchParser", DummyParser)

    def success_stat(_self):
        return SimpleNamespace(st_mtime=1000.0, st_size=5)

    monkeypatch.setattr(Path, "stat", success_stat)

    review_module._PATCH_CACHE.clear()
    for index in range(10):
        review_module._PATCH_CACHE[(f"path-{index}", 1.0, index)] = {"files": []}

    first = review_module.parse_patch_file(patch_path)
    second = review_module.parse_patch_file(patch_path)

    assert first == second == {"files": []}
    assert calls == ["diff --git a/foo.py b/foo.py\n"]
    # Cache should not exceed 10 entries due to eviction
    assert len(review_module._PATCH_CACHE) == 10
    assert (str(patch_path), 1000.0, 5) in review_module._PATCH_CACHE


def test_collect_patch_review_data_handles_invalid_entries() -> None:
    """Collection should ignore malformed hunk data gracefully."""
    parsed = {
        "files": [
            {"path": None},
            {
                "path": "src/app.py",
                "hunks": [
                    {"added_lines": [{"line_number": 5, "content": "print('ok')"}, "bad-entry"]},
                    {
                        "removed_lines": [
                            {"line_number": 1, "content": "print('old')"},
                            {"line_number": "invalid", "content": "print('skip')"},
                        ]
                    },
                ],
            },
        ]
    }

    added, removed, parsed_files = review_module.collect_patch_review_data(parsed)
    assert parsed_files == {"src/app.py"}
    assert added["src/app.py"][5] == "print('ok')"
    assert removed["src/app.py"][1] == "print('old')"


def test_format_patch_dataset_handles_missing_numbers() -> None:
    """Formatter should handle context/additions without explicit line numbers."""
    parsed_patch = {
        "files": [
            {
                "path": "src/app.py",
                "language": "python",
                "change_type": "modified",
                "hunks": [
                    {
                        "header": "@@ -1,3 +1,4 @@",
                        "context_lines": [{"content": "def fn():"}],
                        "added_lines": [{"content": "print('x')"}],
                        "removed_lines": [{"content": "pass"}],
                    }
                ],
            }
        ]
    }

    formatted = review_module.format_patch_dataset(parsed_patch)
    assert "      | def fn():" in formatted
    assert "      + print('x')" in formatted
    assert "      - pass" in formatted


def test_chunk_patch_files_without_overlap() -> None:
    """When overlap is disabled, the output should match input grouping."""
    entries = [
        {
            "path": "src/a.py",
            "hunks": [_make_hunk(added=[(1, "a")])],
        },
        {
            "path": "src/b.py",
            "hunks": [_make_hunk(added=[(1, "b")])],
        },
    ]

    chunks = review_module._chunk_patch_files(entries, max_files_per_chunk=1, overlap_lines=0)
    assert len(chunks) == 2
    assert [entry["path"] for entry in chunks[0]] == ["src/a.py"]
    assert [entry["path"] for entry in chunks[1]] == ["src/b.py"]


def test_dynamic_limits_default_zero_for_small_inputs() -> None:
    """Default dynamic line limit should be zero when under budget."""
    entries = [{"path": "src/app.py", "hunks": [_make_hunk(added=[(1, "x")])]}]
    limit = review_module._compute_dynamic_line_limit(0, "short-rule", entries)
    assert limit == 0


def test_dynamic_file_limit_defaults_to_file_count() -> None:
    """File limit should equal file count when no threshold hit."""
    entries = [
        {"path": f"src/file_{idx}.py", "hunks": [_make_hunk(added=[(1, "x")])]} for idx in range(3)
    ]
    limit = review_module._compute_dynamic_file_limit(0, "rule", entries)
    assert limit == len(entries)
    configured = review_module._compute_dynamic_file_limit(2, "rule", entries)
    assert configured == 2


def test_estimate_prompt_tokens_handles_empty_segments() -> None:
    """Token estimator should ignore falsy segments."""
    assert review_module._estimate_prompt_tokens("", "abcd") == 1


def test_refine_chunks_for_token_budget_without_split() -> None:
    """Chunks under budget should be returned unchanged."""
    entries = [[{"path": "src/a.py", "hunks": [_make_hunk(added=[(1, "x")])]}]]
    refined = review_module._refine_chunks_for_token_budget(
        entries, rule_text="rule", filter_pattern=None, token_budget=1000
    )
    assert refined == entries


def test_validate_review_response_rejects_non_list_violations() -> None:
    """Validator should reject invalid violations payload."""
    with pytest.raises(click.ClickException):
        review_module.validate_review_response(
            {"violations": "not-a-list"},
            added_lines={},
            parsed_files=set(),
        )


def test_validate_review_response_rejects_invalid_summary_type() -> None:
    """Validator should reject invalid summary objects."""
    with pytest.raises(click.ClickException):
        review_module.validate_review_response(
            {"violations": [], "summary": "bad"},
            added_lines={},
            parsed_files=set(),
        )


def test_validate_review_response_discards_non_mapping_violation() -> None:
    """Non-mapping entries should be discarded and reported."""
    response = {"violations": ["invalid"], "summary": {"total_violations": 1}}
    normalized = review_module.validate_review_response(
        response,
        added_lines={},
        parsed_files=set(),
    )
    assert normalized["summary"]["total_violations"] == 0
    assert normalized["summary"]["discarded_violations"] == 1


def test_run_review_requires_api_key(tmp_path) -> None:
    """run_review should enforce API key configuration."""
    settings = Settings()
    settings.api_key = None
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("# Rule")

    with pytest.raises(click.ClickException):
        review_module.run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_path),
            json_output=False,
            settings=settings,
        )


def test_run_review_rule_read_failure(tmp_path) -> None:
    """Reading a non-file rule path should raise click error."""
    settings = Settings()
    settings.api_key = "key"
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_dir = tmp_path / "rule_dir"
    rule_dir.mkdir()

    with pytest.raises(click.ClickException):
        review_module.run_review(
            ctx,
            patch_file=str(patch_path),
            rule_file=str(rule_dir),
            json_output=False,
            settings=settings,
        )


def test_run_review_handles_schema_error_exception(monkeypatch, tmp_path) -> None:
    """Schema failure should yield fallback summary without raising."""
    settings = Settings()
    settings.api_key = "key"
    settings.workspace_root = tmp_path
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("# Rule")

    parsed_patch = {"files": [{"path": "src/app.py", "hunks": [_make_hunk(added=[(1, "x")])]}]}

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _p: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *a, **k: None)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda _value: "")
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    def raising_execute(*_args, **_kwargs):
        raise click.ClickException(
            "Assistant response did not contain valid JSON matching the required schema"
        )

    monkeypatch.setattr(review_module, "_execute_react_assistant", raising_execute)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_path),
        rule_file=str(rule_path),
        json_output=False,
        settings=settings,
    )

    assert result["summary"]["files_reviewed"] == 1
    assert result["summary"]["total_violations"] == 0


def test_run_review_marks_retry_on_failed_read_tool(monkeypatch, tmp_path) -> None:
    """Read tool failures should trigger retry logic."""
    settings = Settings()
    settings.api_key = "key"
    settings.workspace_root = tmp_path
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("# Rule")

    parsed_patch = {
        "files": [
            {"path": "src/app.py", "hunks": [_make_hunk(added=[(1, "x")])]},
            {"path": "src/other.py", "hunks": [_make_hunk(added=[(2, "y")])]},
        ]
    }

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _p: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *a, **k: None)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda _value: "")
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    class Step:
        def __init__(self, *, tool: str, success: bool):
            self.observation = SimpleNamespace(tool=tool, success=success)

    class RunResult:
        def __init__(self, steps):
            self.steps = steps

    responses = [
        {
            "result": RunResult([Step(tool="read", success=False)]),
            "final_json": {
                "violations": [],
                "summary": {"total_violations": 0, "files_reviewed": 1},
            },
        },
        {
            "result": RunResult([]),
            "final_json": {
                "violations": [
                    {
                        "file": "src/app.py",
                        "line": 1,
                        "message": "Review",
                        "code_snippet": "x",
                    }
                ],
                "summary": {"total_violations": 1, "files_reviewed": 1},
            },
        },
    ]

    def execute_sequence(*_args, **_kwargs):
        payload = responses.pop(0)
        return payload

    monkeypatch.setattr(review_module, "_execute_react_assistant", execute_sequence)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_path),
        rule_file=str(rule_path),
        json_output=True,
        settings=settings,
    )

    assert result["summary"]["total_violations"] == 1
    assert result["summary"]["files_reviewed"] == 2


def test_run_review_discards_invalid_final_json(monkeypatch, tmp_path) -> None:
    """Invalid JSON payloads should be normalized via fallback."""
    settings = Settings()
    settings.api_key = "key"
    settings.workspace_root = tmp_path
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("# Rule")

    parsed_patch = {"files": [{"path": "src/app.py", "hunks": [_make_hunk(added=[(1, "x")])]}]}

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _p: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *a, **k: None)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda _value: "")
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    def execute_invalid(*_args, **_kwargs):
        return {
            "result": {"steps": []},
            "final_json": {"violations": "bad", "summary": {"total_violations": 1}},
        }

    monkeypatch.setattr(review_module, "_execute_react_assistant", execute_invalid)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_path),
        rule_file=str(rule_path),
        json_output=False,
        settings=settings,
    )

    assert result["summary"]["total_violations"] == 0
    assert result["summary"]["files_reviewed"] == 1


def test_run_review_returns_empty_when_all_filtered(monkeypatch, tmp_path) -> None:
    """If all files are filtered out, run_review should return fallback summary."""
    settings = Settings()
    settings.api_key = "key"
    settings.workspace_root = tmp_path
    ctx = click.Context(click.Command("review"), obj={"settings": settings})

    patch_path = tmp_path / "changes.patch"
    patch_path.write_text("diff")
    rule_path = tmp_path / "rule.md"
    rule_path.write_text("# Rule\n\n## Applies To\nregex:^does-not-match$")

    parsed_patch = {"files": [{"path": "src/app.py", "hunks": [_make_hunk(added=[(1, "x")])]}]}

    monkeypatch.setattr(review_module, "parse_patch_file", lambda _p: parsed_patch)
    monkeypatch.setattr(review_module, "_record_invocation", lambda *a, **k: None)
    monkeypatch.setattr(
        review_module,
        "ContextOrchestrator",
        lambda *args, **kwargs: SimpleNamespace(build_section=lambda *a, **k: ""),
    )
    monkeypatch.setattr(review_module, "resolve_prompt_input", lambda _value: "")
    monkeypatch.setattr(
        review_module,
        "get_llm_client",
        lambda _ctx: SimpleNamespace(
            complete=lambda *a, **k: None, invoke_tools=lambda *a, **k: None
        ),
    )

    def execute_noop(*_args, **_kwargs):
        return {"result": {"steps": []}, "final_json": {"violations": [], "summary": {}}}

    monkeypatch.setattr(review_module, "_execute_react_assistant", execute_noop)

    result = review_module.run_review(
        ctx,
        patch_file=str(patch_path),
        rule_file=str(rule_path),
        json_output=False,
        settings=settings,
    )

    assert result["summary"]["files_reviewed"] == 0
    assert result["violations"] == []
