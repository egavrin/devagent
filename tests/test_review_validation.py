from __future__ import annotations

import click
import pytest

from ai_dev_agent.cli.review import validate_review_response


def test_validate_review_response_normalizes_summary_counts() -> None:
    response = {
        "violations": [
            {
                "file": "src/new_file.ts",
                "line": 10,
                "message": "Example violation",
                "code_snippet": "export const Foo = 1;",
            }
        ],
        "summary": {
            "total_violations": 42,
            "files_reviewed": 99,
            "rule_name": "ETS001_DOC_PUBLIC_API",
        },
    }

    added_lines = {"src/new_file.ts": {10: "export const Foo = 1;"}}
    parsed_files = {"src/new_file.ts", "src/other_file.ts"}

    normalized = validate_review_response(
        response,
        added_lines=added_lines,
        parsed_files=parsed_files,
    )

    assert normalized["summary"]["total_violations"] == 1
    assert normalized["summary"]["files_reviewed"] == len(parsed_files)
    assert normalized["violations"] == response["violations"]


def test_validate_review_response_rejects_unknown_paths() -> None:
    response = {
        "violations": [
            {
                "file": "src/missing.ts",
                "line": 5,
                "message": "Bad ref",
                "code_snippet": "export const Bar = 2;",
            }
        ],
        "summary": {"total_violations": 1},
    }

    added_lines = {"src/new_file.ts": {10: "export const Foo = 1;"}}
    parsed_files = {"src/new_file.ts"}

    with pytest.raises(click.ClickException):
        validate_review_response(
            response,
            added_lines=added_lines,
            parsed_files=parsed_files,
        )
