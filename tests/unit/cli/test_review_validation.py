from __future__ import annotations

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
    violation = normalized["violations"][0]
    assert violation["file"] == "src/new_file.ts"
    assert violation["line"] == 10
    assert violation["change_type"] == "added"
    assert violation["severity"] == "warning"


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
    normalized = validate_review_response(
        response,
        added_lines=added_lines,
        parsed_files=parsed_files,
    )

    assert normalized["violations"] == []
    summary = normalized["summary"]
    assert summary["total_violations"] == 0
    assert summary.get("discarded_violations") == 1


def test_validate_review_response_rejects_wrong_line_numbers() -> None:
    response = {
        "violations": [
            {
                "file": "src/new_file.ts",
                "line": 99,
                "message": "Bad ref",
                "code_snippet": "export const Foo = 1;",
            }
        ],
        "summary": {"total_violations": 1},
    }

    added_lines = {"src/new_file.ts": {10: "export const Foo = 1;"}}
    parsed_files = {"src/new_file.ts"}

    normalized = validate_review_response(
        response,
        added_lines=added_lines,
        parsed_files=parsed_files,
    )

    assert normalized["violations"] == []
    summary = normalized["summary"]
    assert summary["total_violations"] == 0
    assert summary.get("discarded_violations") == 1


def test_validate_review_response_discards_removed_line_violations() -> None:
    """Violations on removed lines should be discarded (we only review added code)."""
    response = {
        "violations": [
            {
                "file": "src/new_file.ts",
                "line": 20,
                "change_type": "removed",
                "message": "Removed validation call",
                "code_snippet": "validate(input);",
            }
        ],
        "summary": {"total_violations": 1},
    }

    added_lines = {"src/new_file.ts": {10: "export const Foo = 1;"}}
    removed_lines = {"src/new_file.ts": {20: "validate(input);"}}
    parsed_files = {"src/new_file.ts"}

    normalized = validate_review_response(
        response,
        added_lines=added_lines,
        removed_lines=removed_lines,
        parsed_files=parsed_files,
    )

    # Removed violations should be discarded
    assert normalized["violations"] == []
    assert normalized["summary"]["total_violations"] == 0
    assert normalized["summary"].get("discarded_violations") == 1


def test_validate_review_response_allows_matching_violation() -> None:
    response = {
        "violations": [
            {
                "file": "src/new_file.ts",
                "line": 10,
                "message": "Valid violation",
                "code_snippet": "export const Foo = 1;",
            }
        ],
        "summary": {"total_violations": 10, "files_reviewed": 42},
    }

    added_lines = {"src/new_file.ts": {10: "export const Foo = 1;"}}
    parsed_files = {"src/new_file.ts"}

    normalized = validate_review_response(
        response,
        added_lines=added_lines,
        parsed_files=parsed_files,
    )

    violation = normalized["violations"][0]
    assert violation["file"] == "src/new_file.ts"
    assert violation["change_type"] == "added"
    assert violation["severity"] == "warning"
    summary = normalized["summary"]
    assert summary["total_violations"] == 1
    assert summary["files_reviewed"] == 1
