"""Tests for agents schemas module."""

import json

from ai_dev_agent.agents.schemas import VIOLATION_SCHEMA


def test_violation_schema_structure():
    """Test the structure of the violation schema."""
    assert VIOLATION_SCHEMA["type"] == "object"
    assert "violations" in VIOLATION_SCHEMA["properties"]
    assert "summary" in VIOLATION_SCHEMA["properties"]
    assert "violations" in VIOLATION_SCHEMA["required"]


def test_violation_schema_violations_array():
    """Test the violations array structure."""
    violations = VIOLATION_SCHEMA["properties"]["violations"]

    assert violations["type"] == "array"
    assert "items" in violations

    item_schema = violations["items"]
    assert item_schema["type"] == "object"

    # Check required fields
    assert "file" in item_schema["properties"]
    assert "line" in item_schema["properties"]
    assert "message" in item_schema["properties"]

    # Check required list
    assert "file" in item_schema["required"]
    assert "line" in item_schema["required"]
    assert "message" in item_schema["required"]


def test_violation_schema_severity_enum():
    """Test severity enum values."""
    item_props = VIOLATION_SCHEMA["properties"]["violations"]["items"]["properties"]
    severity = item_props["severity"]

    assert severity["type"] == "string"
    assert severity["enum"] == ["error", "warning", "info"]


def test_violation_schema_summary():
    """Test summary object structure."""
    summary = VIOLATION_SCHEMA["properties"]["summary"]

    assert summary["type"] == "object"
    assert "total_violations" in summary["properties"]
    assert "files_reviewed" in summary["properties"]
    assert "rule_name" in summary["properties"]

    # Check types
    assert summary["properties"]["total_violations"]["type"] == "integer"
    assert summary["properties"]["files_reviewed"]["type"] == "integer"
    assert summary["properties"]["rule_name"]["type"] == "string"


def test_violation_schema_valid_json():
    """Test that the schema is valid JSON-serializable."""
    # This should not raise an exception
    json_str = json.dumps(VIOLATION_SCHEMA)

    # Should be able to parse it back
    parsed = json.loads(json_str)
    assert parsed == VIOLATION_SCHEMA


def test_violation_schema_descriptions():
    """Test that descriptions are present for documentation."""
    item_props = VIOLATION_SCHEMA["properties"]["violations"]["items"]["properties"]

    # Check that descriptions exist
    assert "description" in item_props["file"]
    assert "description" in item_props["line"]
    assert "description" in item_props["severity"]
    assert "description" in item_props["rule"]
    assert "description" in item_props["message"]
    assert "description" in item_props["code_snippet"]
