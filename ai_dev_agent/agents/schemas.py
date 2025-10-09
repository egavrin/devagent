"""JSON schemas for agent output formats."""

# Schema for code review violations
VIOLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "File path where violation was found"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number of the violation"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["error", "warning", "info"],
                        "description": "Severity level of the violation"
                    },
                    "rule": {
                        "type": "string",
                        "description": "Name or ID of the violated rule"
                    },
                    "message": {
                        "type": "string",
                        "description": "Explanation of the violation"
                    },
                    "code_snippet": {
                        "type": "string",
                        "description": "The problematic code (optional)"
                    }
                },
                "required": ["file", "line", "message"]
            }
        },
        "summary": {
            "type": "object",
            "properties": {
                "total_violations": {
                    "type": "integer",
                    "description": "Total number of violations found"
                },
                "files_reviewed": {
                    "type": "integer",
                    "description": "Number of files reviewed"
                },
                "rule_name": {
                    "type": "string",
                    "description": "Name of the rule that was applied"
                }
            }
        }
    },
    "required": ["violations"]
}
