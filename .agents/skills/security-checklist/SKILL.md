---
name: security-checklist
description: Review DevAgent changes for unsafe command execution, secret handling, and hidden failure modes.
---

# Security Checklist

Review changes for:

## Command safety

- No shell injection via user-controlled strings.
- Prefer structured arguments over shell concatenation.
- File paths, branch names, and prompts are treated as data.

## Secret handling

- API keys and credentials come from auth flows or environment variables, never the repo.
- Logs, artifacts, and prompts do not leak secrets.

## Fail-fast behavior

- Do not hide security-relevant failures behind silent fallbacks.
- Unsupported capability combinations should fail loudly and early.
