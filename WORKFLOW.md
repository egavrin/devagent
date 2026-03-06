---
version: 1
tracker:
  kind: github
  issue_labels_include: ["da:ready"]
dispatch:
  max_concurrency: 1
workspace:
  mode: worktree
  root: ".devagent/workspaces"
runner:
  approval_mode: auto-edit
  max_iterations: 60
roles:
  triage: architect
  plan: architect
  implement: general
  review: reviewer
verify:
  commands:
    - bun run lint
    - bun run typecheck
    - bun run test
pr:
  draft: true
  open_requires:
    - verify_passed
    - no_blocking_self_review_findings
repair:
  max_rounds: 2
handoff:
  when:
    - draft_pr_opened
    - ci_green
    - no_blocking_auto_review_findings
---

# Workflow

Triage the issue, extract acceptance criteria, create a concrete plan, implement
in an isolated worktree, run the verify commands, open a draft PR, perform
automated review/fix loops, and hand off to a human when the PR is near-merge
quality or when the workflow encounters conflict or ambiguity.
