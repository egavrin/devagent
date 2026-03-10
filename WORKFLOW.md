---
version: 1
tracker:
  kind: github
  issue_labels_include: ["da:ready", "devagent"]
dispatch:
  max_concurrency: 1
workspace:
  mode: worktree
  root: ".devagent/workspaces"
runner:
  bin: "devagent"
  provider: chatgpt
  model: gpt-5.4
  approval_mode: auto-edit
  max_iterations: 60
roles:
  triage: architect
  plan: architect
  implement: general
  review: reviewer
verify:
  commands:
    - "bun run lint"
    - "bun run typecheck"
    - "bun run test"
pr:
  draft: true
  open_requires: [verify_passed, no_blocking_self_review_findings]
repair:
  max_rounds: 2
handoff:
  when: [draft_pr_opened, ci_green, no_blocking_auto_review_findings]
---

# DevAgent Workflow

This file configures how devagent-hub orchestrates work on this repository.

The intended path is: triage, plan, implement in an isolated worktree, verify,
open a draft PR, run review and repair loops, then hand off once CI is green or
a human decision is required.
