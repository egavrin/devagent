---
version: 1
tracker:
  kind: github
  issue_labels_include: [devagent]
dispatch:
  max_concurrency: 4
workspace:
  mode: worktree
  root: "."
runner:
  bin: "devagent"
  provider: chatgpt
  model: gpt-5.4
  approval_mode: full-auto
  max_iterations: 50
verify:
  commands:
    - "bun run test"
    - "bun run typecheck"
pr:
  draft: true
  open_requires: [verify]
repair:
  max_rounds: 3
handoff:
  when: [repair_failed, review_rejected]
---

# DevAgent Workflow

This file configures how devagent-hub orchestrates work on this repository.
