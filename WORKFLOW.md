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
profiles:
  default:
    bin: "devagent"
    provider: chatgpt
    model: gpt-5.4
    approval_mode: auto-edit
  reviewer:
    bin: "devagent"
    provider: chatgpt
    model: gpt-5.4
    approval_mode: auto-edit
  repair:
    bin: "devagent"
    provider: chatgpt
    model: gpt-5.4
    approval_mode: auto-edit
roles:
  triage: default
  plan: default
  implement: default
  verify: default
  review: reviewer
  repair: repair
  gate: default
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

## How work enters this repo

Most changes land through the validated Hub -> Runner -> DevAgent path or through direct
contributor work on the executor, CLI, providers, and tools.

## Supported vs experimental

- Supported: `devagent` as the first-party executor through `devagent execute --request --artifact-dir`
- Experimental: any alternative executor story outside the validated DevAgent path

The workflow frontmatter above remains a hub/runner compatibility contract. For the public interactive CLI surface, DevAgent now defaults to autopilot; use `devagent --mode default` to opt into guarded prompts, or `devagent --mode autopilot` to set it explicitly, rather than legacy approval-mode flags.

## Contributor completion bar

Before merge, contributors should run:

```bash
bun run lint
bun run typecheck
bun run test
bun run check:oss
```
