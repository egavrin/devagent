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

The public machine-facing story for this repository is the fixed staged
`devagent execute --request --artifact-dir` workflow. The lead example flow is:

`design -> breakdown -> issue-generation -> implement -> review -> repair`

Other supported stages such as `task-intake`, `test-plan`, `triage`, `plan`,
`verify`, and `completion` are still part of the supported contract, but the
executor stage set itself is fixed rather than user-defined.

## How work enters this repo

Most changes land through the validated Hub -> Runner -> DevAgent path or through direct
contributor work on the executor, CLI, providers, and tools.

## Supported vs experimental

- Supported: `devagent` as the first-party executor through `devagent execute --request --artifact-dir`
- Experimental: any alternative executor story outside the validated DevAgent path

The workflow frontmatter above remains a hub/runner compatibility contract.
Public stage semantics are code-defined by `taskType`, with dynamic request
context layered on top at runtime. This repository does not treat stage prompts
or stage definitions as user-configurable public surfaces.

For the public interactive CLI surface, DevAgent now defaults to autopilot; use
`devagent --mode default` to opt into guarded prompts, or `devagent --mode autopilot`
to set it explicitly, rather than legacy approval-mode flags.

## Contributor completion bar

Before merge, contributors should run:

```bash
bun run lint
bun run typecheck
bun run test
bun run test:surface-smoke
bun run check:oss
```

When the task changes the public CLI, publish bundle, or `devagent execute` contract, also run the broader validation tier:

```bash
bun run validate:live:provider-smoke
bun run validate:live:tui
bun run verify:publish
```

Treat `bun run typecheck`, `bun run test`, `bun run test:surface-smoke`, and `bun run check:oss` as the fast PR gate.

Use the broader validation tier to check provider credentials, TUI behavior, and publish readiness when those surfaces are in scope.
