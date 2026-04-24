# Autonomous Repository Developer Plan

## Goal

DevAgent should become a workflow-native autonomous developer for repositories. It should turn issues, failures, and requests into designs, code, reviews, repairs, and pull requests while following project-defined process.

This is not a replacement for the current executor contract. It is the product direction above it.

Current supported core:

```bash
devagent execute --request request.json --artifact-dir artifacts/
```

Future autonomous repository mode should compose that fixed typed executor contract with repository integrations, workflow policy, approval policy, reviewable artifacts, and human or agent collaboration.

## Product Shape

DevAgent should keep two clear surfaces:

1. Operator mode

- Interactive terminal UI and direct prompt usage
- Humans steer, inspect, interrupt, and override
- Useful for local investigation, implementation, and review

2. Autonomous repository mode

- Connected to repositories, issues, pull requests, CI, and task systems
- Follows project workflow and autonomy policy
- Creates branches, artifacts, reviews, comments, and pull requests
- Responds to human and agent feedback
- Hands off reviewable work rather than silently merging it

Autonomous repository mode should become the primary long-term differentiator. Operator mode remains the direct human control surface.

## Product Principle

The repository should not merely be context. It should define the workflow contract.

A repository should be able to define:

- Workflow composition
- Approval policy
- Autonomy policy
- Review rules
- Required artifacts
- Verification expectations
- Branch and pull request policy
- Allowed tools, networks, directories, and credentials
- Escalation and handoff rules

The target product is not "an agent in a repo." It is an autonomous developer executing a repo-defined workflow contract.

## Execution Model

Keep the built-in stage primitives fixed and typed:

- `task-intake`
- `triage`
- `design`
- `breakdown`
- `issue-generation`
- `plan`
- `test-plan`
- `implement`
- `verify`
- `review`
- `repair`
- `completion`

Repositories should eventually define workflow compositions over those primitives rather than defining arbitrary prompt stages.

Good direction:

- Fixed stage semantics
- Repo-defined stage order
- Repo-defined gates
- Repo-defined required artifacts
- Repo-defined verification commands
- Repo-defined review and repair policy

Avoid:

- Arbitrary prompt-defined workflow stages
- Unvalidated stage outputs
- Silent fallbacks when workflow policy is missing or contradictory
- Merging autonomous work without explicit policy and review gates

## First-Class Concepts

To scale from one task to continuous repository work, these concepts should become explicit:

- Repository
- Work item
- Workflow
- Stage
- Artifact
- Branch
- Worktree
- Reviewable
- Verification run
- Approval policy
- Autonomy policy
- Tool policy
- Delegated agent role
- Session state
- Repository memory
- Human or agent participant

Work items should include at least:

- Issue
- Pull request comment
- Review finding
- CI failure
- Scheduled maintenance task
- Human request
- Agent request
- Optimization or improvement proposal

## Core Loops

### Intake Loop

Inputs:

- Issues
- Pull request comments
- CI failures
- Roadmap tasks
- Scheduled jobs
- Human requests
- Other-agent requests

Outputs:

- Classified work item
- Selected workflow
- Required artifacts
- Required approvals
- Initial risk and scope assessment

### Design Loop

For meaningful changes, DevAgent should:

- Read project workflow and rules
- Inspect relevant code
- Create a design artifact
- Describe options, risks, and validation strategy
- Request approval when policy requires it

### Implementation Loop

DevAgent should:

- Create an isolated branch or worktree
- Implement bounded slices
- Run verification
- Repair failures
- Produce an implementation summary
- Preserve artifact and event history

### Pull Request Loop

DevAgent should:

- Open or update pull requests
- Attach or link artifacts
- Summarize changes and risks
- Explain validation results
- Respond to review comments
- Push follow-up commits when requested or allowed

### Review Loop

DevAgent should:

- Review its own changes before handoff
- Review human or agent pull requests
- Detect defects, missing tests, regressions, and scope drift
- Separate blocking findings from suggestions
- Produce review artifacts that can drive repair stages

### Improvement Loop

DevAgent should eventually propose proactive work:

- Flaky test investigation
- Performance improvements
- Risky hotspot cleanup
- Dependency update preparation
- Duplicated-code reduction
- Stale abstraction removal
- Missing test coverage
- Documentation drift

These should begin as proposals or draft artifacts, not automatic large rewrites.

## Phased Roadmap

### Phase 1: Repository-Connected Executor

Goal: turn the existing executor into repo-native automation.

Capabilities:

- Ingest issues, pull request comments, CI failures, and manual work items
- Map each work item into a typed `execute` request
- Create or reuse branches and worktrees
- Write stage artifacts to an artifact store
- Publish status and artifact links
- Open draft pull requests when policy allows
- Comment on issues and pull requests with summaries and blockers

User outcomes:

- "Fix this issue"
- "Investigate this CI failure"
- "Implement this approved design"
- "Open a pull request with verification results"

### Phase 2: Repo-Defined Workflow Spec

Goal: let projects define how DevAgent should work without redefining built-in stages.

The workflow spec should define:

- Supported work item types
- Stage sequences
- Required artifacts
- Approval gates
- Required reviewers
- Verification commands
- Branch naming
- Pull request policy
- Review and repair limits
- Autonomy levels
- Tool and network restrictions

This is where DevAgent becomes workflow-native rather than generic.

### Phase 3: Pull Request and Review Collaboration

Goal: make DevAgent participate in the normal development process.

Capabilities:

- Open and update pull requests
- Respond to review threads
- Request clarification with scoped questions
- Re-run verification after feedback
- Produce repair summaries
- Hand work off when blocked
- Interoperate with other agents through comments, artifacts, and status checks

Interaction surfaces:

- Issue comments
- Pull request comments
- Review comments
- Status checks
- Artifact links
- Structured design and review documents

### Phase 4: Role-Separated Multi-Agent Execution

Goal: improve reliability by making role boundaries explicit and auditable.

Roles:

- Triage or explorer
- Designer or architect
- Implementer
- Reviewer
- Repair agent
- Optimizer

Important constraints:

- Reviewer should remain meaningfully independent
- Delegation should be visible in events and artifacts
- Agents should have scoped permissions
- Parallel work should be bounded by file ownership and workflow policy

### Phase 5: Continuous Autonomous Improvement

Goal: move from reactive task handling to proactive value.

Capabilities:

- Scheduled audits
- Flaky test detection
- Performance and reliability recommendations
- Dependency update plans
- Stale code and abstraction reports
- Test gap reports
- Documentation drift reports

Default behavior should be proposal-first. Mutating work should require workflow policy that explicitly allows it.

## Policy Model

Autonomy should be policy-based, not prompt-only.

Policy should answer:

- What can be changed automatically?
- Which stages require human approval?
- Can DevAgent open a pull request?
- Can DevAgent push follow-up commits?
- Can DevAgent merge?
- Can DevAgent deploy?
- Is network access allowed?
- Which tools are allowed?
- Which directories are restricted?
- Which credentials are available?
- What verification is required before handoff?

Autonomy levels should be explicit. A possible progression:

- `observe`: read, inspect, and report only
- `propose`: create designs, plans, and review artifacts
- `draft`: create branches and draft pull requests
- `repair`: respond to review or CI feedback within scope
- `maintain`: run scheduled improvement jobs within policy

Merging and deployment should stay separate from ordinary implementation autonomy and require explicit policy.

## Artifact Model

Artifacts are how autonomous work stays reviewable.

Important artifact types:

- Task specification
- Triage report
- Design document
- Breakdown document
- Issue specification
- Test plan
- Implementation summary
- Verification report
- Review report
- Repair summary
- Workflow summary

Artifacts should be linked from pull requests, comments, and status checks where possible.

## Safety and Trust

Trust should come from clear boundaries, not from optimistic prompt wording.

Safety requirements:

- Fail fast on unsupported constraints
- Keep stage semantics typed and test-backed
- Keep readonly stages readonly
- Keep mutating stages auditable
- Separate implementation and review roles
- Preserve event logs and artifacts
- Make approvals explicit
- Make policy conflicts explicit
- Prefer draft pull requests and reviewable handoff over direct merge

## Non-Goals

Do not build:

- A broad personal assistant surface
- Arbitrary prompt-defined workflows as the primary model
- A sprawling command surface that hides the executor contract
- Autonomous merge or deploy behavior without explicit policy
- A system where repository rules are only loose context
- A product story centered on agent personality rather than workflow compliance

## Near-Term Build Order

1. Repository-connected execution
2. Repo-defined workflow spec over fixed stages
3. Pull request, comment, and review integration
4. Role-separated multi-agent execution
5. Proactive optimization and improvement jobs

This path preserves DevAgent's current typed-contract advantage while moving toward the autonomous developer vision.
