/**
 * Internal headless workflow runner retained as compatibility machinery for
 * `devagent execute`. This is not the supported public orchestration contract.
 *
 * The supported machine entrypoint is:
 *   devagent execute --request <request.json> --artifact-dir <path>
 *
 * Exit codes:
 *   0 = success (output JSON written)
 *   1 = phase failed (partial output may be written)
 *   2 = invalid arguments
 */

import { readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  type WorkflowPhase,
  type WorkflowRunArgs,
  type WorkflowApprovalMode,
  type ReasoningLevel,
  type RunnerDescription,
  isValidPhase,
  isValidApprovalMode,
  isValidReasoningLevel,
  EXIT_CODE,
  WORKFLOW_PHASES,
  WORKFLOW_APPROVAL_MODES,
  REASONING_LEVELS,
} from "@devagent/core/workflow-contract";

// ─── Runner Description ──────────────────────────────────────

export function printRunnerDescription(): void {
  const pkg = { version: "0.1.0" }; // TODO: read from package.json
  const description: RunnerDescription = {
    version: pkg.version,
    supportedPhases: [...WORKFLOW_PHASES],
    availableProviders: ["anthropic", "openai", "ollama", "chatgpt", "github-copilot"],
    supportedApprovalModes: [...WORKFLOW_APPROVAL_MODES],
    supportedReasoningLevels: [...REASONING_LEVELS],
  };
  process.stdout.write(JSON.stringify(description, null, 2) + "\n");
}

// ─── Arg Parsing ─────────────────────────────────────────────

interface RawWorkflowArgs {
  subcommand: string | null;
  phase: string | null;
  input: string | null;
  output: string | null;
  events: string | null;
  repo: string | null;
  provider: string | null;
  model: string | null;
  maxIterations: number | null;
  approvalMode: string | null;
  reasoning: string | null;
}

export function parseWorkflowArgs(argv: string[]): RawWorkflowArgs {
  const args = argv.slice(2); // skip bun + script
  const result: RawWorkflowArgs = {
    subcommand: null,
    phase: null,
    input: null,
    output: null,
    events: null,
    repo: null,
    provider: null,
    model: null,
    maxIterations: null,
    approvalMode: null,
    reasoning: null,
  };

  // Expect: workflow run --phase ... etc
  if (args[0] !== "workflow") return result;
  result.subcommand = args[1] ?? null;

  for (let i = 2; i < args.length; i++) {
    const arg = args[i]!;
    switch (arg) {
      case "--phase":
        result.phase = args[++i] ?? null;
        break;
      case "--input":
        result.input = args[++i] ?? null;
        break;
      case "--output":
        result.output = args[++i] ?? null;
        break;
      case "--events":
        result.events = args[++i] ?? null;
        break;
      case "--repo":
        result.repo = args[++i] ?? null;
        break;
      case "--provider":
        result.provider = args[++i] ?? null;
        break;
      case "--model":
        result.model = args[++i] ?? null;
        break;
      case "--max-iterations": {
        const val = parseInt(args[++i] ?? "", 10);
        result.maxIterations = isNaN(val) ? null : val;
        break;
      }
      case "--approval-mode":
        result.approvalMode = args[++i] ?? null;
        break;
      case "--suggest":
        result.approvalMode = "suggest";
        break;
      case "--auto-edit":
        result.approvalMode = "auto-edit";
        break;
      case "--full-auto":
        result.approvalMode = "full-auto";
        break;
      case "--reasoning":
        result.reasoning = args[++i] ?? null;
        break;
    }
  }

  return result;
}

function die(msg: string): never {
  process.stderr.write(`[devagent workflow] Error: ${msg}\n`);
  process.exit(EXIT_CODE.INVALID_ARGS);
}

export function validateWorkflowArgs(raw: RawWorkflowArgs): WorkflowRunArgs {
  if (raw.subcommand !== "run") {
    die(`Unknown workflow subcommand "${raw.subcommand}". Use: devagent workflow run`);
  }
  if (!raw.phase) {
    die(`--phase is required. Valid phases: ${WORKFLOW_PHASES.join(", ")}`);
  }
  if (!isValidPhase(raw.phase)) {
    die(`Invalid phase "${raw.phase}". Valid phases: ${WORKFLOW_PHASES.join(", ")}`);
  }
  if (!raw.input) die("--input <path> is required");
  if (!raw.output) die("--output <path> is required");
  if (!raw.events) die("--events <path> is required");
  if (!raw.repo) die("--repo <path> is required");

  if (raw.approvalMode && !isValidApprovalMode(raw.approvalMode)) {
    die(`Invalid approval mode "${raw.approvalMode}". Valid modes: ${WORKFLOW_APPROVAL_MODES.join(", ")}`);
  }
  if (raw.reasoning && !isValidReasoningLevel(raw.reasoning)) {
    die(`Invalid reasoning level "${raw.reasoning}". Valid levels: ${REASONING_LEVELS.join(", ")}`);
  }

  return {
    phase: raw.phase as WorkflowPhase,
    inputPath: resolve(raw.input),
    outputPath: resolve(raw.output),
    eventsPath: resolve(raw.events),
    repoPath: resolve(raw.repo),
    provider: raw.provider ?? undefined,
    model: raw.model ?? undefined,
    maxIterations: raw.maxIterations ?? undefined,
    approvalMode: raw.approvalMode as WorkflowApprovalMode | undefined,
    reasoning: raw.reasoning as ReasoningLevel | undefined,
  };
}

// ─── Event Logger ────────────────────────────────────────────

function logEvent(eventsPath: string, event: Record<string, unknown>): void {
  const line = JSON.stringify({ ...event, timestamp: new Date().toISOString() }) + "\n";
  appendFileSync(eventsPath, line);
}

// ─── Input Validation ────────────────────────────────────────

export function validateInput(phase: WorkflowPhase, input: Record<string, unknown>): void {
  const missing = (field: string) => input[field] === undefined || input[field] === null;

  switch (phase) {
    case "triage":
    case "plan":
      if (missing("issueNumber")) throw new Error(`${phase} input requires "issueNumber"`);
      if (missing("title")) throw new Error(`${phase} input requires "title"`);
      break;
    case "implement":
      if (missing("issueNumber")) throw new Error('implement input requires "issueNumber"');
      if (missing("acceptedPlan")) throw new Error('implement input requires "acceptedPlan"');
      break;
    case "verify":
      if (!Array.isArray(input.commands)) throw new Error('verify input requires "commands" array');
      break;
    case "review":
      if (missing("issueNumber")) throw new Error('review input requires "issueNumber"');
      break;
    case "repair":
      if (missing("issueNumber")) throw new Error('repair input requires "issueNumber"');
      if (missing("round")) throw new Error('repair input requires "round"');
      break;
    case "gate":
      if (missing("sourcePhase")) throw new Error('gate input requires "sourcePhase"');
      if (missing("issueNumber")) throw new Error('gate input requires "issueNumber"');
      if (missing("stageOutput")) throw new Error('gate input requires "stageOutput"');
      break;
  }
}

// ─── Gate Prompt Builder ─────────────────────────────────────

const GATE_CRITERIA: Record<string, string> = {
  triage: `Evaluate the triage report quality:
1. Does the summary clearly describe the issue scope?
2. Is the complexity assessment reasonable?
3. Are related files identified? (check the codebase if needed)
4. Are blockers and prerequisites noted?
5. Is the information sufficient to create an implementation plan?`,

  plan: `Evaluate the implementation plan quality:
1. Are the steps specific and actionable (not vague)?
2. Are all files to modify/create identified?
3. Is there a concrete test strategy?
4. Are risks and edge cases considered?
5. Does the plan address the full scope of the issue?
6. Is the plan achievable without architectural overhaul?`,

  implement: `Evaluate the implementation output quality:
1. Were the planned changes actually made?
2. Are all changed files listed?
3. Does the commit message accurately describe the changes?
4. Is the diff summary coherent?
5. Check the actual code changes in the repository for correctness.`,
};

function buildGatePrompt(input: Record<string, unknown>): string {
  const sourcePhase = input.sourcePhase as string;
  const criteria = GATE_CRITERIA[sourcePhase] ?? "Evaluate the output quality and completeness.";
  const stageOutput = input.stageOutput as Record<string, unknown>;

  return `You are a quality gate reviewer evaluating the output of the "${sourcePhase}" phase in an automated development workflow.

## Context
- **Issue**: #${input.issueNumber}
- **Source Phase**: ${sourcePhase}

## Stage Output
${JSON.stringify(stageOutput, null, 2)}

## Evaluation Criteria
${criteria}

## Task
Evaluate the stage output against the criteria above. Be strict but fair — the goal is to catch genuinely inadequate outputs that would cause problems in later stages, not to nitpick minor issues.

If you need to verify claims in the output (e.g., that files exist or code patterns match), use the available tools to check.

## Output Format
Respond with a JSON object (and nothing else):
\`\`\`json
{
  "summary": "Brief evaluation summary",
  "verdict": "pass|block",
  "findings": [
    {"file": "", "severity": "major", "message": "Description of issue", "category": "quality"}
  ],
  "blockingCount": 0,
  "confidence": 0.85
}
\`\`\`

Use verdict "pass" if the output is good enough to proceed. Use "block" if it has serious gaps that would cause downstream failures.`;
}

// ─── Phase Prompt Builder ────────────────────────────────────

function buildPhasePrompt(phase: WorkflowPhase, input: Record<string, unknown>): string {
  switch (phase) {
    case "triage":
      return `You are triaging a GitHub issue for a development workflow.

## Issue
- **Number**: #${input.issueNumber}
- **Title**: ${input.title}
- **Author**: ${input.author}
- **Labels**: ${(input.labels as string[])?.join(", ") || "none"}

### Body
${input.body}

## Task
Analyze this issue and produce a triage report. Determine:
1. Complexity: trivial, small, medium, large, or epic
2. Suggested labels for categorization
3. Any blockers or prerequisites
4. Related files that will likely need changes

Explore the codebase to understand the scope. Use find_files and search_files to identify affected areas.

## Output Format
Respond with a JSON object (and nothing else) with this structure:
\`\`\`json
{
  "summary": "Brief triage summary",
  "complexity": "small|medium|large",
  "suggestedLabels": ["label1"],
  "blockers": [],
  "relatedFiles": ["path/to/file.ts"]
}
\`\`\``;

    case "plan":
      return `You are creating an implementation plan for a GitHub issue.

## Issue
- **Number**: #${input.issueNumber}
- **Title**: ${input.title}
- **Author**: ${input.author}

### Body
${input.body}

${input.triageReport ? `## Triage Report\n${JSON.stringify(input.triageReport, null, 2)}` : ""}

## Task
Create a detailed implementation plan. Explore the codebase thoroughly first, then:
1. List specific steps with files to create or modify
2. Define a testing strategy
3. Identify risks and edge cases

## Output Format
Respond with a JSON object (and nothing else) with this structure:
\`\`\`json
{
  "summary": "Plan summary",
  "steps": [{"description": "...", "file": "path/to/file.ts", "type": "modify"}],
  "filesToCreate": [],
  "filesToModify": ["path/to/file.ts"],
  "testStrategy": "How to test",
  "risks": ["Risk description"]
}
\`\`\``;

    case "implement":
      return `You are implementing changes for a GitHub issue according to an accepted plan.

## Issue
- **Number**: #${input.issueNumber}
- **Title**: ${input.title}

### Body
${input.body}

## Accepted Plan
${JSON.stringify(input.acceptedPlan, null, 2)}

## Task
Implement the changes described in the plan. Follow existing code patterns and conventions.
- Create and modify files as specified in the plan
- Write clean, well-tested code
- Follow the project's coding standards

After making all changes, produce a summary of what was done.

## Output Format
When done, respond with a JSON object (and nothing else):
\`\`\`json
{
  "summary": "Implementation summary",
  "changedFiles": ["path/to/file.ts"],
  "suggestedCommitMessage": "feat: description of changes",
  "diffSummary": "Brief description of the diff"
}
\`\`\``;

    case "verify":
      return `You are verifying that recent code changes are correct.

## Verification Commands
${(Array.isArray(input.commands) ? input.commands as string[] : []).map((c) => `- \`${c}\``).join("\n")}

## Task
Run each verification command and report the results. If any command fails, report the failure details.

## Output Format
Respond with a JSON object (and nothing else):
\`\`\`json
{
  "summary": "Verification summary",
  "passed": true,
  "results": [
    {"command": "bun run test", "exitCode": 0, "stdout": "...", "stderr": "", "passed": true}
  ]
}
\`\`\``;

    case "review":
      return `You are reviewing code changes for a GitHub issue.

## Context
- **Issue**: #${input.issueNumber}
${input.prNumber ? `- **PR**: #${input.prNumber}` : ""}
${input.branch ? `- **Branch**: ${input.branch}` : ""}

${input.diff ? `## Diff\n\`\`\`diff\n${input.diff}\n\`\`\`` : ""}
${input.reviewComments ? `## Existing Review Comments\n${JSON.stringify(input.reviewComments, null, 2)}` : ""}
${input.ciChecks ? `## CI Checks\n${JSON.stringify(input.ciChecks, null, 2)}` : ""}

## Task
Review all changes. Check for:
1. Correctness and completeness
2. Code quality and style
3. Security issues
4. Test coverage
5. Edge cases

Use git_diff to see the full diff if not provided above.

## Output Format
Respond with a JSON object (and nothing else):
\`\`\`json
{
  "summary": "Review summary",
  "verdict": "pass|block",
  "findings": [
    {"file": "path.ts", "line": 42, "severity": "major", "message": "...", "category": "bug"}
  ],
  "blockingCount": 0
}
\`\`\``;

    case "repair":
      return `You are repairing code based on review findings.

## Context
- **Issue**: #${input.issueNumber}
- **Repair Round**: ${input.round}

## Findings to Fix
${JSON.stringify(input.findings, null, 2)}

${Array.isArray(input.ciFailures) ? `## CI Failures\n${(input.ciFailures as string[]).map((f) => `- ${f}`).join("\n")}` : ""}

## Task
Fix each finding. After fixing, run verification to confirm the fixes work.

## Output Format
Respond with a JSON object (and nothing else):
\`\`\`json
{
  "summary": "Repair summary",
  "fixedFindings": ["description of fix"],
  "remainingFindings": 0,
  "verificationPassed": true,
  "changedFiles": ["path/to/file.ts"]
}
\`\`\``;

    case "gate":
      return buildGatePrompt(input);
  }
}

// ─── Output Extraction ──────────────────────────────────────

/**
 * Extract a JSON object from the LLM's text response.
 * Tries to find a JSON code block first, then falls back to parsing the whole string.
 */
function extractJsonOutput(text: string): Record<string, unknown> | null {
  // Try JSON code block first
  const codeBlockMatch = text.match(/```(?:json)?\s*\n([\s\S]*?)\n```/);
  if (codeBlockMatch) {
    try {
      return JSON.parse(codeBlockMatch[1]!) as Record<string, unknown>;
    } catch {
      // fall through
    }
  }

  // Try to find a JSON object in the text
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]) as Record<string, unknown>;
    } catch {
      // fall through
    }
  }

  return null;
}

// ─── Main Runner ─────────────────────────────────────────────

export async function runWorkflowPhase(args: WorkflowRunArgs): Promise<void> {
  const { phase, inputPath, outputPath, eventsPath, repoPath } = args;

  logEvent(eventsPath, { type: "phase_start", phase });

  // 1. Read input
  let input: Record<string, unknown>;
  try {
    input = JSON.parse(readFileSync(inputPath, "utf-8")) as Record<string, unknown>;
  } catch (err) {
    logEvent(eventsPath, { type: "error", message: `Failed to read input: ${err}` });
    die(`Failed to read input file: ${inputPath}`);
  }

  logEvent(eventsPath, { type: "input_loaded", phase, keys: Object.keys(input) });

  // 1b. Validate required fields
  try {
    validateInput(phase, input);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logEvent(eventsPath, { type: "error", message: `Input validation failed: ${msg}` });
    die(msg);
  }

  // 2. Build phase prompt
  const prompt = buildPhasePrompt(phase, input);

  // 3. Set up and run the task
  // We import main's setup functions dynamically to avoid circular deps
  // and to reuse the full DevAgent engine (tools, providers, etc.)
  const { setupAndRunWorkflowQuery } = await import("./workflow-engine.js");

  const result = await setupAndRunWorkflowQuery({
    query: prompt,
    repoPath,
    provider: args.provider,
    model: args.model,
    maxIterations: args.maxIterations,
    approvalMode: args.approvalMode ?? "full-auto",
    reasoning: args.reasoning,
    eventsPath,
  });

  // 4. Extract structured output from the LLM response
  const jsonOutput = extractJsonOutput(result.responseText);

  if (jsonOutput) {
    writeFileSync(outputPath, JSON.stringify(jsonOutput, null, 2));
    logEvent(eventsPath, { type: "output_written", phase, hasOutput: true });
  } else {
    // Write a fallback output with the raw text as summary
    const fallback = { summary: result.responseText, _raw: true };
    writeFileSync(outputPath, JSON.stringify(fallback, null, 2));
    logEvent(eventsPath, { type: "output_written", phase, hasOutput: true, raw: true });
  }

  logEvent(eventsPath, {
    type: "phase_end",
    phase,
    success: result.success,
    iterations: result.iterations,
  });

  if (!result.success) {
    process.exitCode = EXIT_CODE.PHASE_FAILED;
  }
}

export async function handleWorkflowCommand(argv: string[]): Promise<void> {
  const raw = parseWorkflowArgs(argv);
  if (raw.subcommand === "describe") {
    printRunnerDescription();
    return;
  }

  const args = validateWorkflowArgs(raw);
  await runWorkflowPhase(args);
}
