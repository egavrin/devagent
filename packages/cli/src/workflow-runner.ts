/**
 * Headless workflow runner — executes a single workflow phase
 * with structured JSON input/output, no TTY required.
 *
 * Usage:
 *   devagent workflow run \
 *     --phase triage \
 *     --repo /path/to/repo \
 *     --input issue.json \
 *     --output triage.json \
 *     --events events.jsonl
 */

import { readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  EventBus,
  ApprovalGate,
  ContextManager,
  MemoryStore,
  loadConfig,
  resolveProviderCredentials,
  loadModelRegistry,
  lookupModelCapabilities,
  lookupModelEntry,
  DEFAULT_BUDGET,
  DEFAULT_CONTEXT,
  SkillRegistry,
} from "@devagent/core";
import type { DevAgentConfig, ApprovalPolicy } from "@devagent/core";
import { ApprovalMode } from "@devagent/core";
import { createDefaultRegistry } from "@devagent/providers";
import { createDefaultToolRegistry } from "@devagent/tools";
import {
  TaskLoop,
  CheckpointManager,
  DoubleCheck,
  DEFAULT_DOUBLE_CHECK_OPTIONS,
  SessionState,
} from "@devagent/engine";
import type { TaskMode, TaskLoopResult } from "@devagent/engine";
import { assembleSystemPrompt } from "./prompts/index.js";
import { resolveBundledModelsDir } from "./model-registry-path.js";
import type {
  WorkflowPhase,
  PhaseResult,
} from "@devagent/core";
import { WORKFLOW_SCHEMA_VERSION } from "@devagent/core";

// ─── Types ──────────────────────────────────────────────────

export interface WorkflowRunArgs {
  readonly phase: WorkflowPhase;
  readonly repo: string;
  readonly inputFile: string;
  readonly outputFile: string;
  readonly eventsFile?: string;
  readonly provider?: string;
  readonly model?: string;
  readonly maxIterations?: number;
  readonly approval?: string;
}

// ─── Phase Configuration ────────────────────────────────────

/** Map phase to task mode: read-only phases use "plan", mutating phases use "act". */
function phaseToMode(phase: WorkflowPhase): TaskMode {
  switch (phase) {
    case "triage":
    case "plan":
    case "review":
      return "plan";
    case "implement":
    case "verify":
    case "repair":
      return "act";
  }
}

/** Map phase to approval mode: plan-mode phases use suggest, act-mode phases use full-auto. */
function phaseToApprovalMode(phase: WorkflowPhase, override?: string): ApprovalMode {
  if (override) {
    if (override === "suggest") return ApprovalMode.SUGGEST;
    if (override === "auto-edit") return ApprovalMode.AUTO_EDIT;
    if (override === "full-auto") return ApprovalMode.FULL_AUTO;
  }
  // Default: read-only phases get suggest, mutating phases get full-auto
  const mode = phaseToMode(phase);
  return mode === "plan" ? ApprovalMode.SUGGEST : ApprovalMode.FULL_AUTO;
}

/** Build a phase-specific system prompt suffix instructing the agent to output structured JSON. */
function phasePromptSuffix(phase: WorkflowPhase): string {
  const common = `\n\n## Output Format\nYou MUST end your response with a single JSON block wrapped in \`\`\`json fences. This JSON is your structured output for this workflow phase. Do not include any text after the closing fence.`;

  switch (phase) {
    case "triage":
      return `${common}\nThe JSON must conform to the TriageReport schema: { issueId, title, acceptanceCriteria[], risks[], missingContext[], suggestedLabels[], complexity, duplicateSignals[] }.`;
    case "plan":
      return `${common}\nThe JSON must conform to the PlanDraft schema: { issueId, steps[{ order, description, files[], tests[], dependencies[] }], affectedFiles[], testStrategy, rollbackRisks[], estimatedPhases }.`;
    case "implement":
      return `${common}\nThe JSON must conform to the ExecutionReport schema: { issueId, planStepsCompleted, planStepsTotal, filesModified[], filesCreated[], testsAdded[], iterations, cost: { inputTokens, outputTokens, totalUsd } }.`;
    case "verify":
      return `${common}\nThe JSON must conform to the VerificationReport schema: { commands[{ command, exitCode, passed, stdout, stderr, durationMs }], allPassed, failingSummary }.`;
    case "review":
      return `${common}\nThe JSON must conform to the ReviewReport schema: { findings[{ severity, file, line?, message, suggestion? }], blockingCount, warningCount, infoCount, verdict }.`;
    case "repair":
      return `${common}\nThe JSON must conform to the RepairReport schema: { round, inputFindings, fixedFindings, remainingFindings, filesModified[], verificationPassed }.`;
  }
}

// ─── Argument Parsing ───────────────────────────────────────

export function parseWorkflowArgs(argv: string[]): WorkflowRunArgs | null {
  // Expected: devagent workflow run --phase <p> --repo <r> --input <i> --output <o> [opts]
  // argv already has "workflow" and "run" stripped by the time we get here,
  // or we parse from the raw argv.
  const args = argv;
  let phase: WorkflowPhase | null = null;
  let repo: string | null = null;
  let inputFile: string | null = null;
  let outputFile: string | null = null;
  let eventsFile: string | undefined;
  let provider: string | undefined;
  let model: string | undefined;
  let maxIterations: number | undefined;
  let approval: string | undefined;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--phase" && i + 1 < args.length) {
      phase = args[++i]! as WorkflowPhase;
    } else if (arg === "--repo" && i + 1 < args.length) {
      repo = args[++i]!;
    } else if (arg === "--input" && i + 1 < args.length) {
      inputFile = args[++i]!;
    } else if (arg === "--output" && i + 1 < args.length) {
      outputFile = args[++i]!;
    } else if (arg === "--events" && i + 1 < args.length) {
      eventsFile = args[++i]!;
    } else if (arg === "--provider" && i + 1 < args.length) {
      provider = args[++i]!;
    } else if (arg === "--model" && i + 1 < args.length) {
      model = args[++i]!;
    } else if (arg === "--max-iterations" && i + 1 < args.length) {
      const val = parseInt(args[++i]!, 10);
      if (!isNaN(val)) maxIterations = val;
    } else if (arg === "--approval" && i + 1 < args.length) {
      approval = args[++i]!;
    }
  }

  const validPhases: ReadonlyArray<string> = ["triage", "plan", "implement", "verify", "review", "repair"];
  if (!phase || !validPhases.includes(phase)) {
    return null;
  }
  if (!repo || !inputFile || !outputFile) {
    return null;
  }

  return { phase, repo, inputFile, outputFile, eventsFile, provider, model, maxIterations, approval };
}

// ─── JSON Extraction ────────────────────────────────────────

/**
 * Extract JSON from the agent's final response text.
 * Looks for ```json fenced blocks first, then tries raw JSON parse.
 */
function extractJsonFromResponse(text: string): unknown | null {
  // Try fenced JSON block
  const fencedMatch = text.match(/```json\s*\n([\s\S]*?)\n\s*```/);
  if (fencedMatch) {
    try {
      return JSON.parse(fencedMatch[1]!);
    } catch {
      // Fall through
    }
  }

  // Try raw JSON (find first { or [)
  const jsonStart = text.search(/[{[]/);
  if (jsonStart >= 0) {
    const candidate = text.slice(jsonStart);
    try {
      return JSON.parse(candidate);
    } catch {
      // Try to find matching close brace by tracking depth.
      // NOTE: This heuristic tracks unescaped double-quote characters to skip
      // braces inside JSON string literals. It does NOT handle all edge cases,
      // e.g. single-quoted strings (invalid JSON but sometimes seen), nested
      // escaped sequences like \\", or non-string values that happen to
      // contain quote-like characters. For those cases the JSON.parse above
      // or the fenced-block path should be preferred.
      let depth = 0;
      let inString = false;
      const opener = candidate[0];
      const closer = opener === "{" ? "}" : "]";
      for (let i = 0; i < candidate.length; i++) {
        const ch = candidate[i];
        if (inString) {
          if (ch === "\\" ) {
            i++; // skip escaped character
            continue;
          }
          if (ch === '"') {
            inString = false;
          }
          continue;
        }
        // Not inside a string
        if (ch === '"') {
          inString = true;
          continue;
        }
        if (ch === opener) depth++;
        if (ch === closer) depth--;
        if (depth === 0) {
          try {
            return JSON.parse(candidate.slice(0, i + 1));
          } catch {
            break;
          }
        }
      }
    }
  }

  return null;
}

// ─── Event Emitter ──────────────────────────────────────────

/** Wire event bus to emit JSONL events to a file. */
function wireEventFile(bus: EventBus, eventsFile: string): void {
  const write = (event: Record<string, unknown>) => {
    const line = JSON.stringify({ ...event, ts: new Date().toISOString() });
    appendFileSync(eventsFile, line + "\n");
  };

  bus.on("tool:before", (e) => write({ type: "tool:before", name: e.name, callId: e.callId }));
  bus.on("tool:after", (e) => write({ type: "tool:after", name: e.name, callId: e.callId }));
  bus.on("message:assistant", (e) => write({ type: "message:assistant", content: e.content, partial: e.partial }));
  bus.on("approval:request", (e) => write({ type: "approval:request", id: e.id, action: e.action }));
  bus.on("iteration:start", (e) => write({ type: "iteration:start", iteration: e.iteration }));
  bus.on("error", (e) => write({ type: "error", message: e.message, code: e.code }));
}

// ─── Main Runner ────────────────────────────────────────────

/**
 * Execute a single workflow phase headlessly.
 * Reads structured JSON input, runs the agent, writes structured JSON output.
 */
export async function runWorkflowPhase(wfArgs: WorkflowRunArgs): Promise<void> {
  const startTime = Date.now();

  // 1. Read input
  let input: unknown;
  try {
    const inputRaw = readFileSync(wfArgs.inputFile, "utf-8");
    input = JSON.parse(inputRaw);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    process.stderr.write(`Failed to read or parse input file "${wfArgs.inputFile}": ${message}\n`);
    process.exit(1);
  }

  // 2. Load config
  const approvalMode = phaseToApprovalMode(wfArgs.phase, wfArgs.approval);
  const configOverrides: Partial<DevAgentConfig> = {
    ...(wfArgs.provider ? { provider: wfArgs.provider } : {}),
    ...(wfArgs.model ? { model: wfArgs.model } : {}),
    approval: {
      mode: approvalMode,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    } as ApprovalPolicy,
  };

  let config = loadConfig(wfArgs.repo, configOverrides);
  config = await resolveProviderCredentials(config);

  if (wfArgs.maxIterations) {
    config = {
      ...config,
      budget: { ...config.budget, maxIterations: wfArgs.maxIterations },
    };
  }

  // Load model registry
  const cliDir = dirname(fileURLToPath(import.meta.url));
  const devagentModelsDir = resolveBundledModelsDir(cliDir);
  loadModelRegistry(wfArgs.repo, [devagentModelsDir]);

  // Auto-size context budget
  const registryEntry = lookupModelEntry(config.model);
  if (registryEntry && config.budget.maxContextTokens === DEFAULT_BUDGET.maxContextTokens) {
    config = {
      ...config,
      budget: {
        ...config.budget,
        maxContextTokens: registryEntry.contextWindow,
        responseHeadroom: registryEntry.responseHeadroom,
      },
    };
  }

  // Scale keepRecentMessages
  if (config.context.keepRecentMessages === DEFAULT_CONTEXT.keepRecentMessages) {
    const effectiveBudget = config.budget.maxContextTokens - config.budget.responseHeadroom;
    const scaledKeep = Math.floor(effectiveBudget / 1500);
    if (scaledKeep > DEFAULT_CONTEXT.keepRecentMessages) {
      config = {
        ...config,
        context: { ...config.context, keepRecentMessages: scaledKeep },
      };
    }
  }

  // 3. Create provider
  const providerRegistry = createDefaultRegistry();
  const baseProviderConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
  };
  const registryCaps = lookupModelCapabilities(config.model);
  const providerConfig = {
    ...baseProviderConfig,
    ...(!baseProviderConfig.capabilities && registryCaps ? { capabilities: registryCaps } : {}),
  };
  const provider = providerRegistry.get(config.provider, providerConfig);

  // 4. Create tools, bus, gate
  const toolRegistry = createDefaultToolRegistry();
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);

  // Wire events file
  if (wfArgs.eventsFile) {
    wireEventFile(bus, wfArgs.eventsFile);
  }

  // Minimal infrastructure — no interactive features
  const contextManager = new ContextManager(config.context);
  const memoryStore = new MemoryStore({
    dailyDecay: config.memory.dailyDecay,
    minRelevance: config.memory.minRelevance,
    accessBoost: config.memory.accessBoost,
  });
  const sessionState = new SessionState(config.sessionState);
  const checkpointManager = new CheckpointManager({
    repoRoot: wfArgs.repo,
    bus,
    enabled: false, // No checkpoints in headless mode
  });
  const doubleCheck = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);

  // 5. Build system prompt
  const mode = phaseToMode(wfArgs.phase);
  const skills = new SkillRegistry();
  const baseSystemPrompt = assembleSystemPrompt({
    mode,
    repoRoot: wfArgs.repo,
    skills,
    approvalMode,
    provider: config.provider,
    model: config.model,
  });
  const systemPrompt = baseSystemPrompt + phasePromptSuffix(wfArgs.phase);

  // 6. Build user query from input
  const query = buildPhaseQuery(wfArgs.phase, input);

  // 7. Run task loop
  const loop = new TaskLoop({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot: wfArgs.repo,
    mode,
    contextManager,
    memoryStore,
    checkpointManager,
    doubleCheck,
    sessionState,
  });

  const result: TaskLoopResult = await loop.run(query);

  // 8. Extract structured output
  const durationMs = Date.now() - startTime;
  const extracted = result.lastText ? extractJsonFromResponse(result.lastText) : null;

  const phaseResult: PhaseResult<unknown> = {
    schemaVersion: WORKFLOW_SCHEMA_VERSION,
    phase: wfArgs.phase,
    timestamp: new Date().toISOString(),
    durationMs,
    result: extracted ?? { raw: result.lastText, parseError: "Could not extract structured JSON from agent response" },
    summary: result.lastText ?? "(no response)",
  };

  // 9. Write output
  try {
    writeFileSync(wfArgs.outputFile, JSON.stringify(phaseResult, null, 2) + "\n");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    process.stderr.write(`Failed to write output file "${wfArgs.outputFile}": ${message}\n`);
  }

  // Emit completion event
  if (wfArgs.eventsFile) {
    const line = JSON.stringify({
      type: "phase:complete",
      phase: wfArgs.phase,
      durationMs,
      status: result.status,
      iterations: result.iterations,
      ts: new Date().toISOString(),
    });
    appendFileSync(wfArgs.eventsFile, line + "\n");
  }

  // Exit with non-zero if the agent aborted
  if (result.aborted) {
    process.stderr.write(`Workflow phase "${wfArgs.phase}" aborted after ${result.iterations} iterations\n`);
    process.exit(1);
  }
}

/** Build a natural-language query for the agent from the phase input data. */
function buildPhaseQuery(phase: WorkflowPhase, input: unknown): string {
  const inputStr = JSON.stringify(input, null, 2);

  switch (phase) {
    case "triage":
      return `Triage the following issue. Analyze it for complexity, acceptance criteria, risks, and missing context.\n\nInput:\n${inputStr}`;
    case "plan":
      return `Create a detailed implementation plan for the following issue. Break it into ordered steps with affected files and test strategy.\n\nInput:\n${inputStr}`;
    case "implement":
      return `Implement the following plan. Make the necessary code changes, create tests, and report what was done.\n\nInput:\n${inputStr}`;
    case "verify":
      return `Verify the implementation by running tests and checking for issues. Report all verification commands and their results.\n\nInput:\n${inputStr}`;
    case "review":
      return `Review the changes made in this implementation. Check for correctness, style, and potential issues.\n\nInput:\n${inputStr}`;
    case "repair":
      return `Fix the issues identified in the review. Apply the necessary repairs and verify they resolve the findings.\n\nInput:\n${inputStr}`;
  }
}

// ─── CLI Entrypoint ─────────────────────────────────────────

/**
 * Handle the `devagent workflow run` command.
 * Called from main.ts when the first positional arg is "workflow".
 */
export async function handleWorkflowCommand(argv: string[]): Promise<void> {
  // argv: full process.argv
  const args = argv.slice(2); // strip node/bun + script

  if (args[0] !== "workflow") {
    process.stderr.write("Internal error: handleWorkflowCommand called without 'workflow' arg\n");
    process.exit(1);
  }

  const subcommand = args[1];
  if (subcommand !== "run") {
    process.stderr.write(`Unknown workflow subcommand: ${subcommand ?? "(none)"}\n`);
    process.stderr.write("Usage: devagent workflow run --phase <phase> --repo <path> --input <file> --output <file> [--events <file>]\n");
    process.exit(1);
  }

  const wfArgs = parseWorkflowArgs(args.slice(2)); // strip "workflow" and "run"
  if (!wfArgs) {
    process.stderr.write("Missing required arguments.\n");
    process.stderr.write("Usage: devagent workflow run --phase <phase> --repo <path> --input <file> --output <file> [--events <file>]\n");
    process.stderr.write("  --phase: triage | plan | implement | verify | review | repair\n");
    process.stderr.write("  --repo: path to the repository\n");
    process.stderr.write("  --input: path to input JSON file\n");
    process.stderr.write("  --output: path to output JSON file\n");
    process.stderr.write("  --events: (optional) path to JSONL events file\n");
    process.stderr.write("  --provider: (optional) LLM provider\n");
    process.stderr.write("  --model: (optional) model ID\n");
    process.stderr.write("  --max-iterations: (optional) max iterations\n");
    process.stderr.write("  --approval: (optional) suggest | auto-edit | full-auto\n");
    process.exit(1);
  }

  await runWorkflowPhase(wfArgs);
}
