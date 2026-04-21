/**
 * ToolScriptEngine — batched readonly tool execution.
 *
 * Accepts a script (array of steps), validates all tools are readonly,
 * executes sequentially with inter-step reference resolution, and returns
 * aggregated results. Reduces N LLM round-trips to 1 for information-gathering.
 *
 * Reference syntax:
 *   $stepId       → full output of that step
 *   $stepId.lines[N] → Nth line (0-indexed) of that step's output
 */

import type { ToolResult, ToolContext , EventBus } from "../core/index.js";
import { extractErrorMessage, extractToolFileChangePreviewSummary } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

// ─── Types ──────────────────────────────────────────────────

export interface ToolScriptStep {
  readonly id: string;
  readonly tool: string;
  readonly args: Record<string, unknown>;
}

export interface ToolScript {
  readonly steps: ReadonlyArray<ToolScriptStep>;
}

/**
 * Parse a raw `steps` argument from provider output into validated steps.
 * Accepts both arrays (primary path) and JSON strings (legacy fallback).
 * Returns null if the input is malformed.
 */
export function parseToolScriptStepsArg(raw: unknown): ToolScriptStep[] | null {
  const entries = parseStepEntries(raw);
  if (!entries) return null;
  const steps = entries.map(parseToolScriptStepEntry);
  return steps.every((step): step is ToolScriptStep => step !== null) ? steps : null;
}

function parseStepEntries(raw: unknown): unknown[] | null {
  if (Array.isArray(raw)) return raw;
  if (typeof raw !== "string") return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function parseToolScriptStepEntry(entry: unknown): ToolScriptStep | null {
  if (!entry || typeof entry !== "object") return null;
  const step = entry as Record<string, unknown>;
  const args = parseStepArgs(step["args"]);
  if (typeof step["id"] !== "string" || typeof step["tool"] !== "string" || !args) {
    return null;
  }
  return {
    id: step["id"],
    tool: step["tool"],
    args,
  };
}

function parseStepArgs(rawArgs: unknown): Record<string, unknown> | null {
  const args = typeof rawArgs === "string" ? parseJsonObject(rawArgs) : rawArgs;
  return args && typeof args === "object" && !Array.isArray(args)
    ? { ...(args as Record<string, unknown>) }
    : null;
}

function parseJsonObject(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export interface StepResult {
  readonly id: string;
  readonly tool: string;
  readonly success: boolean;
  readonly output: string;
  readonly error: string | null;
  readonly durationMs: number;
}

export interface ToolScriptResult {
  readonly steps: ReadonlyArray<StepResult>;
  readonly totalDurationMs: number;
  readonly truncated: boolean;
}

export interface ToolScriptEngineOptions {
  readonly registry: ToolRegistry;
  readonly context: ToolContext;
  readonly bus: EventBus;
  readonly maxOutputBytes?: number; // default: 50KB
  readonly maxSteps?: number; // default: 20
}

// ─── Constants ──────────────────────────────────────────────

const DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024; // 50KB
const DEFAULT_MAX_STEPS = 20;

/** Tools that cannot appear in scripts (recursion guard). */
const BLOCKED_TOOLS = new Set(["execute_tool_script"]);

/** Pattern: $stepId or $stepId.lines[N] */
const REF_PATTERN = /\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.lines\[(\d+)\])?/g;

interface ExecuteScriptState {
  readonly results: Map<string, StepResult>;
  readonly completed: Set<string>;
  readonly stepResults: StepResult[];
  cumulativeOutputBytes: number;
  truncated: boolean;
}

// ─── Engine ─────────────────────────────────────────────────

export class ToolScriptEngine {
  private readonly registry: ToolRegistry;
  private readonly context: ToolContext;
  private readonly bus: EventBus;
  private readonly maxOutputBytes: number;
  private readonly maxSteps: number;

  constructor(options: ToolScriptEngineOptions) {
    this.registry = options.registry;
    this.context = options.context;
    this.bus = options.bus;
    this.maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
    this.maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
  }

  /**
   * Execute a tool script — validate, then run steps using a dependency-aware
   * hybrid scheduler. Steps with no dependencies run in parallel; steps that
   * reference earlier outputs wait for their dependencies to complete first.
   */
  async execute(script: ToolScript): Promise<ToolScriptResult> {
    const validationError = this.validate(script);
    if (validationError) {
      return {
        steps: [
          {
            id: "__validation__",
            tool: "",
            success: false,
            output: "",
            error: validationError,
            durationMs: 0,
          },
        ],
        totalDurationMs: 0,
        truncated: false,
      };
    }

    const totalStart = Date.now();
    const deps = buildDependencyGraph(script.steps, (args) => this.extractReferences(args));
    const state = makeExecuteScriptState();
    await this.executeReadyWaves(script.steps, deps, state);

    return {
      steps: state.stepResults,
      totalDurationMs: Date.now() - totalStart,
      truncated: state.truncated,
    };
  }

  private async executeReadyWaves(
    steps: ReadonlyArray<ToolScriptStep>,
    deps: ReadonlyMap<string, ReadonlySet<string>>,
    state: ExecuteScriptState,
  ): Promise<void> {
    while (state.completed.size < steps.length && !state.truncated) {
      const ready = getReadySteps(steps, deps, state.completed);
      if (ready.length === 0) break;
      const waveResults = await this.executeWave(ready, state.results);
      this.recordWaveResults(waveResults, steps, state);
    }
  }

  private async executeWave(
    ready: ReadonlyArray<ToolScriptStep>,
    results: Map<string, StepResult>,
  ): Promise<StepResult[]> {
    return ready.length === 1
      ? [await this.executeStep(ready[0]!, results)]
      : Promise.all(ready.map((step) => this.executeStep(step, results)));
  }

  private recordWaveResults(
    waveResults: ReadonlyArray<StepResult>,
    steps: ReadonlyArray<ToolScriptStep>,
    state: ExecuteScriptState,
  ): void {
    for (const stepResult of waveResults) {
      recordStepResult(stepResult, state);
      if (state.cumulativeOutputBytes > this.maxOutputBytes) {
        state.truncated = true;
        appendSkippedSteps(steps, state, this.maxOutputBytes);
        break;
      }
    }
  }

  // ─── Step Execution ─────────────────────────────────────────

  /**
   * Execute a single step: resolve references, call handler, emit events.
   */
  private async executeStep(
    step: ToolScriptStep,
    results: Map<string, StepResult>,
  ): Promise<StepResult> {
    const resolvedArgs = this.resolveReferences(step.args, results);
    const tool = this.registry.get(step.tool);

    // Emit tool:before
    const callId = `script_${step.id}`;
    this.bus.emit("tool:before", {
      name: step.tool,
      params: resolvedArgs,
      callId,
    });

    // Execute
    const stepStart = Date.now();
    let toolResult: ToolResult;
    try {
      toolResult = await tool.handler(resolvedArgs, this.context);
    } catch (err) {
      const message = extractErrorMessage(err);
      toolResult = {
        success: false,
        output: "",
        error: message,
        artifacts: [],
      };
    }
    const durationMs = Date.now() - stepStart;

    // Emit tool:after
    const fileEditSummary = extractToolFileChangePreviewSummary(toolResult.metadata);
    this.bus.emit("tool:after", {
      name: step.tool,
      result: toolResult,
      fileEdits: fileEditSummary.fileEdits,
      fileEditHiddenCount: fileEditSummary.hiddenFileCount > 0 ? fileEditSummary.hiddenFileCount : undefined,
      callId,
      durationMs,
    });

    return {
      id: step.id,
      tool: step.tool,
      success: toolResult.success,
      output: toolResult.success
        ? toolResult.output
        : `Error: ${toolResult.error}`,
      error: toolResult.error,
      durationMs,
    };
  }

  // ─── Validation ─────────────────────────────────────────────

  /**
   * Validate a script before execution. Returns error string or null if valid.
   */
  validate(script: ToolScript): string | null {
    if (!script.steps || script.steps.length === 0) {
      return "Script must have at least one step";
    }

    if (script.steps.length > this.maxSteps) {
      return `Script exceeds maximum of ${this.maxSteps} steps (got ${script.steps.length})`;
    }

    const seenIds = new Set<string>();
    const declaredIds: string[] = [];

    for (const step of script.steps) {
      const validationError = this.validateStep(step, seenIds, declaredIds);
      if (validationError) return validationError;

      declaredIds.push(step.id);
    }

    return null;
  }

  private validateStep(
    step: ToolScriptStep,
    seenIds: Set<string>,
    declaredIds: ReadonlyArray<string>,
  ): string | null {
    const idError = validateStepId(step, seenIds);
    if (idError) return idError;
    const toolError = validateStepTool(step, this.registry);
    if (toolError) return toolError;
    return validateStepReferences(step, declaredIds, this.extractReferences(step.args));
  }

  // ─── Reference Resolution ──────────────────────────────────

  /**
   * Deep-resolve references in tool arguments using results from earlier steps.
   */
  private resolveReferences(
    args: Record<string, unknown>,
    results: Map<string, StepResult>,
  ): Record<string, unknown> {
    const resolved: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(args)) {
      if (typeof value === "string") {
        resolved[key] = this.resolveStringReferences(value, results);
      } else if (Array.isArray(value)) {
        resolved[key] = value.map((item) =>
          typeof item === "string"
            ? this.resolveStringReferences(item, results)
            : item,
        );
      } else {
        resolved[key] = value;
      }
    }

    return resolved;
  }

  /**
   * Resolve $stepId and $stepId.lines[N] references in a string.
   */
  private resolveStringReferences(
    str: string,
    results: Map<string, StepResult>,
  ): string {
    return str.replace(
      REF_PATTERN,
      (match: string, stepId: string, lineIndex: string | undefined) => {
        const stepResult = results.get(stepId);
        if (!stepResult) {
          return match; // Not a reference — leave as-is
        }

        if (!stepResult.success) {
          return `<ref error: step "${stepId}" failed>`;
        }

        if (lineIndex !== undefined) {
          const lines = stepResult.output.split("\n");
          const idx = parseInt(lineIndex, 10);
          if (idx < 0 || idx >= lines.length) {
            return `<ref error: $${stepId}.lines[${lineIndex}] out of bounds (${lines.length} lines)>`;
          }
          return lines[idx]!;
        }

        return stepResult.output;
      },
    );
  }

  /**
   * Extract all reference IDs from args (for forward-reference validation).
   */
  private extractReferences(args: Record<string, unknown>): string[] {
    const refs: string[] = [];

    const extractFromValue = (value: unknown): void => {
      if (typeof value === "string") {
        let m: RegExpExecArray | null;
        const pattern = new RegExp(REF_PATTERN.source, REF_PATTERN.flags);
        while ((m = pattern.exec(value)) !== null) {
          refs.push(m[1]!);
        }
      } else if (Array.isArray(value)) {
        for (const item of value) {
          extractFromValue(item);
        }
      }
    };

    for (const value of Object.values(args)) {
      extractFromValue(value);
    }

    return refs;
  }
}

function makeExecuteScriptState(): ExecuteScriptState {
  return {
    results: new Map(),
    completed: new Set(),
    stepResults: [],
    cumulativeOutputBytes: 0,
    truncated: false,
  };
}

function buildDependencyGraph(
  steps: ReadonlyArray<ToolScriptStep>,
  extractReferences: (args: Record<string, unknown>) => ReadonlyArray<string>,
): Map<string, Set<string>> {
  const deps = new Map<string, Set<string>>();
  for (const step of steps) {
    deps.set(step.id, new Set(extractReferences(step.args)));
  }
  return deps;
}

function getReadySteps(
  steps: ReadonlyArray<ToolScriptStep>,
  deps: ReadonlyMap<string, ReadonlySet<string>>,
  completed: ReadonlySet<string>,
): ToolScriptStep[] {
  return steps.filter((step) => {
    if (completed.has(step.id)) return false;
    const stepDeps = deps.get(step.id) ?? new Set<string>();
    return [...stepDeps].every((dependency) => completed.has(dependency));
  });
}

function recordStepResult(stepResult: StepResult, state: ExecuteScriptState): void {
  state.results.set(stepResult.id, stepResult);
  state.completed.add(stepResult.id);
  state.stepResults.push(stepResult);
  state.cumulativeOutputBytes += Buffer.byteLength(stepResult.output, "utf8");
}

function appendSkippedSteps(
  steps: ReadonlyArray<ToolScriptStep>,
  state: ExecuteScriptState,
  maxOutputBytes: number,
): void {
  for (const step of steps) {
    if (state.completed.has(step.id)) continue;
    state.stepResults.push({
      id: step.id,
      tool: step.tool,
      success: false,
      output: "",
      error: `Skipped: output limit exceeded (${maxOutputBytes} bytes)`,
      durationMs: 0,
    });
    state.completed.add(step.id);
  }
}

function validateStepId(
  step: ToolScriptStep,
  seenIds: Set<string>,
): string | null {
  if (!step.id || typeof step.id !== "string") {
    return `Step missing required 'id' field`;
  }
  if (seenIds.has(step.id)) {
    return `Duplicate step ID: "${step.id}"`;
  }
  seenIds.add(step.id);
  return null;
}

function validateStepTool(step: ToolScriptStep, registry: ToolRegistry): string | null {
  const namespacedHint = namespacedToolHint(step.tool, registry);
  if (namespacedHint) return namespacedHint;
  if (BLOCKED_TOOLS.has(step.tool)) {
    return `Tool "${step.tool}" cannot be used inside scripts (recursion prevention)`;
  }
  if (!registry.has(step.tool)) return `Unknown tool: "${step.tool}"`;
  const tool = registry.get(step.tool);
  return tool.category === "readonly"
    ? null
    : `Only readonly tools are allowed in scripts. "${step.tool}" is "${tool.category}"`;
}

function validateStepReferences(
  step: ToolScriptStep,
  declaredIds: ReadonlyArray<string>,
  refs: ReadonlyArray<string>,
): string | null {
  for (const refId of refs) {
    if (refId === step.id) return `Step "${step.id}" references itself`;
    if (!declaredIds.includes(refId)) {
      return `Step "${step.id}" has forward reference to undeclared step "${refId}"`;
    }
  }
  return null;
}

function namespacedToolHint(toolName: string, registry: ToolRegistry): string | null {
  if (!/^(?:functions|function|tools)\./.test(toolName)) return null;

  const canonical = toolName.replace(/^(?:functions|function|tools)\./, "");
  if (registry.has(canonical)) {
    return `Invalid tool name "${toolName}". Use canonical tool names only: "${canonical}" (without namespace prefixes).`;
  }

  return `Invalid tool name "${toolName}". Use canonical tool names only (no prefixes like functions./function./tools.).`;
}
