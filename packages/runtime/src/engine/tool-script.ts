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

import type { ToolSpec, ToolResult, ToolContext } from "../core/index.js";
import type { EventBus } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";
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
  let entries: unknown[];
  if (Array.isArray(raw)) {
    entries = raw;
  } else if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return null;
      entries = parsed;
    } catch {
      return null;
    }
  } else {
    return null;
  }

  const steps: ToolScriptStep[] = [];
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") return null;
    const step = entry as Record<string, unknown>;
    const id = step["id"];
    const tool = step["tool"];
    let args = step["args"];
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        return null;
      }
    }
    if (typeof id !== "string" || typeof tool !== "string" || !args || typeof args !== "object") {
      return null;
    }
    steps.push({
      id,
      tool,
      args: { ...(args as Record<string, unknown>) },
    });
  }
  return steps;
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
    const results = new Map<string, StepResult>();
    let cumulativeOutputBytes = 0;
    let truncated = false;

    // Build dependency graph: stepId → set of stepIds it depends on
    const deps = new Map<string, Set<string>>();
    for (const step of script.steps) {
      const refs = this.extractReferences(step.args);
      deps.set(step.id, new Set(refs));
    }

    // Topological execution in waves: each wave contains steps whose
    // dependencies are all resolved. Steps within a wave run in parallel.
    const stepMap = new Map(script.steps.map((s) => [s.id, s]));
    const completed = new Set<string>();
    const stepResults: StepResult[] = [];

    while (completed.size < script.steps.length && !truncated) {
      // Find all steps ready to run (all deps satisfied)
      const ready: ToolScriptStep[] = [];
      for (const step of script.steps) {
        if (completed.has(step.id)) continue;
        const stepDeps = deps.get(step.id)!;
        const allDepsReady = [...stepDeps].every((d) => completed.has(d));
        if (allDepsReady) {
          ready.push(step);
        }
      }

      if (ready.length === 0) {
        // Should not happen with valid topological ordering (validation catches cycles)
        break;
      }

      // Execute the wave — single step runs directly, multiple run in parallel
      const waveResults = ready.length === 1
        ? [await this.executeStep(ready[0]!, results)]
        : await Promise.all(
            ready.map((step) => this.executeStep(step, results)),
          );

      // Record results in original script order (stable ordering)
      for (const stepResult of waveResults) {
        results.set(stepResult.id, stepResult);
        completed.add(stepResult.id);
        stepResults.push(stepResult);

        // Track cumulative output size
        cumulativeOutputBytes += Buffer.byteLength(stepResult.output, "utf8");
        if (cumulativeOutputBytes > this.maxOutputBytes) {
          truncated = true;
          // Mark remaining steps as skipped
          for (const step of script.steps) {
            if (!completed.has(step.id)) {
              stepResults.push({
                id: step.id,
                tool: step.tool,
                success: false,
                output: "",
                error: `Skipped: output limit exceeded (${this.maxOutputBytes} bytes)`,
                durationMs: 0,
              });
              completed.add(step.id);
            }
          }
          break;
        }
      }
    }

    return {
      steps: stepResults,
      totalDurationMs: Date.now() - totalStart,
      truncated,
    };
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
    this.bus.emit("tool:after", {
      name: step.tool,
      result: toolResult,
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
      // Validate step ID
      if (!step.id || typeof step.id !== "string") {
        return `Step missing required 'id' field`;
      }

      // Check duplicate IDs
      if (seenIds.has(step.id)) {
        return `Duplicate step ID: "${step.id}"`;
      }
      seenIds.add(step.id);

      // Namespaced tool names are invalid — use canonical registry names.
      const namespacedHint = namespacedToolHint(step.tool, this.registry);
      if (namespacedHint) {
        return namespacedHint;
      }

      // Check blocked tools (recursion guard)
      if (BLOCKED_TOOLS.has(step.tool)) {
        return `Tool "${step.tool}" cannot be used inside scripts (recursion prevention)`;
      }

      // Check tool exists
      if (!this.registry.has(step.tool)) {
        return `Unknown tool: "${step.tool}"`;
      }

      // Check tool is readonly
      const tool = this.registry.get(step.tool);
      if (tool.category !== "readonly") {
        return `Only readonly tools are allowed in scripts. "${step.tool}" is "${tool.category}"`;
      }

      // Check for self-references and forward references
      const refs = this.extractReferences(step.args);
      for (const refId of refs) {
        if (refId === step.id) {
          return `Step "${step.id}" references itself`;
        }
        if (!declaredIds.includes(refId)) {
          return `Step "${step.id}" has forward reference to undeclared step "${refId}"`;
        }
      }

      declaredIds.push(step.id);
    }

    return null;
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

function namespacedToolHint(toolName: string, registry: ToolRegistry): string | null {
  if (!/^(?:functions|function|tools)\./.test(toolName)) return null;

  const canonical = toolName.replace(/^(?:functions|function|tools)\./, "");
  if (registry.has(canonical)) {
    return `Invalid tool name "${toolName}". Use canonical tool names only: "${canonical}" (without namespace prefixes).`;
  }

  return `Invalid tool name "${toolName}". Use canonical tool names only (no prefixes like functions./function./tools.).`;
}
