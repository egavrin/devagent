/**
 * Plan file persistence -- writes plan updates to a human-readable
 * markdown file and reads them back for session resume.
 *
 * Plans are written to {repoRoot}/.devagent/plans/{sessionId}.md
 * as markdown checkboxes. This is supplementary to SessionState
 * persistence -- failures are non-fatal.
 */

import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";

import type { PlanStep } from "./plan-tool.js";

// ─── Path Resolution ───────────────────────────────────────

/**
 * Returns the absolute path to the plan markdown file for a given session.
 */
export function getPlanFilePath(sessionId: string, repoRoot: string): string {
  return join(repoRoot, ".devagent", "plans", `${sessionId}.md`);
}

// ─── Write ─────────────────────────────────────────────────

/**
 * Writes the current plan steps to a human-readable markdown file.
 *
 * Non-fatal: any I/O error is silently swallowed so plan persistence
 * never blocks the main agent loop.
 */
export function writePlanFile(
  sessionId: string,
  repoRoot: string,
  steps: ReadonlyArray<PlanStep>,
): void {
  try {
    const filePath = getPlanFilePath(sessionId, repoRoot);
    mkdirSync(dirname(filePath), { recursive: true });

    const lines = steps.map((step) => {
      const checkbox =
        step.status === "completed"
          ? "[x]"
          : step.status === "in_progress"
            ? "[>]"
            : "[ ]";

      const meta = formatStepMeta(step);
      return `- ${checkbox} ${step.description}${meta}`;
    });

    const content = `# Plan\n\n${lines.join("\n")}\n`;
    writeFileSync(filePath, content, "utf-8");
  } catch {
    // Non-fatal -- plan file is supplementary to SessionState persistence.
  }
}

// ─── Read ──────────────────────────────────────────────────

/**
 * Reads a plan markdown file back into structured PlanStep objects.
 *
 * Returns `null` when the file does not exist or cannot be parsed.
 * Non-fatal: any I/O or parse error returns `null`.
 */
export function readPlanFile(
  sessionId: string,
  repoRoot: string,
): PlanStep[] | null {
  try {
    const filePath = getPlanFilePath(sessionId, repoRoot);
    if (!existsSync(filePath)) return null;

    const content = readFileSync(filePath, "utf-8");
    const lines = content.split("\n");
    const steps: PlanStep[] = [];

    for (const line of lines) {
      const parsed = parsePlanLine(line);
      if (parsed) steps.push(parsed);
    }

    return steps.length > 0 ? steps : null;
  } catch {
    // Non-fatal -- return null so the caller falls back to SessionState.
    return null;
  }
}

// ─── Helpers ───────────────────────────────────────────────

/** Regex for plan lines: `- [x] description (iter 5, completed 2026-03-31T16:00:00Z)` */
const PLAN_LINE_RE =
  /^- \[([ x>])\] (.+?)(?:\s*\(([^)]*)\))?\s*$/;

/**
 * Parse a single markdown plan line into a PlanStep, or return `null`
 * if the line is not a valid plan checkbox.
 */
function parsePlanLine(line: string): PlanStep | null {
  const match = PLAN_LINE_RE.exec(line.trim());
  if (!match) return null;

  const [, marker, description, meta] = match;
  if (!description) return null;

  const status: PlanStep["status"] =
    marker === "x"
      ? "completed"
      : marker === ">"
        ? "in_progress"
        : "pending";

  const step: PlanStep = { description, status };

  if (meta) {
    const iterMatch = /iter\s+(\d+)/.exec(meta);
    if (iterMatch) {
      return {
        ...step,
        lastTransitionIteration: Number(iterMatch[1]),
        ...parseTimestamp(meta),
      };
    }
    const tsResult = parseTimestamp(meta);
    if (tsResult.lastTransitionTimestamp !== undefined) {
      return { ...step, ...tsResult };
    }
  }

  return step;
}

/**
 * Extract an ISO timestamp from metadata parenthetical if present.
 */
function parseTimestamp(
  meta: string,
): { lastTransitionTimestamp?: number } {
  const tsMatch = /(?:completed|started|changed)\s+(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)/.exec(meta);
  if (tsMatch) {
    const ts = Date.parse(tsMatch[1]!);
    if (!Number.isNaN(ts)) {
      return { lastTransitionTimestamp: ts };
    }
  }
  return {};
}

/**
 * Format optional step metadata as a parenthetical suffix.
 */
function formatStepMeta(step: PlanStep): string {
  const parts: string[] = [];

  if (step.lastTransitionIteration !== undefined) {
    parts.push(`iter ${step.lastTransitionIteration}`);
  }

  if (step.lastTransitionTimestamp !== undefined) {
    const label =
      step.status === "completed"
        ? "completed"
        : step.status === "in_progress"
          ? "started"
          : "changed";
    parts.push(`${label} ${new Date(step.lastTransitionTimestamp).toISOString()}`);
  }

  if (step.status === "pending" && parts.length === 0) {
    parts.push("pending");
  }

  return parts.length > 0 ? ` (${parts.join(", ")})` : "";
}
