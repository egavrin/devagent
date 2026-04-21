/**
 * update_plan — Workflow tool for tracking multi-step task progress.
 * Lets the LLM create and update a structured plan with step statuses.
 * Persisted via SessionState when a persistence backend is bound.
 */

import { writePlanFile } from "./plan-persistence.js";
import type { SessionState } from "./session-state.js";
import type { ToolSpec , EventBus } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";

// ─── Types ──────────────────────────────────────────────────

export interface PlanStep {
  readonly description: string;
  readonly status: "pending" | "in_progress" | "completed";
  readonly lastTransitionIteration?: number;
  readonly lastTransitionTimestamp?: number;
}

export interface Plan {
  readonly steps: ReadonlyArray<PlanStep>;
  readonly explanation: string | null;
}

const PLAN_PARAM_SCHEMA = {
  type: "object",
  properties: {
    steps: {
      type: "string",
      description:
        'JSON array of step objects: [{"description": "...", "status": "pending|in_progress|completed"}, ...]',
    },
    explanation: {
      type: "string",
      description: "Brief explanation of plan changes (optional)",
    },
    allow_regression: {
      type: "boolean",
      description: "Set true only when intentionally resetting/removing previously completed steps.",
    },
  },
  required: ["steps"],
};

const PLAN_RESULT_SCHEMA = {
  type: "object",
  properties: {
    plan: { type: "string" },
  },
};

// ─── Tool Definition ────────────────────────────────────────
export function createPlanTool(
  bus: EventBus,
  getSessionState: () => SessionState | undefined,
  getIteration?: () => number,
  onPlanUpdated?: (steps: ReadonlyArray<PlanStep>, oldPlan: ReadonlyArray<PlanStep> | null) => Promise<string | null>,
  options?: { readonly repoRoot?: string; readonly sessionId?: string },
): ToolSpec {
  return {
    name: "update_plan",
    description:
      "Track progress on multi-step tasks. Create or update a plan with steps (pending/in_progress/completed). Use for tasks with 3+ steps. Keep step descriptions under 7 words. One step should be in_progress at a time.",
    category: "state",
    errorGuidance: {
      common: "Ensure steps is a valid JSON array of {description, status} objects. Valid statuses: pending, in_progress, completed.",
    },
    paramSchema: PLAN_PARAM_SCHEMA,
    resultSchema: PLAN_RESULT_SCHEMA,
    handler: createPlanHandler({ bus, getSessionState, getIteration, onPlanUpdated, options }),
  };
}

// ─── Helpers ────────────────────────────────────────────────

function createPlanHandler(input: {
  readonly bus: EventBus;
  readonly getSessionState: () => SessionState | undefined;
  readonly getIteration?: () => number;
  readonly onPlanUpdated?: (steps: ReadonlyArray<PlanStep>, oldPlan: ReadonlyArray<PlanStep> | null) => Promise<string | null>;
  readonly options?: { readonly repoRoot?: string; readonly sessionId?: string };
}) {
  return async (params: Record<string, unknown>) => {
    const parsed = parsePlanSteps(params["steps"] as string);
    if ("error" in parsed) return parsed.error;

    const oldPlan = input.getSessionState()?.getPlan();
    const guarded = buildGuardedPlanUpdate(parsed.steps, oldPlan, params, input);
    if ("error" in guarded) return guarded.error;

    const judgeFeedback = await persistAndJudgePlan(guarded.steps, oldPlan, input);
    emitPlanUpdated(input.bus, guarded.steps, (params["explanation"] as string | undefined) ?? null);

    return {
      success: true,
      output: formatPlanUpdateOutput(guarded.steps, guarded, judgeFeedback),
      error: null,
      artifacts: [],
    };
  };
}

function buildGuardedPlanUpdate(
  steps: ReadonlyArray<PlanStep>,
  oldPlan: ReadonlyArray<PlanStep> | null | undefined,
  params: Record<string, unknown>,
  input: {
    readonly bus: EventBus;
    readonly getIteration?: () => number;
  },
): RegressionGuardResult | { readonly error: ReturnType<typeof planError> } {
  const enriched = enrichPlanTransitions(steps, oldPlan, input.getIteration?.() ?? 0);
  return applyRegressionGuard(enriched, oldPlan ?? null, params["allow_regression"] === true, input.bus);
}

async function persistAndJudgePlan(
  steps: ReadonlyArray<PlanStep>,
  oldPlan: ReadonlyArray<PlanStep> | null | undefined,
  input: {
    readonly getSessionState: () => SessionState | undefined;
    readonly onPlanUpdated?: (steps: ReadonlyArray<PlanStep>, oldPlan: ReadonlyArray<PlanStep> | null) => Promise<string | null>;
    readonly options?: { readonly repoRoot?: string; readonly sessionId?: string };
  },
): Promise<string | null> {
  input.getSessionState()?.setPlan([...steps]);
  if (input.options?.repoRoot && input.options?.sessionId) {
    writePlanFile(input.options.sessionId, input.options.repoRoot, steps);
  }
  if (input.onPlanUpdated && isStructuralChange(oldPlan ?? null, steps)) {
    return await input.onPlanUpdated(steps, oldPlan ?? null);
  }
  return null;
}

function emitPlanUpdated(
  bus: EventBus,
  steps: ReadonlyArray<PlanStep>,
  explanation: string | null,
): void {
  bus.emit("plan:updated", {
    steps: steps.map((s) => ({
      description: s.description,
      status: s.status,
      lastTransitionIteration: s.lastTransitionIteration,
    })),
    explanation,
  });
}

function planError(error: string) {
  return { success: false, output: "", error, artifacts: [] };
}

function parsePlanSteps(stepsRaw: string): { readonly steps: PlanStep[] } | { readonly error: ReturnType<typeof planError> } {
  try {
    const parsed = JSON.parse(stepsRaw) as Array<{ description: string; status: string }>;
    const steps = parsed.map((s, i) => ({
      description: validateDescription(s.description, i),
      status: validateStatus(s.status),
    }));
    return validateParsedPlanSteps(steps);
  } catch (err) {
    return { error: planError(formatPlanParseError(extractErrorMessage(err))) };
  }
}

function validateParsedPlanSteps(steps: PlanStep[]): { readonly steps: PlanStep[] } | { readonly error: ReturnType<typeof planError> } {
  if (steps.length === 0) return { error: planError("Plan must have at least one step.") };
  const inProgressCount = steps.filter((s) => s.status === "in_progress").length;
  if (inProgressCount > 1) {
    return { error: planError(`Plan must have at most one in_progress step (found ${inProgressCount}).`) };
  }
  return { steps };
}

function formatPlanParseError(message: string): string {
  const isValidationError = message.startsWith("Invalid plan step status:")
    || message.startsWith("Invalid plan step description");
  return isValidationError
    ? message
    : `Invalid steps JSON. Expected: [{"description": "...", "status": "pending|in_progress|completed"}]`;
}

function enrichPlanTransitions(
  steps: ReadonlyArray<PlanStep>,
  oldPlan: ReadonlyArray<PlanStep> | null | undefined,
  currentIter: number,
): PlanStep[] {
  const now = Date.now();
  return steps.map((newStep) => enrichPlanStep(newStep, oldPlan, currentIter, now));
}

function enrichPlanStep(
  newStep: PlanStep,
  oldPlan: ReadonlyArray<PlanStep> | null | undefined,
  currentIter: number,
  now: number,
): PlanStep {
  const oldStep = oldPlan?.find((o) => isSamePlanStep(o.description, newStep.description));
  if (!oldStep || oldStep.status !== newStep.status) {
    return { ...newStep, lastTransitionIteration: currentIter, lastTransitionTimestamp: now };
  }
  return {
    ...newStep,
    lastTransitionIteration: oldStep.lastTransitionIteration,
    lastTransitionTimestamp: oldStep.lastTransitionTimestamp,
  };
}

interface RegressionGuardResult {
  readonly steps: PlanStep[];
  readonly mergeOccurred: boolean;
  readonly preservedCount: number;
}

function applyRegressionGuard(
  steps: PlanStep[],
  oldPlan: ReadonlyArray<PlanStep> | null,
  allowRegression: boolean,
  bus: EventBus,
): RegressionGuardResult | { readonly error: ReturnType<typeof planError> } {
  if (!oldPlan || oldPlan.length === 0) {
    return { steps, mergeOccurred: false, preservedCount: 0 };
  }

  const churn = detectPlanChurn(oldPlan, steps);
  if (churn && !allowRegression) return { error: planError(churn.error) };

  const regression = detectPlanRegression(oldPlan, steps);
  if (regression.revertedSteps.length === 0) {
    return { steps, mergeOccurred: false, preservedCount: 0 };
  }

  bus.emit("plan:regression", regression.event);
  if (!allowRegression) return { error: planError(regression.error) };
  return maybeAutoMergeCompletedSteps(steps, regression.oldCompleted, regression.revertedSteps);
}

function detectPlanChurn(
  oldPlan: ReadonlyArray<PlanStep>,
  steps: ReadonlyArray<PlanStep>,
): { readonly error: string } | null {
  const hasIncompleteOld = oldPlan.some((s) => s.status !== "completed");
  const matchedOldSteps = oldPlan.filter((oldStep) =>
    steps.some((newStep) => isSamePlanStep(oldStep.description, newStep.description)),
  ).length;
  const minMatchedForActivePlan = Math.max(1, Math.ceil(oldPlan.length * 0.5));
  if (!hasIncompleteOld || matchedOldSteps >= minMatchedForActivePlan) return null;
  return {
    error: `Plan churn detected: this update keeps only ${matchedOldSteps}/${oldPlan.length} existing step(s) while the current plan is still in progress. Update statuses on existing steps instead of replacing the plan, or retry with allow_regression=true if reset is intentional.`,
  };
}

function detectPlanRegression(oldPlan: ReadonlyArray<PlanStep>, steps: ReadonlyArray<PlanStep>) {
  const oldCompleted = oldPlan.filter((s) => s.status === "completed");
  const newCompletedSteps = steps.filter((s) => s.status === "completed");
  const revertedSteps = oldCompleted
    .filter((oldStep) =>
      !newCompletedSteps.some((newStep) => isSamePlanStep(oldStep.description, newStep.description)),
    )
    .map((s) => s.description);
  return {
    oldCompleted,
    revertedSteps,
    event: {
      oldCompleted: oldCompleted.length,
      newCompleted: newCompletedSteps.length,
      reverted: revertedSteps.length,
    },
    error: `Plan regression detected: this update reverts ${revertedSteps.length} previously completed step(s): ${revertedSteps.map((d) => `"${d}"`).join(", ")}. Preserve completed steps, or retry with allow_regression=true if reset is intentional.`,
  };
}

function maybeAutoMergeCompletedSteps(
  steps: PlanStep[],
  oldCompleted: ReadonlyArray<PlanStep>,
  revertedSteps: ReadonlyArray<string>,
): RegressionGuardResult {
  const catastropheThreshold = Math.max(3, Math.ceil(oldCompleted.length * 0.5));
  if (revertedSteps.length < catastropheThreshold) {
    return { steps, mergeOccurred: false, preservedCount: 0 };
  }
  const newNonOverlapping = steps.filter(
    (ns) => !oldCompleted.some((oc) => isSamePlanStep(oc.description, ns.description)),
  );
  return {
    steps: [...oldCompleted, ...newNonOverlapping],
    mergeOccurred: true,
    preservedCount: oldCompleted.length,
  };
}

function formatPlanUpdateOutput(
  steps: ReadonlyArray<PlanStep>,
  guarded: RegressionGuardResult,
  judgeFeedback: string | null,
): string {
  const newCompleted = steps.filter((s) => s.status === "completed").length;
  const mergeNote = guarded.mergeOccurred
    ? ` (${guarded.preservedCount} previously completed step(s) auto-preserved)`
    : "";
  const judgeNote = judgeFeedback ? `\n\n${judgeFeedback}` : "";
  return `Plan updated (${newCompleted}/${steps.length} completed)${mergeNote}:\n${formatPlanDisplay(steps)}${judgeNote}`;
}

function formatPlanDisplay(steps: ReadonlyArray<PlanStep>): string {
  return steps.map((s) => `${statusIcon(s.status)} ${s.description}`).join("\n");
}

function statusIcon(status: PlanStep["status"]): string {
  if (status === "completed") return "[x]";
  if (status === "in_progress") return "[>]";
  return "[ ]";
}

function validateDescription(description: unknown, index: number): string {
  if (typeof description !== "string" || description.trim().length === 0) {
    throw new Error(
      `Invalid plan step description at index ${index}: must be a non-empty string.`,
    );
  }
  return description;
}

function validateStatus(
  status: string,
): "pending" | "in_progress" | "completed" {
  if (
    status === "pending" ||
    status === "in_progress" ||
    status === "completed"
  ) {
    return status;
  }
  throw new Error(
    `Invalid plan step status: "${status}". Must be pending, in_progress, or completed.`,
  );
}

function isSamePlanStep(a: string, b: string): boolean {
  return normalizeStepDescription(a) === normalizeStepDescription(b);
}

function normalizeStepDescription(description: string): string {
  return description
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

/**
 * Detect whether a plan update is structural (description changes)
 * vs. status-only (same steps, just status transitions).
 */
export function isStructuralChange(
  oldPlan: ReadonlyArray<PlanStep> | null,
  newPlan: ReadonlyArray<PlanStep>,
): boolean {
  if (!oldPlan) return true;
  if (oldPlan.length !== newPlan.length) return true;

  for (let i = 0; i < oldPlan.length; i++) {
    if (oldPlan[i]!.description !== newPlan[i]!.description) return true;
  }
  return false;
}
