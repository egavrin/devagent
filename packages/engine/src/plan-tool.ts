/**
 * update_plan — Workflow tool for tracking multi-step task progress.
 * Lets the LLM create and update a structured plan with step statuses.
 * Persisted via SessionState when a persistence backend is bound.
 */

import type { ToolSpec } from "@devagent/core";
import type { EventBus } from "@devagent/core";
import type { SessionState } from "./session-state.js";

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

// ─── Tool Definition ────────────────────────────────────────

export function createPlanTool(
  bus: EventBus,
  getSessionState: () => SessionState | undefined,
  getIteration?: () => number,
): ToolSpec {
  return {
    name: "update_plan",
    description:
      "Track progress on multi-step tasks. Create or update a plan with steps (pending/in_progress/completed). Use for tasks with 3+ steps. Keep step descriptions under 7 words. One step should be in_progress at a time.",
    category: "state",
    paramSchema: {
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
    },
    resultSchema: {
      type: "object",
      properties: {
        plan: { type: "string" },
      },
    },
    handler: async (params) => {
      const stepsRaw = params["steps"] as string;
      const explanation = (params["explanation"] as string | undefined) ?? null;
      const allowRegression = params["allow_regression"] === true;

      let steps: PlanStep[];
      try {
        const parsed = JSON.parse(stepsRaw) as Array<{
          description: string;
          status: string;
        }>;
        steps = parsed.map((s, i) => ({
          description: validateDescription(s.description, i),
          status: validateStatus(s.status),
        }));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        // Surface validation errors directly; wrap parse errors with guidance
        const isValidationError = message.startsWith("Invalid plan step status:") || message.startsWith("Invalid plan step description");
        return {
          success: false,
          output: "",
          error: isValidationError
            ? message
            : `Invalid steps JSON. Expected: [{"description": "...", "status": "pending|in_progress|completed"}]`,
          artifacts: [],
        };
      }

      if (steps.length === 0) {
        return {
          success: false,
          output: "",
          error: "Plan must have at least one step.",
          artifacts: [],
        };
      }

      // Single-pass status counting
      let inProgressCount = 0;
      let newCompleted = 0;
      for (const s of steps) {
        if (s.status === "in_progress") inProgressCount++;
        else if (s.status === "completed") newCompleted++;
      }

      if (inProgressCount > 1) {
        return {
          success: false,
          output: "",
          error: `Plan must have at most one in_progress step (found ${inProgressCount}).`,
          artifacts: [],
        };
      }

      // ── Transition metadata enrichment ──────────────────────
      const oldPlan = getSessionState()?.getPlan();
      const currentIter = getIteration?.() ?? 0;
      const now = Date.now();
      steps = steps.map((newStep) => {
        const oldStep = oldPlan?.find((o) => isSamePlanStep(o.description, newStep.description));
        if (!oldStep || oldStep.status !== newStep.status) {
          // Status changed or new step — stamp transition
          return {
            ...newStep,
            lastTransitionIteration: currentIter,
            lastTransitionTimestamp: now,
          };
        }
        // Unchanged — preserve prior metadata
        return {
          ...newStep,
          lastTransitionIteration: oldStep.lastTransitionIteration,
          lastTransitionTimestamp: oldStep.lastTransitionTimestamp,
        };
      });

      // ── Regression guard ──────────────────────────────────
      // Fail fast when an update reverts/drops completed steps unless
      // caller explicitly opts into regression.
      if (oldPlan && oldPlan.length > 0) {
        const oldCompleted = oldPlan.filter((s) => s.status === "completed");
        const newCompleted = steps.filter((s) => s.status === "completed");
        const revertedSteps = oldCompleted
          .filter((oldStep) =>
            !newCompleted.some((newStep) =>
              isSamePlanStep(oldStep.description, newStep.description)
            )
          )
          .map((s) => s.description);

        const hasIncompleteOld = oldPlan.some((s) => s.status !== "completed");
        const matchedOldSteps = oldPlan.filter((oldStep) =>
          steps.some((newStep) => isSamePlanStep(oldStep.description, newStep.description))
        ).length;
        const minMatchedForActivePlan = Math.max(1, Math.ceil(oldPlan.length * 0.5));
        const churnDetected = hasIncompleteOld && matchedOldSteps < minMatchedForActivePlan;

        if (churnDetected) {
          if (!allowRegression) {
            return {
              success: false,
              output: "",
              error:
                `Plan churn detected: this update keeps only ${matchedOldSteps}/${oldPlan.length} existing step(s) while the current plan is still in progress. Update statuses on existing steps instead of replacing the plan, or retry with allow_regression=true if reset is intentional.`,
              artifacts: [],
            };
          }
        }

        if (revertedSteps.length > 0) {
          bus.emit("plan:regression", {
            oldCompleted: oldCompleted.length,
            newCompleted: newCompleted.length,
            reverted: revertedSteps.length,
          });
          if (!allowRegression) {
            return {
              success: false,
              output: "",
              error: `Plan regression detected: this update reverts ${revertedSteps.length} previously completed step(s): ${revertedSteps.map((d) => `"${d}"`).join(", ")}. Preserve completed steps, or retry with allow_regression=true if reset is intentional.`,
              artifacts: [],
            };
          }
        }
      }

      // Sync to session state sidecar (survives compaction)
      getSessionState()?.setPlan(steps);

      // Emit plan event
      bus.emit("plan:updated", {
        steps: steps.map((s) => ({
          description: s.description,
          status: s.status,
          lastTransitionIteration: s.lastTransitionIteration,
        })),
        explanation,
      });

      // Format plan for display
      const planDisplay = steps
        .map((s) => {
          const icon =
            s.status === "completed"
              ? "[x]"
              : s.status === "in_progress"
                ? "[>]"
                : "[ ]";
          return `${icon} ${s.description}`;
        })
        .join("\n");

      const total = steps.length;

      return {
        success: true,
        output: `Plan updated (${newCompleted}/${total} completed):\n${planDisplay}`,
        error: null,
        artifacts: [],
      };
    },
  };
}

// ─── Helpers ────────────────────────────────────────────────

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
