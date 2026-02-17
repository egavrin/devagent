/**
 * update_plan — Workflow tool for tracking multi-step task progress.
 * Lets the LLM create and update a structured plan with step statuses.
 * Session-scoped (not persisted).
 */

import type { ToolSpec } from "@devagent/core";
import type { EventBus } from "@devagent/core";

// ─── Types ──────────────────────────────────────────────────

export interface PlanStep {
  readonly description: string;
  readonly status: "pending" | "in_progress" | "completed";
}

export interface Plan {
  readonly steps: ReadonlyArray<PlanStep>;
  readonly explanation: string | null;
}

// ─── Plan State ─────────────────────────────────────────────

// Module-level plan state (per-process, session-scoped)
let currentPlan: Plan | null = null;

export function getCurrentPlan(): Plan | null {
  return currentPlan;
}

export function clearPlan(): void {
  currentPlan = null;
}

// ─── Tool Definition ────────────────────────────────────────

export function createPlanTool(bus: EventBus): ToolSpec {
  return {
    name: "update_plan",
    description:
      "Track progress on multi-step tasks. Create or update a plan with steps (pending/in_progress/completed). Use for tasks with 3+ steps. Keep step descriptions under 7 words. One step should be in_progress at a time.",
    category: "workflow",
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

      let steps: PlanStep[];
      try {
        const parsed = JSON.parse(stepsRaw) as Array<{
          description: string;
          status: string;
        }>;
        steps = parsed.map((s) => ({
          description: s.description,
          status: validateStatus(s.status),
        }));
      } catch {
        return {
          success: false,
          output: "",
          error:
            'Invalid steps JSON. Expected: [{"description": "...", "status": "pending|in_progress|completed"}]',
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

      currentPlan = { steps, explanation };

      // Emit plan event
      bus.emit("plan:updated", {
        steps: steps.map((s) => ({
          description: s.description,
          status: s.status,
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

      const completed = steps.filter((s) => s.status === "completed").length;
      const total = steps.length;

      return {
        success: true,
        output: `Plan updated (${completed}/${total} completed):\n${planDisplay}`,
        error: null,
        artifacts: [],
      };
    },
  };
}

// ─── Helpers ────────────────────────────────────────────────

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
  return "pending";
}
