/**
 * Session timeline inspection — reads JSONL logs and renders a timeline.
 * Used by: `devagent session inspect <session-id>`
 */

import type { LogEntry } from "@devagent/core";
import { truncate, formatDuration } from "./format.js";

// ─── Types ──────────────────────────────────────────────────

export interface TimelineEntry {
  readonly timestamp: number;
  readonly type: "iteration" | "tool" | "compaction" | "plan" | "cost" | "error";
  readonly label: string;
  readonly durationMs?: number;
  readonly metadata?: Record<string, unknown>;
}

// ─── Timeline Building ─────────────────────────────────────

export function buildTimeline(entries: LogEntry[]): TimelineEntry[] {
  if (entries.length === 0) return [];

  const timeline: TimelineEntry[] = [];
  const pendingTools = new Map<string, { name: string; ts: number }>();

  for (const entry of entries) {
    switch (entry.event) {
      case "iteration:start": {
        const data = entry.data as Record<string, unknown>;
        timeline.push({
          timestamp: entry.ts,
          type: "iteration",
          label: `Iteration ${data["iteration"] ?? "?"}`,
        });
        break;
      }

      case "tool:before": {
        const data = entry.data as Record<string, unknown>;
        const callId = data["callId"] as string | undefined;
        const name = (data["name"] as string) ?? "unknown";
        if (callId) {
          pendingTools.set(callId, { name, ts: entry.ts });
        }
        // Summarize params
        const params = data["params"] as Record<string, unknown> | undefined;
        let paramSummary = "";
        if (params) {
          const firstStr = Object.values(params).find((v) => typeof v === "string") as string | undefined;
          if (firstStr) {
            paramSummary = ` "${truncate(firstStr, 30)}"`;

          }
        }
        // Don't push yet — wait for tool:after to compute duration
        if (!callId) {
          timeline.push({
            timestamp: entry.ts,
            type: "tool",
            label: `${name}${paramSummary}`,
          });
        }
        break;
      }

      case "tool:after": {
        const data = entry.data as Record<string, unknown>;
        const callId = data["callId"] as string | undefined;
        const name = (data["name"] as string) ?? "unknown";
        const durationMs = data["durationMs"] as number | undefined;
        const pending = callId ? pendingTools.get(callId) : undefined;

        if (pending) {
          pendingTools.delete(callId!);
          // Reconstruct param summary from before event name
          timeline.push({
            timestamp: pending.ts,
            type: "tool",
            label: pending.name,
            durationMs: durationMs ?? (entry.ts - pending.ts),
          });
        } else {
          timeline.push({
            timestamp: entry.ts,
            type: "tool",
            label: name,
            durationMs,
          });
        }
        break;
      }

      case "plan:updated": {
        const data = entry.data as Record<string, unknown>;
        const steps = data["steps"] as Array<{ status: string }> | undefined;
        if (steps) {
          const completed = steps.filter((s) => s.status === "completed").length;
          timeline.push({
            timestamp: entry.ts,
            type: "plan",
            label: `Plan: ${completed}/${steps.length} completed`,
          });
        }
        break;
      }

      case "context:compacted": {
        const data = entry.data as Record<string, unknown>;
        const before = data["tokensBefore"] as number | undefined;
        const after = data["estimatedTokens"] as number | undefined;
        const kBefore = before ? `${Math.round(before / 1000)}k` : "?";
        const kAfter = after ? `${Math.round(after / 1000)}k` : "?";
        timeline.push({
          timestamp: entry.ts,
          type: "compaction",
          label: `Context compacted: ${kBefore} → ${kAfter} tokens`,
        });
        break;
      }

      case "cost:update": {
        const data = entry.data as Record<string, unknown>;
        const cost = data["totalCost"] as number | undefined;
        if (cost && cost > 0) {
          timeline.push({
            timestamp: entry.ts,
            type: "cost",
            label: `Cost: $${cost.toFixed(4)}`,
            metadata: data as Record<string, unknown>,
          });
        }
        break;
      }

      case "error": {
        const data = entry.data as Record<string, unknown>;
        const msg = (data["message"] as string) ?? "Unknown error";
        timeline.push({
          timestamp: entry.ts,
          type: "error",
          label: truncate(msg, 60),
        });
        break;
      }
    }
  }

  // Sort by timestamp
  timeline.sort((a, b) => a.timestamp - b.timestamp);
  return timeline;
}

// ─── Timeline Rendering ────────────────────────────────────

const ICONS: Record<string, string> = {
  iteration: "●",
  tool: "◆",
  compaction: "◎",
  plan: "▸",
  cost: "$",
  error: "✗",
};

export function renderTimeline(timeline: TimelineEntry[]): string {
  if (timeline.length === 0) return "No events found.";

  const baseTs = timeline[0]!.timestamp;
  const lines: string[] = [];

  for (const entry of timeline) {
    const offset = (entry.timestamp - baseTs) / 1000;
    const offsetStr = `+${offset.toFixed(1)}s`.padEnd(13);
    const icon = ICONS[entry.type] ?? "·";
    const durStr = entry.durationMs !== undefined
      ? ` (${formatDuration(entry.durationMs)})`
      : "";
    lines.push(`${offsetStr}${icon} ${entry.label}${durStr}`);
  }

  return lines.join("\n");
}
