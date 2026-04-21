import { formatDuration } from "@devagent/runtime";

import { bold, dim, green, red, yellow } from "./format-colors.js";

interface SubagentPanelData {
  readonly agentId: string;
  readonly agentType: string;
  readonly laneLabel?: string | null;
  readonly model: string;
  readonly reasoningEffort?: string;
  readonly status: "running" | "completed" | "error";
  readonly currentIteration: number;
  readonly startedAtMs: number;
  readonly durationMs?: number;
  readonly currentActivity: string;
  readonly recentActivity: ReadonlyArray<string>;
  readonly quality?: {
    readonly score: number;
    readonly completeness: string;
  };
}

export class SubagentPanelRenderer {
  private static readonly REDRAW_DEBOUNCE_MS = 100;
  private readonly enabled: boolean;
  private panels: ReadonlyArray<SubagentPanelData> = [];
  private renderedLineCount = 0;
  private hidden = false;
  private redrawTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(enabled: boolean) {
    this.enabled = enabled && !!process.stderr.isTTY;
  }

  get active(): boolean {
    return this.enabled;
  }

  setPanels(panels: ReadonlyArray<SubagentPanelData>): void {
    this.panels = panels;
    if (!this.enabled || this.hidden) return;
    if (panels.length === 0) {
      this.clearPendingRedraw();
      this.redraw();
      return;
    }
    this.scheduleRedraw();
  }

  suspend(): void {
    if (!this.enabled || this.hidden) return;
    this.clearPendingRedraw();
    this.clearRendered();
    this.hidden = true;
  }

  resume(): void {
    if (!this.enabled) return;
    this.hidden = false;
  }

  cleanup(): void {
    if (!this.enabled) return;
    this.clearPendingRedraw();
    this.clearRendered();
    this.hidden = false;
    this.panels = [];
  }

  formatPanels(panels: ReadonlyArray<SubagentPanelData>, now = Date.now()): string[] {
    return panels.flatMap((panel) => formatSubagentPanel(panel, now));
  }

  private redraw(): void {
    this.clearPendingRedraw();
    if (this.hidden) return;
    this.clearRendered();
    const lines = this.formatPanels(this.panels);
    if (lines.length === 0) return;
    process.stderr.write(lines.join("\n") + "\n");
    this.renderedLineCount = lines.length;
  }

  private scheduleRedraw(): void {
    if (this.redrawTimer) return;
    this.redrawTimer = setTimeout(() => {
      this.redrawTimer = null;
      this.redraw();
    }, SubagentPanelRenderer.REDRAW_DEBOUNCE_MS);
  }

  private clearPendingRedraw(): void {
    if (!this.redrawTimer) return;
    clearTimeout(this.redrawTimer);
    this.redrawTimer = null;
  }

  private clearRendered(): void {
    if (!this.enabled || this.renderedLineCount === 0) return;
    process.stderr.write(`\x1b[${this.renderedLineCount}F`);
    for (let i = 0; i < this.renderedLineCount; i++) {
      process.stderr.write("\x1b[2K");
      if (i < this.renderedLineCount - 1) {
        process.stderr.write("\x1b[1E");
      }
    }
    if (this.renderedLineCount > 1) {
      process.stderr.write(`\x1b[${this.renderedLineCount - 1}F`);
    }
    this.renderedLineCount = 0;
  }
}

function formatSubagentPanel(panel: SubagentPanelData, now: number): string[] {
  return [
    formatSubagentPanelHeading(panel),
    `    ${formatSubagentPanelStatus(panel, now).join("  ")}`,
    `    ${truncate(formatSubagentPanelActivity(panel), 120)}`,
  ];
}

function formatSubagentPanelHeading(panel: SubagentPanelData): string {
  const lane = panel.laneLabel ? ` ${dim(panel.laneLabel)}` : "";
  const modelBits = [panel.model, panel.reasoningEffort].filter(Boolean).join(", ");
  const model = modelBits ? ` ${dim(`(${modelBits})`)}` : "";
  return `${dim("  ↳")} ${bold(`Subagent ${panel.agentId}`)} ${dim(panel.agentType)}${lane}${model}`;
}

function formatSubagentPanelStatus(panel: SubagentPanelData, now: number): string[] {
  const elapsedMs = panel.status === "running"
    ? Math.max(0, now - panel.startedAtMs)
    : (panel.durationMs ?? Math.max(0, now - panel.startedAtMs));
  return [
    formatSubagentPanelStatusLabel(panel.status),
    ...(panel.currentIteration > 0 ? [dim(`iter ${panel.currentIteration}`)] : []),
    dim(formatDuration(elapsedMs)),
    ...(panel.quality
      ? [dim(`score ${panel.quality.score.toFixed(2)}`), dim(panel.quality.completeness)]
      : []),
  ];
}

function formatSubagentPanelStatusLabel(status: SubagentPanelData["status"]): string {
  switch (status) {
    case "running":
      return yellow("running");
    case "completed":
      return green("completed");
    case "error":
      return red("failed");
  }
}

function formatSubagentPanelActivity(panel: SubagentPanelData): string {
  return panel.recentActivity.length > 0
    ? `Recent: ${panel.recentActivity.slice(0, 2).join(" • ")}`
    : panel.currentActivity;
}

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.substring(0, maxLen - 1) + "…";
}
