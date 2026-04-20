/**
 * EventLogger — persists EventBus events as JSONL files for post-hoc analysis.
 * One file per session: <logDir>/<sessionId>.jsonl
 */

import { mkdirSync, appendFileSync, readFileSync, readdirSync, statSync, existsSync, writeFileSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import type {
  EventBus,
  EventMap,
  SubagentStartEvent,
  SubagentEndEvent,
  SubagentErrorEvent,
} from "./events.js";
import {
  aggregateDelegatedWork,
  formatDuration,
  loggedSubagentRunFromEvent,
} from "./subagent-summary.js";
import type { LoggedSubagentRun, DelegatedWorkSummary } from "./subagent-summary.js";
import { stripToolFileChangePresentationData } from "./tool-file-change.js";
import type { ToolFileChangePreview } from "./types.js";

// ─── Types ──────────────────────────────────────────────────

export interface LogEntry {
  readonly ts: number;
  readonly event: string;
  readonly sessionId: string;
  readonly data: unknown;
}

// ─── Default log directory ──────────────────────────────────

const DEFAULT_LOG_DIR = join(process.env["HOME"] ?? homedir(), ".config", "devagent", "logs");

// ─── EventLogger ────────────────────────────────────────────

export class EventLogger {
  private readonly logPath: string;
  private readonly sessionId: string;
  private closed = false;
  private unsubscribers: Array<() => void> = [];

  constructor(sessionId: string, logDir?: string) {
    this.sessionId = sessionId;
    const dir = logDir ?? DEFAULT_LOG_DIR;
    mkdirSync(dir, { recursive: true });
    this.logPath = join(dir, `${sessionId}.jsonl`);
    // Touch the file to ensure it exists
    if (!existsSync(this.logPath)) {
      writeFileSync(this.logPath, "");
    }
  }

  /**
   * Subscribe to ALL event types on the bus.
   */
  attach(bus: EventBus): void {
    const eventTypes: Array<keyof EventMap> = [
      "tool:before",
      "tool:after",
      "subagent:start",
      "subagent:update",
      "subagent:end",
      "subagent:error",
      "message:assistant",
      "message:tool",
      "message:user",
      "approval:request",
      "approval:response",
      "session:start",
      "session:end",
      "cost:update",
      "plan:updated",
      "context:compacting",
      "context:compacted",
      "iteration:start",
      "plan:regression",
      "error",
    ];

    for (const eventType of eventTypes) {
      const unsub = bus.on(eventType, (data: unknown) => {
        if (eventType === "message:assistant" && (data as EventMap["message:assistant"]).partial) return;
        this.write({
          ts: Date.now(),
          event: eventType,
          sessionId: this.sessionId,
          data: this.sanitizeEventData(eventType, data),
        });
      });
      this.unsubscribers.push(unsub);
    }
  }

  private sanitizeEventData(eventType: keyof EventMap, data: unknown): unknown {
    if (eventType !== "tool:after") return data;
    const event = sanitizeToolAfterEvent(data as EventMap["tool:after"]);
    if (event.name !== "delegate") return event;

    const summary = event.result.metadata?.["delegateSummary"];
    if (!summary || typeof summary !== "object") return data;

    const delegateSummary = summary as Record<string, unknown>;
    const quality = delegateSummary["quality"];
    const detailParts: string[] = [];
    if (typeof delegateSummary["durationMs"] === "number") {
      detailParts.push(formatDuration(delegateSummary["durationMs"]));
    }
    if (typeof delegateSummary["iterations"] === "number") {
      detailParts.push(`${delegateSummary["iterations"]} iterations`);
    }
    if (quality && typeof quality === "object") {
      const qualityRecord = quality as Record<string, unknown>;
      if (typeof qualityRecord["score"] === "number") {
        detailParts.push(`score ${Number(qualityRecord["score"]).toFixed(2)}`);
      }
      if (typeof qualityRecord["completeness"] === "string") {
        detailParts.push(qualityRecord["completeness"]);
      }
    }

    const labelParts: string[] = [];
    if (typeof delegateSummary["agentId"] === "string") labelParts.push(delegateSummary["agentId"]);
    if (typeof delegateSummary["agentType"] === "string") labelParts.push(delegateSummary["agentType"]);
    if (typeof delegateSummary["laneLabel"] === "string" && delegateSummary["laneLabel"].length > 0) {
      labelParts.push(delegateSummary["laneLabel"]);
    }
    const summaryText = `${labelParts.join(" ")} completed${
      detailParts.length > 0 ? ` (${detailParts.join(", ")})` : ""
    }`;

    return {
      ...event,
      result: {
        ...event.result,
        output: summaryText,
        metadata: {
          agentMeta: event.result.metadata?.["agentMeta"],
          delegateSummary: event.result.metadata?.["delegateSummary"],
          quality: event.result.metadata?.["quality"],
        },
      },
    } satisfies EventMap["tool:after"];
  }

  /**
   * Append a single JSONL entry. No-op after close().
   */
  write(entry: LogEntry): void {
    if (this.closed) return;
    appendFileSync(this.logPath, JSON.stringify(entry) + "\n");
  }

  /**
   * Unsubscribe from bus and stop writing.
   */
  close(): void {
    this.closed = true;
    for (const unsub of this.unsubscribers) {
      unsub();
    }
    this.unsubscribers = [];
  }

  /**
   * Read all log entries for a session.
   */
  static readLog(sessionId: string, logDir?: string): LogEntry[] {
    const dir = logDir ?? DEFAULT_LOG_DIR;
    const logPath = join(dir, `${sessionId}.jsonl`);
    if (!existsSync(logPath)) return [];

    const content = readFileSync(logPath, "utf-8").trim();
    if (content.length === 0) return [];

    return content.split("\n").map((line) => JSON.parse(line) as LogEntry);
  }

  /**
   * List all log files in the log directory.
   */
  static listLogs(logDir?: string): Array<{ sessionId: string; createdAt: number; sizeBytes: number }> {
    const dir = logDir ?? DEFAULT_LOG_DIR;
    if (!existsSync(dir)) return [];

    const files = readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    return files.map((f) => {
      const filePath = join(dir, f);
      const stats = statSync(filePath);
      return {
        sessionId: f.replace(/\.jsonl$/, ""),
        createdAt: stats.birthtimeMs,
        sizeBytes: stats.size,
      };
    });
  }

  /**
   * Delete .jsonl files older than retentionDays. Returns count deleted.
   */
  static rotate(retentionDays: number, logDir?: string): number {
    const dir = logDir ?? DEFAULT_LOG_DIR;
    if (!existsSync(dir)) return 0;

    const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000;
    const files = readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    let deleted = 0;

    for (const f of files) {
      const filePath = join(dir, f);
      const stats = statSync(filePath);
      if (stats.mtimeMs < cutoff) {
        unlinkSync(filePath);
        deleted++;
      }
    }

    return deleted;
  }

  /**
   * Total size of all log files in bytes.
   */
  static getTotalSize(logDir?: string): number {
    const dir = logDir ?? DEFAULT_LOG_DIR;
    if (!existsSync(dir)) return 0;

    const files = readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    let total = 0;
    for (const f of files) {
      total += statSync(join(dir, f)).size;
    }
    return total;
  }

  static summarizeSubagents(entries: ReadonlyArray<LogEntry>): DelegatedWorkSummary {
    const runs = new Map<string, LoggedSubagentRun>();

    for (const entry of entries) {
      if (entry.event === "subagent:start") {
        const data = entry.data as SubagentStartEvent;
        runs.set(data.agentId, loggedSubagentRunFromEvent(data, runs.get(data.agentId)));
        continue;
      }

      if (entry.event === "tool:before") {
        const data = entry.data as EventMap["tool:before"];
        if (!data.agentId) continue;
        const run = runs.get(data.agentId);
        if (!run) continue;
        runs.set(data.agentId, {
          ...run,
          toolCalls: run.toolCalls + 1,
        });
        continue;
      }

      if (entry.event === "subagent:end") {
        const data = entry.data as SubagentEndEvent;
        runs.set(data.agentId, loggedSubagentRunFromEvent(data, runs.get(data.agentId)));
        continue;
      }

      if (entry.event === "subagent:error") {
        const data = entry.data as SubagentErrorEvent;
        runs.set(data.agentId, loggedSubagentRunFromEvent(data, runs.get(data.agentId)));
      }
    }

    return aggregateDelegatedWork([...runs.values()]);
  }
}

function sanitizeToolAfterEvent(event: EventMap["tool:after"]): EventMap["tool:after"] {
  const metadata = event.result.metadata;
  const sanitizedMetadata = metadata && typeof metadata === "object"
    ? sanitizeToolResultMetadata(metadata)
    : metadata;

  return {
    ...event,
    ...(Array.isArray(event.fileEdits) ? { fileEdits: sanitizePersistedFileEdits(event.fileEdits) } : {}),
    result: {
      ...event.result,
      ...(sanitizedMetadata !== undefined ? { metadata: sanitizedMetadata } : {}),
    },
  };
}

function sanitizeToolResultMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  let next: Record<string, unknown> = metadata;

  if (Array.isArray(metadata["fileEdits"])) {
    next = {
      ...next,
      fileEdits: sanitizePersistedFileEdits(metadata["fileEdits"]),
    };
  }

  const commandResult = metadata["commandResult"];
  if (commandResult && typeof commandResult === "object") {
    const command = commandResult as Record<string, unknown>;
    next = {
      ...next,
      commandResult: {
        command: command["command"],
        cwd: command["cwd"],
        exitCode: command["exitCode"],
        timedOut: command["timedOut"],
        warningOnly: command["warningOnly"],
        stdoutTruncated: command["stdoutTruncated"],
        stderrTruncated: command["stderrTruncated"],
      },
    };
  }

  const validationResult = metadata["validationResult"];
  if (validationResult && typeof validationResult === "object") {
    const validation = validationResult as Record<string, unknown>;
    const testSummary = validation["testSummary"];
    next = {
      ...next,
      validationResult: {
        passed: validation["passed"],
        diagnosticCount: Array.isArray(validation["diagnosticErrors"]) ? validation["diagnosticErrors"].length : 0,
        testPassed: validation["testPassed"],
        baselineFiltered: validation["baselineFiltered"],
        ...(testSummary && typeof testSummary === "object"
          ? {
            testSummary: {
              framework: (testSummary as Record<string, unknown>)["framework"],
              passed: (testSummary as Record<string, unknown>)["passed"],
              failed: (testSummary as Record<string, unknown>)["failed"],
            },
          }
          : {}),
      },
    };
  }

  return next;
}

function sanitizePersistedFileEdits(fileEdits: ReadonlyArray<unknown>): ReadonlyArray<ToolFileChangePreview> {
  return fileEdits
    .filter(isToolFileChangePreviewLike)
    .map((fileEdit) => stripToolFileChangePresentationData(fileEdit));
}

function isToolFileChangePreviewLike(fileEdit: unknown): fileEdit is ToolFileChangePreview {
  return !!fileEdit
    && typeof fileEdit === "object"
    && typeof (fileEdit as ToolFileChangePreview).path === "string"
    && ((fileEdit as ToolFileChangePreview).kind === "create"
      || (fileEdit as ToolFileChangePreview).kind === "update"
      || (fileEdit as ToolFileChangePreview).kind === "delete"
      || (fileEdit as ToolFileChangePreview).kind === "move")
    && typeof (fileEdit as ToolFileChangePreview).additions === "number"
    && typeof (fileEdit as ToolFileChangePreview).deletions === "number"
    && typeof (fileEdit as ToolFileChangePreview).unifiedDiff === "string"
    && typeof (fileEdit as ToolFileChangePreview).truncated === "boolean";
}
