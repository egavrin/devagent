/**
 * EventLogger — persists EventBus events as JSONL files for post-hoc analysis.
 * One file per session: <logDir>/<sessionId>.jsonl
 */

import { mkdirSync, appendFileSync, readFileSync, readdirSync, statSync, existsSync, writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { EventBus, EventMap } from "./events.js";

// ─── Types ──────────────────────────────────────────────────

export interface LogEntry {
  readonly ts: number;
  readonly event: string;
  readonly sessionId: string;
  readonly data: unknown;
}

// ─── Default log directory ──────────────────────────────────

const DEFAULT_LOG_DIR = join(homedir(), ".config", "devagent", "logs");

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
      "message:assistant",
      "message:tool",
      "message:user",
      "approval:request",
      "approval:response",
      "checkpoint:created",
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
        this.write({
          ts: Date.now(),
          event: eventType,
          sessionId: this.sessionId,
          data,
        });
      });
      this.unsubscribers.push(unsub);
    }
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
}
