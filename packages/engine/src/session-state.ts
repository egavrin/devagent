/**
 * SessionState — Structured facts that survive context compaction.
 *
 * An in-memory sidecar that stores plan progress, modified files,
 * environment facts, and tool result summaries. On compaction the
 * TaskLoop injects the serialized state as a system message so the
 * next context window starts with full situational awareness.
 *
 * Supports optional disk persistence via SessionStatePersistence —
 * every mutation auto-saves when bound to a persistence backend.
 */

import type { SessionStateConfigCore } from "@devagent/core";
import type { PlanStep } from "./plan-tool.js";

// ─── Constants (defaults) ──────────────────────────────────────

const DEFAULT_MAX_MODIFIED_FILES = 50;
const DEFAULT_MAX_ENV_FACTS = 20;
const DEFAULT_MAX_TOOL_SUMMARIES = 30;
const DEFAULT_MAX_READONLY_COVERAGE_PER_TOOL = 200;
const DEFAULT_MAX_FINDINGS = 20;
export const SUMMARY_MAX_CHARS = 2000;
export const FINDING_DETAIL_MAX_CHARS = 500;

/** Marker prefix used to identify session-state system messages. */
export const SESSION_STATE_MARKER = "[SESSION STATE";
/** Marker prefix for pruned tool output placeholders. */
export const PRUNED_MARKER_PREFIX = "[Previously:";
/** Marker prefix for superseded (deduplicated) tool output. */
export const SUPERSEDED_MARKER_PREFIX = "[Superseded";

export interface EnvFact {
  readonly key: string;
  readonly message: string;
}

export interface ToolResultSummary {
  readonly tool: string;
  readonly target: string;
  readonly summary: string;
  readonly iteration: number;
}

export interface Finding {
  readonly title: string;
  readonly detail: string;
  readonly iteration: number;
}

/** Configuration for SessionState — controls which sections are active.
 *  Extends SessionStateConfigCore from @devagent/core with engine-specific
 *  fields, making all inherited optional fields required. */
export interface SessionStateConfig extends Required<SessionStateConfigCore> {
  readonly maxReadonlyCoveragePerTool: number;
}

export const DEFAULT_SESSION_STATE_CONFIG: SessionStateConfig = {
  persist: true,
  trackPlan: true,
  trackFiles: true,
  trackEnv: true,
  trackToolResults: true,
  trackFindings: true,
  maxModifiedFiles: DEFAULT_MAX_MODIFIED_FILES,
  maxEnvFacts: DEFAULT_MAX_ENV_FACTS,
  maxToolSummaries: DEFAULT_MAX_TOOL_SUMMARIES,
  maxReadonlyCoveragePerTool: DEFAULT_MAX_READONLY_COVERAGE_PER_TOOL,
  maxFindings: DEFAULT_MAX_FINDINGS,
};

/** Serialized form for disk persistence (JSON round-trip safe). */
export interface SessionStateJSON {
  readonly version: 1;
  readonly plan: PlanStep[] | null;
  readonly modifiedFiles: string[];
  readonly envFacts: Array<{ key: string; value: string }>;
  readonly toolSummaries: ToolResultSummary[];
  readonly readonlyCoverage?: Array<{ tool: string; targets: string[] }>;
  readonly findings?: Finding[];
}

/** Persistence interface — decouples SessionState from any specific storage. */
export interface SessionStatePersistence {
  save(sessionId: string, state: SessionStateJSON): void;
  load(sessionId: string): SessionStateJSON | null;
}

// ─── SessionState ──────────────────────────────────────────────

export class SessionState {
  private plan: PlanStep[] | null = null;
  private modifiedFiles: string[] = [];
  private envFacts: Map<string, string> = new Map();
  private toolSummaries: ToolResultSummary[] = [];
  private readonlyCoverage: Map<string, string[]> = new Map();
  private findings: Finding[] = [];

  private persistence: SessionStatePersistence | null = null;
  private sessionId: string | null = null;
  private readonly config: SessionStateConfig;

  constructor(config?: Partial<SessionStateConfig>) {
    this.config = { ...DEFAULT_SESSION_STATE_CONFIG, ...config };
  }

  // ─── Persistence Binding ─────────────────────────────────────

  /**
   * Bind to a persistence backend. Once bound, every mutation auto-saves.
   */
  bind(sessionId: string, persistence: SessionStatePersistence): void {
    this.sessionId = sessionId;
    this.persistence = persistence;
  }

  /**
   * Load from persistence if available, or create fresh.
   * Auto-binds to the given persistence for ongoing saves.
   */
  static loadOrCreate(
    sessionId: string,
    persistence: SessionStatePersistence,
    config?: Partial<SessionStateConfig>,
  ): SessionState {
    const data = persistence.load(sessionId);
    const ss = data ? SessionState.fromJSON(data, config) : new SessionState(config);
    ss.bind(sessionId, persistence);
    return ss;
  }

  private batchDepth = 0;
  private batchDirty = false;

  /**
   * Group multiple mutations into a single autosave.
   * Nested calls are supported; only the outermost flush triggers a write.
   */
  batch(fn: () => void): void {
    this.batchDepth++;
    try {
      fn();
    } finally {
      this.batchDepth--;
      if (this.batchDepth === 0 && this.batchDirty) {
        this.batchDirty = false;
        this.flushSave();
      }
    }
  }

  private autosave(): void {
    if (this.batchDepth > 0) {
      this.batchDirty = true;
      return;
    }
    this.flushSave();
  }

  private flushSave(): void {
    if (!this.config.persist) return;
    if (this.persistence && this.sessionId) {
      this.persistence.save(this.sessionId, this.toJSON());
    }
  }

  // ─── Plan ──────────────────────────────────────────────────

  /**
   * Store plan steps. Overwrites any previous plan.
   */
  setPlan(steps: PlanStep[]): void {
    if (!this.config.trackPlan) return;
    this.plan = steps.map((s) => ({ ...s }));
    this.autosave();
  }

  /**
   * Return current plan or null if none set.
   */
  getPlan(): ReadonlyArray<PlanStep> | null {
    return this.plan;
  }

  /**
   * True if the plan has any pending or in_progress steps.
   */
  hasPendingPlanSteps(): boolean {
    return this.plan != null && this.plan.some(
      (s) => s.status === "pending" || s.status === "in_progress",
    );
  }

  /**
   * Count of completed plan steps (0 if no plan).
   */
  getPlanCompletedCount(): number {
    if (!this.plan) return 0;
    return this.plan.filter((s) => s.status === "completed").length;
  }

  // ─── Modified Files ────────────────────────────────────────

  /**
   * Record a modified file path. Deduplicates automatically.
   * When the cap is reached, the oldest entries are dropped.
   */
  recordModifiedFile(filePath: string): void {
    if (!this.config.trackFiles) return;

    // Deduplicate: reinsert at the tail so recency reflects
    // latest re-modification of the same file.
    const idx = this.modifiedFiles.indexOf(filePath);
    if (idx !== -1) {
      this.modifiedFiles.splice(idx, 1);
      this.modifiedFiles.push(filePath);
      this.autosave();
      return;
    }

    this.modifiedFiles.push(filePath);

    // Enforce cap — drop oldest entries
    const cap = this.config.maxModifiedFiles;
    if (this.modifiedFiles.length > cap) {
      this.modifiedFiles = this.modifiedFiles.slice(
        this.modifiedFiles.length - cap,
      );
    }

    this.autosave();
  }

  /**
   * Return array of unique modified file paths.
   */
  getModifiedFiles(): string[] {
    return [...this.modifiedFiles];
  }

  // ─── Environment Facts ─────────────────────────────────────

  /**
   * Store an environment fact, deduplicated by key.
   * When the cap is reached, the oldest entries are dropped.
   */
  addEnvFact(key: string, fact: string): void {
    if (!this.config.trackEnv) return;

    // If the key already exists, delete it first so reinsertion
    // moves it to the end (preserving insertion order).
    if (this.envFacts.has(key)) {
      this.envFacts.delete(key);
    }

    this.envFacts.set(key, fact);

    // Enforce cap — drop oldest entries
    if (this.envFacts.size > this.config.maxEnvFacts) {
      const firstKey = this.envFacts.keys().next().value!;
      this.envFacts.delete(firstKey);
    }

    this.autosave();
  }

  /**
   * Return array of fact strings (values only, keys are internal).
   */
  getEnvFacts(): string[] {
    return [...this.envFacts.values()];
  }

  // ─── Tool Result Summaries ───────────────────────────────────

  /**
   * Record a summary of a tool result for post-compaction context.
   */
  addToolSummary(summary: ToolResultSummary): void {
    if (!this.config.trackToolResults) return;

    // Deep-clone and truncate summary text if needed
    const truncated: ToolResultSummary = {
      ...summary,
      summary: summary.summary.length > SUMMARY_MAX_CHARS
        ? summary.summary.slice(0, SUMMARY_MAX_CHARS)
        : summary.summary,
    };

    // Deduplicate by tool+target: if the same tool was called on the same
    // target, replace the old entry so the summary list stays diverse.
    // This prevents repeated run_command (typecheck/test) from crowding out
    // read_file/git_diff entries that carry more useful context after compaction.
    const existingIdx = this.toolSummaries.findIndex(
      (s) => s.tool === truncated.tool && s.target === truncated.target,
    );
    if (existingIdx !== -1) {
      this.toolSummaries.splice(existingIdx, 1);
    }

    this.toolSummaries.push(truncated);

    const cap = this.config.maxToolSummaries;
    if (this.toolSummaries.length > cap) {
      this.toolSummaries = this.toolSummaries.slice(
        this.toolSummaries.length - cap,
      );
    }

    this.autosave();
  }

  /**
   * Return array of tool result summaries.
   */
  getToolSummaries(): ReadonlyArray<ToolResultSummary> {
    return this.toolSummaries;
  }

  /**
   * Count of tool result summaries (avoids cloning).
   */
  getToolSummariesCount(): number {
    return this.toolSummaries.length;
  }

  /**
   * Track successful readonly coverage independently from tool summary eviction.
   * Deduplicates by target per tool and preserves recency order.
   */
  recordReadonlyCoverage(tool: string, target: string): void {
    if (!this.config.trackToolResults) return;

    const toolKey = tool.trim();
    const targetKey = target.trim();
    if (toolKey.length === 0 || targetKey.length === 0) return;

    const existing = this.readonlyCoverage.get(toolKey) ?? [];
    const idx = existing.indexOf(targetKey);
    if (idx !== -1) {
      existing.splice(idx, 1);
    }
    existing.push(targetKey);

    const cap = this.config.maxReadonlyCoveragePerTool;
    if (existing.length > cap) {
      this.readonlyCoverage.set(
        toolKey,
        existing.slice(existing.length - cap),
      );
    } else {
      this.readonlyCoverage.set(toolKey, existing);
    }

    this.autosave();
  }

  /**
   * Return readonly coverage map.
   */
  getReadonlyCoverage(): ReadonlyMap<string, ReadonlyArray<string>> {
    return this.readonlyCoverage;
  }

  /**
   * Total count of readonly coverage targets across all tools (avoids cloning).
   */
  getReadonlyCoverageTargetCount(): number {
    let count = 0;
    for (const targets of this.readonlyCoverage.values()) {
      count += targets.length;
    }
    return count;
  }

  // ─── Findings ───────────────────────────────────────────────

  /**
   * Store an analysis finding that should survive compaction.
   * Deduplicates by title — re-adding moves it to the end with updated detail.
   */
  addFinding(title: string, detail: string, iteration: number): void {
    if (!this.config.trackFindings) return;

    const truncatedDetail = detail.length > FINDING_DETAIL_MAX_CHARS
      ? detail.slice(0, FINDING_DETAIL_MAX_CHARS)
      : detail;

    // Deduplicate by title
    const existingIdx = this.findings.findIndex((f) => f.title === title);
    if (existingIdx !== -1) {
      this.findings.splice(existingIdx, 1);
    }

    this.findings.push({ title, detail: truncatedDetail, iteration });

    // Enforce cap — drop oldest
    const cap = this.config.maxFindings;
    if (this.findings.length > cap) {
      this.findings = this.findings.slice(this.findings.length - cap);
    }

    this.autosave();
  }

  /**
   * Return array of findings.
   */
  getFindings(): ReadonlyArray<Finding> {
    return this.findings;
  }

  /**
   * Count of findings (avoids cloning).
   */
  getFindingsCount(): number {
    return this.findings.length;
  }

  // ─── JSON Serialization ────────────────────────────────────

  /**
   * Serialize to a plain JSON-safe object for disk persistence.
   */
  toJSON(): SessionStateJSON {
    const base: SessionStateJSON = {
      version: 1,
      plan: this.plan ? this.plan.map((s) => ({ ...s })) : null,
      modifiedFiles: [...this.modifiedFiles],
      envFacts: [...this.envFacts.entries()].map(([key, value]) => ({ key, value })),
      toolSummaries: this.toolSummaries.map((s) => ({ ...s })),
      findings: this.findings.length > 0 ? this.findings.map((f) => ({ ...f })) : undefined,
    };
    if (this.readonlyCoverage.size > 0) {
      return {
        ...base,
        readonlyCoverage: [...this.readonlyCoverage.entries()].map(([tool, targets]) => ({
          tool,
          targets: [...targets],
        })),
      };
    }
    return base;
  }

  /**
   * Restore from a serialized JSON object.
   * Throws on unsupported version (fail-fast).
   */
  static fromJSON(
    data: SessionStateJSON,
    config?: Partial<SessionStateConfig>,
  ): SessionState {
    if (data.version !== 1) {
      throw new Error(`Unsupported SessionState version: ${data.version}`);
    }

    const ss = new SessionState(config);

    if (data.plan) {
      ss.plan = data.plan.map((s) => ({ ...s }));
    }
    for (const f of data.modifiedFiles ?? []) {
      ss.modifiedFiles.push(f);
    }
    for (const { key, value } of data.envFacts ?? []) {
      ss.envFacts.set(key, value);
    }
    for (const summary of data.toolSummaries ?? []) {
      ss.toolSummaries.push({ ...summary });
    }
    for (const entry of data.readonlyCoverage ?? []) {
      if (!entry || typeof entry.tool !== "string" || !Array.isArray(entry.targets)) continue;
      ss.readonlyCoverage.set(entry.tool, entry.targets.filter((t): t is string => typeof t === "string"));
    }
    for (const finding of data.findings ?? []) {
      ss.findings.push({ ...finding });
    }

    // Re-apply caps: persisted data may exceed configured limits
    // (e.g., config was tightened after a prior session).
    const cfg = ss.config;
    if (ss.modifiedFiles.length > cfg.maxModifiedFiles) {
      ss.modifiedFiles = ss.modifiedFiles.slice(ss.modifiedFiles.length - cfg.maxModifiedFiles);
    }
    if (ss.envFacts.size > cfg.maxEnvFacts) {
      const keys = [...ss.envFacts.keys()];
      for (let i = 0; i < keys.length - cfg.maxEnvFacts; i++) {
        ss.envFacts.delete(keys[i]!);
      }
    }
    if (ss.toolSummaries.length > cfg.maxToolSummaries) {
      ss.toolSummaries = ss.toolSummaries.slice(ss.toolSummaries.length - cfg.maxToolSummaries);
    }
    for (const [tool, targets] of ss.readonlyCoverage.entries()) {
      if (targets.length > cfg.maxReadonlyCoveragePerTool) {
        ss.readonlyCoverage.set(
          tool,
          targets.slice(targets.length - cfg.maxReadonlyCoveragePerTool),
        );
      }
    }
    if (ss.findings.length > cfg.maxFindings) {
      ss.findings = ss.findings.slice(ss.findings.length - cfg.maxFindings);
    }

    return ss;
  }

  // ─── System Message Serialization ──────────────────────────

  /**
   * Serialize state into a system message for injection after compaction.
   * Returns null if the state is completely empty.
   *
   * Tier controls verbosity:
   * - "full": all sections including tool summaries
   * - "compact": plan + modified files + env facts + compact recent activity
   * - "minimal": plan + compact recent activity
   */
  toSystemMessage(tier: "full" | "compact" | "minimal" = "full"): string | null {
    const sections: string[] = [];

    // Plan section (always included if present)
    if (this.plan !== null && this.plan.length > 0) {
      const hasCompleted = this.plan.some((s) => s.status === "completed");
      const instruction = hasCompleted
        ? "IMPORTANT: The following plan steps reflect verified progress. Do NOT reset completed steps when calling update_plan. Continue from where the plan left off.\n"
        : "";
      const lines = this.plan.map(
        (s) => `- [${s.status}] ${s.description}`,
      );
      sections.push(`## Plan\n${instruction}${lines.join("\n")}`);
    }

    if (tier !== "minimal") {
      // Modified files section
      if (this.modifiedFiles.length > 0) {
        const files = this.modifiedFiles;
        const lines = files.map((f) => `- ${f}`);
        const scopeHint =
          "IMPORTANT: Treat this as the current review scope. Do NOT rerun broad discovery commands (for example `git diff --name-only`) unless this list is stale.\n";
        sections.push(`## Modified files\n${scopeHint}${lines.join("\n")}`);
      }

      // Environment section
      if (this.envFacts.size > 0) {
        const lines = [...this.envFacts.values()].map((f) => `- ${f}`);
        sections.push(`## Environment\n${lines.join("\n")}`);
      }
    }

    // Tool summaries section — included in all tiers so compaction
    // never drops critical "already reviewed" context.
    if (this.toolSummaries.length > 0) {
      const count = tier === "minimal" ? 5
        : tier === "compact" ? 20
        : this.toolSummaries.length;
      const recent = this.toolSummaries.slice(-count);
      const lines = recent.map(
        (s) => {
          const flatSummary = s.summary.replace(/\n/g, " | ");
          return `- [iter ${s.iteration}] ${s.tool}(${s.target}): ${flatSummary}`;
        },
      );
      const antiReread = "IMPORTANT: The files listed below have already been examined. Do NOT re-read them unless you need to verify a specific detail not captured in the summary. Use save_finding to persist analysis conclusions before context compaction.\n";
      sections.push(`## Recent activity\n${antiReread}${lines.join("\n")}`);
    }

    // Readonly coverage section — compact index of successfully reviewed
    // targets that survives tool-summary eviction.
    if (this.readonlyCoverage.size > 0) {
      const maxTools = tier === "minimal" ? 3
        : tier === "compact" ? 6
        : 12;
      const maxTargetsPerTool = tier === "minimal" ? 5
        : tier === "compact" ? 10
        : 25;
      const entries = [...this.readonlyCoverage.entries()];
      const visibleTools = entries.slice(-maxTools);
      const hiddenToolCount = entries.length - visibleTools.length;

      const lines = visibleTools.map(([tool, targets]) => {
        const recentTargets = targets.slice(-maxTargetsPerTool);
        const hiddenTargets = targets.length - recentTargets.length;
        const targetList = recentTargets.join(", ");
        const overflow = hiddenTargets > 0 ? `, ... (+${hiddenTargets} more)` : "";
        return `- ${tool} (${targets.length}): ${targetList}${overflow}`;
      });
      if (hiddenToolCount > 0) {
        lines.push(`- ... (+${hiddenToolCount} more tools)`);
      }

      const totalTargets = this.getReadonlyCoverageTargetCount();
      const instruction = `IMPORTANT: ${totalTargets} readonly inspection(s) below are tracked and will be SKIPPED if re-requested. Do NOT attempt to re-read these files — the results are cached. Only a mutating tool (edit/write) resets the cache for affected paths.\n`;
      sections.push(`## Readonly coverage\n${instruction}${lines.join("\n")}`);
    }

    // Findings section — always included at all tiers (these are the LLM's
    // persisted analysis conclusions and MUST survive compaction).
    if (this.findings.length > 0) {
      const lines = this.findings.map(
        (f) => `- **${f.title}**: ${f.detail}`,
      );
      sections.push(`## Findings\n${lines.join("\n")}`);
    }

    if (sections.length === 0) return null;

    return `${SESSION_STATE_MARKER} — preserved across compaction]\n\n${sections.join("\n\n")}`;
  }
}

// ─── extractEnvFact ─────────────────────────────────────────────

/**
 * Heuristic extraction of structured environment facts from tool failure output.
 * Returns null if no actionable fact can be extracted.
 */
export function extractEnvFact(
  toolName: string,
  error: string,
  output: string,
): EnvFact | null {
  // Only extract from run_command failures
  if (toolName !== "run_command") return null;

  const combined = `${error}\n${output}`;

  // Pattern: command not found (exit code 127)
  const cmdNotFound = combined.match(/(?:command not found|not found):\s*(\S+)/i)
    ?? combined.match(/(\S+):\s*(?:command not found|No such file)/i);
  if (cmdNotFound?.[1]) {
    const cmd = cmdNotFound[1];
    return {
      key: `cmd-not-found:${cmd}`,
      message: `${cmd} is not installed on this system. Use an alternative command.`,
    };
  }

  // Pattern: permission denied
  const permDenied = combined.match(/(\S+):\s*[Pp]ermission denied/);
  if (permDenied?.[1]) {
    return {
      key: `permission-denied:${permDenied[1]}`,
      message: `Permission denied for ${permDenied[1]}. Check file permissions or use sudo.`,
    };
  }

  // Pattern: build tool failure with missing crate/module/package
  const missingDep = combined.match(/can't find crate for `([^`]+)`/)
    ?? combined.match(/ModuleNotFoundError:\s*No module named '([^']+)'/)
    ?? combined.match(/Cannot find module '([^']+)'/);
  if (missingDep?.[1]) {
    return {
      key: `build-fail:missing-${missingDep[1]}`,
      message: `Build fails — missing dependency: ${missingDep[1]}. Install it or skip build verification.`,
    };
  }

  // Pattern: exit code 101 (Rust/cargo) or specific build codes
  if (combined.includes("exit code: 101") || combined.includes("Exit code: 101")) {
    if (combined.includes("cargo") || combined.includes("rustc") || combined.includes("error[E")) {
      return {
        key: "build-fail:cargo",
        message: "cargo check/build fails in this environment. Skip cargo verification or fix native dependencies first.",
      };
    }
  }

  // Pattern: network / proxy failures
  if (/Could not resolve host|Connection refused|ETIMEDOUT|ECONNREFUSED/i.test(combined)) {
    return {
      key: "network-failure",
      message: "Network access appears unavailable or restricted. Use offline alternatives.",
    };
  }

  // Pattern: disk space / quota
  if (/No space left on device|Disk quota exceeded/i.test(combined)) {
    return {
      key: "disk-full",
      message: "Disk is full or quota exceeded. Free space before writing files.",
    };
  }

  // Pattern: runtime version mismatch
  const versionMismatch = combined.match(/requires Node\.js (\d+)/)
    ?? combined.match(/requires Python (\d+\.\d+)/);
  if (versionMismatch) {
    return {
      key: "version-mismatch",
      message: `Runtime version mismatch detected: ${versionMismatch[0]}. Use compatible syntax or check version.`,
    };
  }

  // Pattern: git state issues
  const gitIssue = combined.match(/fatal: not a git repository/)
    ?? combined.match(/CONFLICT.*Merge conflict/i);
  if (gitIssue) {
    return {
      key: "git-issue",
      message: `Git state issue: ${gitIssue[0]}. Resolve before proceeding with file operations.`,
    };
  }

  // Pattern: tool-specific timeout
  if (combined.includes("timed out") || combined.includes("SIGTERM")) {
    return {
      key: "tool-timeout",
      message: "Command timed out. Use shorter-running commands or increase timeout.",
    };
  }

  return null;
}
