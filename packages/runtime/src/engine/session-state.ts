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

import type { PlanStep } from "./plan-tool.js";
import { extractEnvFact } from "./session-state-env-facts.js";
import type { EnvFact } from "./session-state-env-facts.js";
import type { SessionStateConfigCore } from "../core/index.js";

// ─── Constants (defaults) ──────────────────────────────────────

const DEFAULT_MAX_MODIFIED_FILES = 50;
const DEFAULT_MAX_ENV_FACTS = 20;
const DEFAULT_MAX_TOOL_SUMMARIES = 60;
const DEFAULT_MAX_READONLY_COVERAGE_PER_TOOL = 200;
const DEFAULT_MAX_FINDINGS = 20;
const DEFAULT_MAX_KNOWLEDGE = 10;
export const SUMMARY_MAX_CHARS = 2000;
const FINDING_DETAIL_MAX_CHARS = 500;
export const KNOWLEDGE_CONTENT_MAX_CHARS = 1000;

/** Marker prefix used to identify session-state system messages. */
export const SESSION_STATE_MARKER = "[SESSION STATE";
/** Marker prefix for pruned tool output placeholders. */
export const PRUNED_MARKER_PREFIX = "[Previously:";
/** Marker prefix for superseded (deduplicated) tool output. */
export const SUPERSEDED_MARKER_PREFIX = "[Superseded";

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

export interface KnowledgeEntry {
  readonly key: string;
  readonly content: string;
  readonly iteration: number;
}

/** Configuration for SessionState — controls which sections are active.
 *  Extends SessionStateConfigCore from @devagent/runtime with engine-specific
 *  fields, making all inherited optional fields required. */
export interface SessionStateConfig extends Required<SessionStateConfigCore> {
  readonly maxReadonlyCoveragePerTool: number;
  readonly maxKnowledge: number;
}

export const DEFAULT_SESSION_STATE_CONFIG: SessionStateConfig = {
  persist: true,
  trackPlan: true,
  trackFiles: true,
  trackEnv: true,
  trackToolResults: true,
  trackFindings: true,
  trackKnowledge: true,
  maxModifiedFiles: DEFAULT_MAX_MODIFIED_FILES,
  maxEnvFacts: DEFAULT_MAX_ENV_FACTS,
  maxToolSummaries: DEFAULT_MAX_TOOL_SUMMARIES,
  maxReadonlyCoveragePerTool: DEFAULT_MAX_READONLY_COVERAGE_PER_TOOL,
  maxFindings: DEFAULT_MAX_FINDINGS,
  maxKnowledge: DEFAULT_MAX_KNOWLEDGE,
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
  readonly knowledge?: KnowledgeEntry[];
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
  private knowledge: KnowledgeEntry[] = [];

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

  /**
   * Total count of plan steps (completed + in_progress + pending).
   * Returns 0 if no plan is set.
   */
  getTotalPlanCount(): number {
    return this.plan?.length ?? 0;
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

  // ─── Knowledge ─────────────────────────────────────────────

  /**
   * Store a knowledge entry extracted before compaction.
   * Deduplicates by key — re-adding replaces the old entry at the end.
   */
  addKnowledge(key: string, content: string, iteration: number): void {
    if (!this.config.trackKnowledge) return;

    const truncatedContent = content.length > KNOWLEDGE_CONTENT_MAX_CHARS
      ? content.slice(0, KNOWLEDGE_CONTENT_MAX_CHARS)
      : content;

    // Deduplicate by key
    const existingIdx = this.knowledge.findIndex((k) => k.key === key);
    if (existingIdx !== -1) {
      this.knowledge.splice(existingIdx, 1);
    }

    this.knowledge.push({ key, content: truncatedContent, iteration });

    // Enforce cap — drop oldest
    const cap = this.config.maxKnowledge;
    if (this.knowledge.length > cap) {
      this.knowledge = this.knowledge.slice(this.knowledge.length - cap);
    }

    this.autosave();
  }

  /**
   * Return array of knowledge entries.
   */
  getKnowledge(): ReadonlyArray<KnowledgeEntry> {
    return this.knowledge;
  }

  hasContent(): boolean {
    return (
      (this.plan?.length ?? 0) > 0 ||
      this.modifiedFiles.length > 0 ||
      this.envFacts.size > 0 ||
      this.toolSummaries.length > 0 ||
      this.readonlyCoverage.size > 0 ||
      this.findings.length > 0 ||
      this.knowledge.length > 0
    );
  }

  mergeDelegatedState(data: SessionStateJSON): void {
    const child = SessionState.fromJSON(data, this.config);
    this.batch(() => {
      for (const file of child.getModifiedFiles()) {
        this.recordModifiedFile(file);
      }
      for (const [tool, targets] of child.getReadonlyCoverage()) {
        for (const target of targets) {
          this.recordReadonlyCoverage(tool, target);
        }
      }
      for (const finding of child.getFindings()) {
        this.addFinding(finding.title, finding.detail, finding.iteration);
      }
      for (const entry of child.getKnowledge()) {
        this.addKnowledge(entry.key, entry.content, entry.iteration);
      }
      for (const fact of data.envFacts ?? []) {
        this.addEnvFact(fact.key, fact.value);
      }
    });
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
      knowledge: this.knowledge.length > 0 ? this.knowledge.map((k) => ({ ...k })) : undefined,
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
    ss.hydrateFromJSON(data);
    ss.applyCaps();
    return ss;
  }

  private hydrateFromJSON(data: SessionStateJSON): void {
    this.plan = data.plan === null ? null : data.plan.map((s) => ({ ...s }));
    this.modifiedFiles.push(...data.modifiedFiles);

    for (const { key, value } of data.envFacts) this.envFacts.set(key, value);
    for (const summary of data.toolSummaries) this.toolSummaries.push({ ...summary });
    for (const entry of data.readonlyCoverage ?? []) this.hydrateReadonlyCoverage(entry);
    for (const finding of data.findings ?? []) this.findings.push({ ...finding });
    for (const entry of data.knowledge ?? []) this.knowledge.push({ ...entry });
  }

  private hydrateReadonlyCoverage(
    entry: { readonly tool: string; readonly targets: string[] } | undefined,
  ): void {
    if (!entry || typeof entry.tool !== "string" || !Array.isArray(entry.targets)) return;
    this.readonlyCoverage.set(
      entry.tool,
      entry.targets.filter((target): target is string => typeof target === "string"),
    );
  }

  private applyCaps(): void {
    const cfg = this.config;
    this.modifiedFiles = keepLast(this.modifiedFiles, cfg.maxModifiedFiles);
    this.toolSummaries = keepLast(this.toolSummaries, cfg.maxToolSummaries);
    this.findings = keepLast(this.findings, cfg.maxFindings);
    this.knowledge = keepLast(this.knowledge, cfg.maxKnowledge);
    this.trimEnvFacts(cfg.maxEnvFacts);
    this.trimReadonlyCoverage(cfg.maxReadonlyCoveragePerTool);
  }

  private trimEnvFacts(maxEnvFacts: number): void {
    const keys = [...this.envFacts.keys()];
    for (const key of keys.slice(0, Math.max(0, keys.length - maxEnvFacts))) {
      this.envFacts.delete(key);
    }
  }

  private trimReadonlyCoverage(maxTargets: number): void {
    for (const [tool, targets] of this.readonlyCoverage.entries()) {
      this.readonlyCoverage.set(tool, keepLast(targets, maxTargets));
    }
  }

  // ─── System Message Serialization ──────────────────────────

  /**
   * Build the "Completed work (verified)" evidence section.
   * Returns null if no files have been modified.
   *
   * For each modified file (most recent first), finds the most recent
   * toolSummary whose target matches the file path (exact or suffix match)
   * and formats it as a one-line entry with iteration and tool context.
   */
  private buildEvidenceSection(): string | null {
    if (this.modifiedFiles.length === 0) return null;

    const lines: string[] = [];
    for (const file of [...this.modifiedFiles].reverse()) {
      const match = [...this.toolSummaries].reverse().find((s) => {
        const t = s.target.replaceAll("\\", "/");
        const f = file.replaceAll("\\", "/");
        return t === f || f.endsWith("/" + t) || t.endsWith("/" + f);
      });
      if (match) {
        const flat = match.summary.replace(/\n/g, " | ").slice(0, 120);
        lines.push(`- ${file}: ${flat} (iter ${match.iteration}, ${match.tool})`);
      } else {
        lines.push(`- ${file}`);
      }
    }

    return [
      "## Completed work (verified)",
      "IMPORTANT: The files below were physically modified in this session. Do NOT redo this work or reset the plan steps that cover them.",
      lines.join("\n"),
    ].join("\n");
  }

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

    if (tier !== "minimal") {
      const evidenceSection = this.buildEvidenceSection();
      if (evidenceSection) sections.push(evidenceSection);
    }

    pushOptionalSection(sections, this.buildPlanSection());

    if (tier !== "minimal") {
      pushOptionalSection(sections, this.buildModifiedFilesSection());
      pushOptionalSection(sections, this.buildEnvironmentSection());
    }

    pushOptionalSection(sections, this.buildToolSummariesSection(tier));
    pushOptionalSection(sections, this.buildReadonlyCoverageSection(tier));
    pushOptionalSection(sections, this.buildFindingsSection());
    pushOptionalSection(sections, this.buildKnowledgeSection());

    if (sections.length === 0) return null;

    return `${SESSION_STATE_MARKER} — preserved across compaction]\n\n${sections.join("\n\n")}`;
  }

  private buildPlanSection(): string | null {
    if (this.plan === null || this.plan.length === 0) return null;
    const hasCompleted = this.plan.some((s) => s.status === "completed");
    const instruction = hasCompleted
      ? "IMPORTANT: The following plan steps reflect verified progress. Do NOT reset completed steps when calling update_plan. Continue from where the plan left off.\n"
      : "IMPORTANT: This is the active plan. Do NOT replace or rename these steps — only update their statuses (pending → in_progress → completed).\n";
    const lines = this.plan.map((s) => `- [${s.status}] ${s.description}`);
    return `## Plan\n${instruction}${lines.join("\n")}`;
  }

  private buildModifiedFilesSection(): string | null {
    if (this.modifiedFiles.length === 0) return null;
    const lines = this.modifiedFiles.map((f) => `- ${f}`);
    const scopeHint =
      "IMPORTANT: Treat this as the current review scope. Do NOT rerun broad discovery commands (for example `git diff --name-only`) unless this list is stale.\n";
    return `## Modified files\n${scopeHint}${lines.join("\n")}`;
  }

  private buildEnvironmentSection(): string | null {
    if (this.envFacts.size === 0) return null;
    const lines = [...this.envFacts.values()].map((f) => `- ${f}`);
    return `## Environment\n${lines.join("\n")}`;
  }

  private buildToolSummariesSection(tier: "full" | "compact" | "minimal"): string | null {
    if (this.toolSummaries.length === 0) return null;
    const count = getTierLimit(tier, { minimal: 5, compact: 20, full: this.toolSummaries.length });
    const lines = this.toolSummaries.slice(-count).map(formatToolSummaryLine);
    const antiReread = "IMPORTANT: The files listed below have already been examined. Do NOT re-read them unless you need to verify a specific detail not captured in the summary. Use save_finding to persist analysis conclusions before context compaction.\n";
    return `## Recent activity\n${antiReread}${lines.join("\n")}`;
  }

  private buildReadonlyCoverageSection(tier: "full" | "compact" | "minimal"): string | null {
    if (this.readonlyCoverage.size === 0) return null;
    const maxTools = getTierLimit(tier, { minimal: 3, compact: 6, full: 12 });
    const maxTargetsPerTool = getTierLimit(tier, { minimal: 5, compact: 10, full: 25 });
    const lines = formatReadonlyCoverageLines([...this.readonlyCoverage.entries()], {
      maxTools,
      maxTargetsPerTool,
    });
    const totalTargets = this.getReadonlyCoverageTargetCount();
    const instruction = `IMPORTANT: ${totalTargets} readonly inspection(s) below are tracked and will be SKIPPED if re-requested. Do NOT attempt to re-read these files — the results are cached. Only a mutating tool (edit/write) resets the cache for affected paths.\n`;
    return `## Readonly coverage\n${instruction}${lines.join("\n")}`;
  }

  private buildFindingsSection(): string | null {
    if (this.findings.length === 0) return null;
    const lines = this.findings.map((f) => `- **${f.title}**: ${f.detail}`);
    return `## Findings\n${lines.join("\n")}`;
  }

  private buildKnowledgeSection(): string | null {
    if (this.knowledge.length === 0) return null;
    const lines = this.knowledge.map((k) => `- **${k.key}**: ${k.content}`);
    return `## Accumulated knowledge\n${lines.join("\n")}`;
  }
}

function keepLast<T>(items: T[], maxItems: number): T[] {
  return items.length > maxItems ? items.slice(items.length - maxItems) : items;
}

function pushOptionalSection(sections: string[], section: string | null): void {
  if (section) sections.push(section);
}

function getTierLimit(
  tier: "full" | "compact" | "minimal",
  limits: { readonly full: number; readonly compact: number; readonly minimal: number },
): number {
  return limits[tier];
}

function formatToolSummaryLine(summary: ToolResultSummary): string {
  const flatSummary = summary.summary.replace(/\n/g, " | ");
  return `- [iter ${summary.iteration}] ${summary.tool}(${summary.target}): ${flatSummary}`;
}

function formatReadonlyCoverageLines(
  entries: Array<[string, string[]]>,
  limits: { readonly maxTools: number; readonly maxTargetsPerTool: number },
): string[] {
  const visibleTools = entries.slice(-limits.maxTools);
  const lines = visibleTools.map(([tool, targets]) =>
    formatReadonlyCoverageLine(tool, targets, limits.maxTargetsPerTool),
  );
  const hiddenToolCount = entries.length - visibleTools.length;
  if (hiddenToolCount > 0) lines.push(`- ... (+${hiddenToolCount} more tools)`);
  return lines;
}

function formatReadonlyCoverageLine(
  tool: string,
  targets: ReadonlyArray<string>,
  maxTargets: number,
): string {
  const recentTargets = targets.slice(-maxTargets);
  const hiddenTargets = targets.length - recentTargets.length;
  const targetList = recentTargets.join(", ");
  const overflow = hiddenTargets > 0 ? `, ... (+${hiddenTargets} more)` : "";
  return `- ${tool} (${targets.length}): ${targetList}${overflow}`;
}

export { extractEnvFact };
export type { EnvFact };
