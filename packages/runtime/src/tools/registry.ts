/**
 * Tool registry — manages tool registration, lookup, and deferred loading.
 *
 * Tools can be registered as:
 * - **Loaded**: Full ToolSpec, always included in LLM calls
 * - **Deferred**: Stub only (name + description), resolved on demand via tool_search
 *
 * Deferred loading reduces prompt size by ~30-50% for iterations that
 * don't use specialized tools (git, LSP, diagnostics, etc.).
 *
 * Fail fast: throws on duplicate names or missing tools.
 */

import { ToolNotFoundError } from "../core/errors.js";
import type { ToolSpec, ToolCategory } from "../core/types.js";

// ─── Types ──────────────────────────────────────────────────

/** Lightweight stub for deferred tools — name + description only. */
export interface DeferredToolStub {
  readonly name: string;
  readonly description: string;
  readonly category: ToolCategory;
}

interface ScoredDeferredTool {
  readonly stub: DeferredToolStub;
  readonly score: number;
}

function scoreDeferredTools(
  entries: Iterable<{ readonly stub: DeferredToolStub }>,
  terms: ReadonlyArray<string>,
): ReadonlyArray<ScoredDeferredTool> {
  return [...entries]
    .map(({ stub }) => ({ stub, score: scoreDeferredTool(stub, terms) }))
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score);
}

function scoreDeferredTool(stub: DeferredToolStub, terms: ReadonlyArray<string>): number {
  const name = stub.name.toLowerCase();
  const text = `${stub.name} ${stub.description}`.toLowerCase();
  return terms.reduce((score, term) => score + scoreTermMatch(name, text, term), 0);
}

function scoreTermMatch(name: string, text: string, term: string): number {
  if (name.includes(term)) return 3;
  if (text.includes(term)) return 1;
  return 0;
}

// ─── ToolRegistry ───────────────────────────────────────────

export class ToolRegistry {
  /** Always-loaded tools (full specs). */
  private readonly tools = new Map<string, ToolSpec>();
  /** Deferred tools — stubs in prompt, full spec resolved on demand. */
  private readonly deferred = new Map<string, { stub: DeferredToolStub; full: ToolSpec }>();

  /** Cached arrays — invalidated on register/resolve. */
  private cachedAll: ReadonlyArray<ToolSpec> | null = null;
  private cachedLoaded: ReadonlyArray<ToolSpec> | null = null;
  private cachedByCategory = new Map<ToolCategory, ReadonlyArray<ToolSpec>>();
  private cachedPlanMode: ReadonlyArray<ToolSpec> | null = null;
  private cachedDeferredStubs: ReadonlyArray<DeferredToolStub> | null = null;

  /** Register a tool as always-loaded. */
  register(tool: ToolSpec): void {
    if (this.tools.has(tool.name) || this.deferred.has(tool.name)) {
      throw new Error(`Tool "${tool.name}" is already registered`);
    }
    this.tools.set(tool.name, tool);
    this.invalidateCache();
  }

  /**
   * Register a tool as deferred — only a stub (name + description) is
   * included in the prompt. Call resolve() to make the full spec available.
   */
  registerDeferred(tool: ToolSpec): void {
    if (this.tools.has(tool.name) || this.deferred.has(tool.name)) {
      throw new Error(`Tool "${tool.name}" is already registered`);
    }
    this.deferred.set(tool.name, {
      stub: { name: tool.name, description: tool.description, category: tool.category },
      full: tool,
    });
    this.invalidateCache();
  }

  /**
   * Resolve a deferred tool — moves it from deferred to loaded.
   * Returns the full ToolSpec, or null if not found in deferred set.
   */
  resolve(name: string): ToolSpec | null {
    const entry = this.deferred.get(name);
    if (!entry) return null;
    this.deferred.delete(name);
    this.tools.set(name, entry.full);
    this.invalidateCache();
    return entry.full;
  }

  /**
   * Search deferred tools by keyword match on name and description.
   * Returns matching stubs sorted by relevance. Also resolves them.
   */
  search(query: string, maxResults: number = 5): ReadonlyArray<DeferredToolStub> {
    const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 0);
    if (terms.length === 0) return [];

    const scored = scoreDeferredTools(this.deferred.values(), terms);
    const results = scored.slice(0, maxResults).map((s) => s.stub);
    if (this.resolveDeferredResults(results)) this.invalidateCache();

    return results;
  }

  private resolveDeferredResults(results: ReadonlyArray<DeferredToolStub>): boolean {
    let resolved = false;
    for (const stub of results) {
      const entry = this.deferred.get(stub.name);
      if (!entry) continue;
      this.deferred.delete(stub.name);
      this.tools.set(stub.name, entry.full);
      resolved = true;
    }
    return resolved;
  }

  /** Get a tool by name (loaded or deferred — resolves deferred on access). */
  get(name: string): ToolSpec {
    const tool = this.tools.get(name);
    if (tool) return tool;

    // Check deferred — auto-resolve on direct get()
    const entry = this.deferred.get(name);
    if (entry) {
      this.resolve(name);
      return entry.full;
    }

    throw new ToolNotFoundError(name);
  }

  /** Check if a tool exists (loaded or deferred). */
  has(name: string): boolean {
    return this.tools.has(name) || this.deferred.has(name);
  }

  /** List all tool names (loaded + deferred). */
  list(): ReadonlyArray<string> {
    return [...this.tools.keys(), ...this.deferred.keys()];
  }

  /** Get all tools (loaded + deferred resolved). For backward compatibility. */
  getAll(): ReadonlyArray<ToolSpec> {
    if (!this.cachedAll) {
      const all = [...this.tools.values()];
      for (const { full } of this.deferred.values()) {
        all.push(full);
      }
      this.cachedAll = all;
    }
    return this.cachedAll;
  }

  /** Get only loaded tools (not deferred). Use for LLM tool schemas. */
  getLoaded(): ReadonlyArray<ToolSpec> {
    if (!this.cachedLoaded) {
      this.cachedLoaded = Array.from(this.tools.values());
    }
    return this.cachedLoaded;
  }

  /** Get deferred tool stubs for prompt injection. */
  getDeferred(): ReadonlyArray<DeferredToolStub> {
    if (!this.cachedDeferredStubs) {
      this.cachedDeferredStubs = Array.from(this.deferred.values()).map((e) => e.stub);
    }
    return this.cachedDeferredStubs;
  }

  getByCategory(category: ToolCategory): ReadonlyArray<ToolSpec> {
    let cached = this.cachedByCategory.get(category);
    if (!cached) {
      cached = Array.from(this.tools.values()).filter(
        (t) => t.category === category,
      );
      this.cachedByCategory.set(category, cached);
    }
    return cached;
  }

  getReadOnly(): ReadonlyArray<ToolSpec> {
    return this.getByCategory("readonly");
  }

  /** Tools available in plan mode: readonly + state (internal agent state). */
  getPlanModeTools(): ReadonlyArray<ToolSpec> {
    if (!this.cachedPlanMode) {
      this.cachedPlanMode = Array.from(this.tools.values()).filter(
        (t) => t.category === "readonly" || t.category === "state",
      );
    }
    return this.cachedPlanMode;
  }

  /** True if any deferred tools remain unresolved. */
  hasDeferredTools(): boolean {
    return this.deferred.size > 0;
  }

  private invalidateCache(): void {
    this.cachedAll = null;
    this.cachedLoaded = null;
    this.cachedByCategory.clear();
    this.cachedPlanMode = null;
    this.cachedDeferredStubs = null;
  }
}
