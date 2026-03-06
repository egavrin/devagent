/**
 * Agent types — General, Reviewer, Architect, Explore.
 * Each agent type wraps a TaskLoop with a specialized system prompt
 * and restricted tool set. Agents are spawned by the delegate tool.
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type {
  LLMProvider,
  DevAgentConfig,
  CostRecord,
} from "@devagent/core";
import { AgentType, EventBus, ApprovalGate } from "@devagent/core";
import { ToolRegistry } from "@devagent/tools";
import { TaskLoop } from "./task-loop.js";
import type { TaskMode, TaskLoopResult } from "./task-loop.js";
import { SessionState } from "./session-state.js";

// ─── Agent Definition ────────────────────────────────────────

export interface AgentDefinition {
  readonly type: AgentType;
  readonly name: string;
  readonly description: string;
  readonly systemPromptTemplate: string;
  readonly defaultMode: TaskMode;
  readonly allowedToolCategories: ReadonlyArray<string>;
}

export interface AgentRunOptions {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly repoRoot: string;
  readonly parentId: string | null;
  readonly agentId: string;
  /** Parent's session state — seeded into the subagent so it knows what was already read. */
  readonly parentSessionState?: SessionState;
}

export interface AgentRunResult {
  readonly agentId: string;
  readonly agentType: AgentType;
  readonly result: TaskLoopResult;
  readonly cost: CostRecord;
}

// ─── System Prompts (loaded from markdown files) ─────────────

const PROMPTS_DIR = join(dirname(fileURLToPath(import.meta.url)), "prompts");

function loadAgentPrompt(filename: string): string {
  return readFileSync(join(PROMPTS_DIR, filename), "utf-8");
}

// Cache loaded prompts (never change during process lifetime)
let cachedCommonPrompt: string | null = null;
let cachedGeneralPrompt: string | null = null;
let cachedReviewerPrompt: string | null = null;
let cachedArchitectPrompt: string | null = null;
let cachedExplorePrompt: string | null = null;

function getCommonPrompt(): string {
  cachedCommonPrompt ??= loadAgentPrompt("agent-common.md");
  return cachedCommonPrompt;
}

function getGeneralPrompt(): string {
  cachedGeneralPrompt ??= loadAgentPrompt("agent-general.md");
  return cachedGeneralPrompt;
}

function getReviewerPrompt(): string {
  cachedReviewerPrompt ??= loadAgentPrompt("agent-reviewer.md");
  return cachedReviewerPrompt;
}

function getArchitectPrompt(): string {
  cachedArchitectPrompt ??= loadAgentPrompt("agent-architect.md");
  return cachedArchitectPrompt;
}

function getExplorePrompt(): string {
  cachedExplorePrompt ??= loadAgentPrompt("agent-explore.md");
  return cachedExplorePrompt;
}

// ─── Agent Registry ──────────────────────────────────────────

function getAgentDefinitions(): ReadonlyArray<AgentDefinition> {
  return [
    {
      type: AgentType.GENERAL,
      name: "General",
      description: "Default agent. Answers questions, writes code, runs commands.",
      systemPromptTemplate: getGeneralPrompt(),
      defaultMode: "act",
      allowedToolCategories: ["readonly", "mutating", "workflow", "external"],
    },
    {
      type: AgentType.REVIEWER,
      name: "Reviewer",
      description: "Code review with structured output. Read-only tools only.",
      systemPromptTemplate: getReviewerPrompt(),
      defaultMode: "plan",
      allowedToolCategories: ["readonly"],
    },
    {
      type: AgentType.ARCHITECT,
      name: "Architect",
      description: "Design documents and task breakdown. Read-only tools only.",
      systemPromptTemplate: getArchitectPrompt(),
      defaultMode: "plan",
      allowedToolCategories: ["readonly"],
    },
    {
      type: AgentType.EXPLORE,
      name: "Explore",
      description: "Codebase search and discovery. Read-only tools only. Fast iteration cap.",
      systemPromptTemplate: getExplorePrompt(),
      defaultMode: "act",
      allowedToolCategories: ["readonly"],
    },
  ];
}

export class AgentRegistry {
  private readonly definitions = new Map<AgentType, AgentDefinition>();

  constructor() {
    for (const def of getAgentDefinitions()) {
      this.definitions.set(def.type, def);
    }
  }

  get(type: AgentType): AgentDefinition {
    const def = this.definitions.get(type);
    if (!def) {
      throw new Error(`Unknown agent type: ${type}`);
    }
    return def;
  }

  has(type: AgentType): boolean {
    return this.definitions.has(type);
  }

  list(): ReadonlyArray<AgentDefinition> {
    return Array.from(this.definitions.values());
  }

  register(definition: AgentDefinition): void {
    this.definitions.set(definition.type, definition);
  }
}

// ─── Agent Execution ─────────────────────────────────────────

/**
 * Spawn and run an agent with the given type and query.
 * The agent runs in an isolated TaskLoop with its own message history.
 */
export async function runAgent(
  agentType: AgentType,
  query: string,
  options: AgentRunOptions,
  registry: AgentRegistry,
): Promise<AgentRunResult> {
  const definition = registry.get(agentType);

  // Build system prompt: common guidance + agent-specific template
  const systemPrompt = (getCommonPrompt() + "\n\n" + definition.systemPromptTemplate)
    .replace(/\{\{repoRoot\}\}/g, options.repoRoot);

  // Filter tools based on agent's allowed categories
  const filteredTools = filterToolsByCategories(
    options.tools,
    definition.allowedToolCategories,
  );

  // Create isolated TaskLoop with its own SessionState so stagnation
  // detection, tool-output pruning, and post-compaction summaries work.
  // Seed from parent's coverage data so the subagent doesn't re-read files
  // that have already been examined.
  const sessionState = new SessionState({ persist: false });
  if (options.parentSessionState) {
    seedSessionState(sessionState, options.parentSessionState);
  }

  const loop = new TaskLoop({
    provider: options.provider,
    tools: filteredTools,
    bus: options.bus,
    approvalGate: options.approvalGate,
    config: options.config,
    systemPrompt,
    repoRoot: options.repoRoot,
    mode: definition.defaultMode,
    sessionState,
  });

  const result = await loop.run(query);

  return {
    agentId: options.agentId,
    agentType,
    result,
    cost: result.cost,
  };
}

// ─── Helpers ─────────────────────────────────────────────────

/**
 * Create a filtered ToolRegistry containing only tools in the allowed categories.
 */
function filterToolsByCategories(
  registry: ToolRegistry,
  allowedCategories: ReadonlyArray<string>,
): ToolRegistry {
  const filtered = new ToolRegistry();

  const allTools = registry.getAll();
  for (const tool of allTools) {
    if (allowedCategories.includes(tool.category)) {
      filtered.register(tool);
    }
  }

  return filtered;
}

/**
 * Seed a subagent's SessionState with the parent's readonly coverage
 * and tool summaries so the subagent knows what files were already examined.
 * Does NOT copy findings (those belong to the parent's analysis).
 */
function seedSessionState(child: SessionState, parent: SessionState): void {
  // Copy tool summaries — tells the subagent what was already read
  for (const summary of parent.getToolSummaries()) {
    child.addToolSummary({ ...summary });
  }
  // Copy readonly coverage — prevents redundant calls
  for (const [tool, targets] of parent.getReadonlyCoverage()) {
    for (const target of targets) {
      child.recordReadonlyCoverage(tool, target);
    }
  }
  // Copy modified files — scope awareness
  for (const file of parent.getModifiedFiles()) {
    child.recordModifiedFile(file);
  }
}
