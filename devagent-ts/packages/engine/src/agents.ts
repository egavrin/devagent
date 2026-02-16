/**
 * Agent types — General, Reviewer, Architect.
 * Each agent type wraps a TaskLoop with a specialized system prompt
 * and restricted tool set. Agents are spawned by the delegate tool.
 */

import type {
  LLMProvider,
  DevAgentConfig,
  CostRecord,
} from "@devagent/core";
import { AgentType, EventBus, ApprovalGate } from "@devagent/core";
import { ToolRegistry } from "@devagent/tools";
import { TaskLoop } from "./task-loop.js";
import type { TaskMode, TaskLoopResult } from "./task-loop.js";

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
}

export interface AgentRunResult {
  readonly agentId: string;
  readonly agentType: AgentType;
  readonly result: TaskLoopResult;
  readonly cost: CostRecord;
}

// ─── System Prompts ──────────────────────────────────────────

const GENERAL_PROMPT = `You are a General development agent.

Working directory: {{repoRoot}}

You have access to tools for reading files, writing files, searching code, running commands, and git operations.

When the user asks you to perform a task:
1. Understand the request
2. Use tools to explore the codebase and gather information
3. Make changes or provide analysis as requested
4. Report what you did

Be concise and direct. Fail fast — report errors immediately rather than guessing.`;

const REVIEWER_PROMPT = `You are a Code Review agent.

Working directory: {{repoRoot}}

You have access to read-only tools for analyzing code. You CANNOT modify files or run commands.

When reviewing code:
1. Read the relevant files and understand the context
2. Check for bugs, security issues, performance problems, and style violations
3. Provide structured feedback with file paths and line numbers
4. Rate severity: critical, warning, suggestion, nitpick
5. Suggest specific fixes where possible

Be thorough but concise. Focus on issues that matter — skip trivial formatting.`;

const ARCHITECT_PROMPT = `You are an Architecture agent.

Working directory: {{repoRoot}}

You have access to read-only tools for analyzing code. You CANNOT modify files or run commands.

When designing or analyzing architecture:
1. Read the relevant files and understand the current structure
2. Identify patterns, dependencies, and architectural decisions
3. Break down complex tasks into concrete implementation steps
4. Consider trade-offs and alternatives
5. Produce a clear, actionable plan

Be specific about file paths and function signatures. Avoid hand-waving.`;

// ─── Agent Registry ──────────────────────────────────────────

const AGENT_DEFINITIONS: ReadonlyArray<AgentDefinition> = [
  {
    type: AgentType.GENERAL,
    name: "General",
    description: "Default agent. Answers questions, writes code, runs commands.",
    systemPromptTemplate: GENERAL_PROMPT,
    defaultMode: "act",
    allowedToolCategories: ["readonly", "mutating", "workflow", "external"],
  },
  {
    type: AgentType.REVIEWER,
    name: "Reviewer",
    description: "Code review with structured output. Read-only tools only.",
    systemPromptTemplate: REVIEWER_PROMPT,
    defaultMode: "plan",
    allowedToolCategories: ["readonly"],
  },
  {
    type: AgentType.ARCHITECT,
    name: "Architect",
    description: "Design documents and task breakdown. Read-only tools only.",
    systemPromptTemplate: ARCHITECT_PROMPT,
    defaultMode: "plan",
    allowedToolCategories: ["readonly"],
  },
];

export class AgentRegistry {
  private readonly definitions = new Map<AgentType, AgentDefinition>();

  constructor() {
    for (const def of AGENT_DEFINITIONS) {
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

  // Build system prompt from template
  const systemPrompt = definition.systemPromptTemplate.replace(
    /\{\{repoRoot\}\}/g,
    options.repoRoot,
  );

  // Filter tools based on agent's allowed categories
  const filteredTools = filterToolsByCategories(
    options.tools,
    definition.allowedToolCategories,
  );

  // Create isolated TaskLoop for this agent
  const loop = new TaskLoop({
    provider: options.provider,
    tools: filteredTools,
    bus: options.bus,
    approvalGate: options.approvalGate,
    config: options.config,
    systemPrompt,
    repoRoot: options.repoRoot,
    mode: definition.defaultMode,
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
