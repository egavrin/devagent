/**
 * Agent types — General, Reviewer, Architect, Explore.
 * Each agent type wraps a TaskLoop with a specialized system prompt
 * and restricted tool set. Agents are spawned by the delegate tool.
 */

import { readFileSync } from "node:fs";
import type {
  LLMProvider,
  DevAgentConfig,
  CostRecord,
  ToolSpec,
  SkillRegistry,
  Message,
} from "../core/index.js";
import { AgentType, MessageRole, EventBus, ApprovalGate } from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";
import { TaskLoop } from "./task-loop.js";
import type { TaskMode, TaskLoopResult } from "./task-loop.js";
import { SessionState } from "./session-state.js";
import type { SessionStateJSON } from "./session-state.js";
import { assembleAgentSystemPrompt } from "./agent-prompt.js";
import type { TurnBriefing } from "./briefing.js";
import { parseStructuredAgentOutput } from "./subagent-contract.js";

// ─── Agent Definition ────────────────────────────────────────

export interface AgentDefinition {
  readonly type: AgentType;
  readonly name: string;
  readonly description: string;
  readonly systemPromptTemplate: string;
  readonly defaultMode: TaskMode;
  readonly allowedToolCategories: ReadonlyArray<string>;
}

export type AgentProviderFactory = (
  config: DevAgentConfig,
  agentType: AgentType,
) => LLMProvider;

export interface AgentAmbientContext {
  readonly skills?: SkillRegistry;
  readonly approvalMode?: string;
  readonly providerLabel?: string;
  readonly briefing?: TurnBriefing;
  readonly projectInstructions?: string | null;
  readonly providerFactory?: AgentProviderFactory;
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
  readonly depth: number;
  readonly ambient?: AgentAmbientContext;
  readonly createDelegateTool?: (ctx: {
    provider: LLMProvider;
    tools: ToolRegistry;
    bus: EventBus;
    approvalGate: ApprovalGate;
    config: DevAgentConfig;
    repoRoot: string;
    agentRegistry: AgentRegistry;
    parentAgentId: string;
    getParentSessionState?: () => SessionState | undefined;
    depth: number;
    parentAgentType: AgentType;
    ambient?: AgentAmbientContext;
  }) => ToolSpec;
  readonly laneLabel?: string | null;
  readonly batchId?: string;
  readonly batchSize?: number;
}

export interface AgentRunResult {
  readonly agentId: string;
  readonly agentType: AgentType;
  readonly agentMeta: {
    readonly agentId: string;
    readonly parentId: string | null;
    readonly depth: number;
    readonly agentType: AgentType;
  };
  readonly result: TaskLoopResult;
  readonly cost: CostRecord;
  readonly finalMessage: string;
  readonly childSessionState: SessionStateJSON;
  readonly parsedOutput: Record<string, unknown> | null;
}

// ─── Agent Registry ──────────────────────────────────────────

function getAgentDefinitions(): ReadonlyArray<AgentDefinition> {
  return [
    {
      type: AgentType.GENERAL,
      name: "General",
      description: "Default agent. Answers questions, writes code, runs commands.",
      systemPromptTemplate: loadRolePrompt("agent-general.md"),
      defaultMode: "act",
      allowedToolCategories: ["readonly", "mutating", "workflow", "external"],
    },
    {
      type: AgentType.REVIEWER,
      name: "Reviewer",
      description: "Code review with structured output. Read-only tools only.",
      systemPromptTemplate: loadRolePrompt("agent-reviewer.md"),
      defaultMode: "plan",
      allowedToolCategories: ["readonly"],
    },
    {
      type: AgentType.ARCHITECT,
      name: "Architect",
      description: "Design documents and task breakdown. Read-only tools only.",
      systemPromptTemplate: loadRolePrompt("agent-architect.md"),
      defaultMode: "plan",
      allowedToolCategories: ["readonly"],
    },
    {
      type: AgentType.EXPLORE,
      name: "Explore",
      description: "Codebase search and discovery. Read-only tools only. Fast iteration cap.",
      systemPromptTemplate: loadRolePrompt("agent-explore.md"),
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

  const filteredTools = filterToolsByCategories(
    options.tools,
    agentType,
    options.config,
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

  const childTools = new ToolRegistry();
  for (const tool of filteredTools.getAll()) {
    if (tool.name === "delegate") continue;
    childTools.register(tool);
  }

  if (
    options.createDelegateTool &&
    options.depth < 1 &&
    allowsChildDelegation(agentType, options.config)
  ) {
    childTools.register(options.createDelegateTool({
      provider: options.provider,
      tools: childTools,
      bus: options.bus,
      approvalGate: options.approvalGate,
      config: options.config,
      repoRoot: options.repoRoot,
      agentRegistry: registry,
      parentAgentId: options.agentId,
      getParentSessionState: () => sessionState,
      depth: options.depth,
      parentAgentType: agentType,
      ambient: options.ambient,
    }));
  }

  const systemPrompt = assembleAgentSystemPrompt({
    agentType,
    repoRoot: options.repoRoot,
    rolePrompt: definition.systemPromptTemplate,
    availableTools: childTools.getAll(),
    approvalMode: options.ambient?.approvalMode,
    providerLabel: options.ambient?.providerLabel,
    skills: options.ambient?.skills,
    briefing: options.ambient?.briefing,
    projectInstructions: options.ambient?.projectInstructions,
  });

  const provider = options.ambient?.providerFactory
    ? options.ambient.providerFactory(options.config, agentType)
    : options.provider;

  const loop = new TaskLoop({
    provider,
    tools: childTools,
    bus: options.bus,
    approvalGate: options.approvalGate,
    config: options.config,
    systemPrompt,
    repoRoot: options.repoRoot,
    mode: definition.defaultMode,
    sessionState,
    injectSessionStateOnFirstTurn: true,
    agentContext: {
      agentId: options.agentId,
      parentAgentId: options.parentId,
      depth: options.depth,
      agentType,
      laneLabel: options.laneLabel,
      batchId: options.batchId,
      batchSize: options.batchSize,
    },
  });

  const result = await runWithTimeout(loop, query, options.config.subagentTimeoutMs);
  const finalMessage = extractFinalAssistantMessage(result);
  const parsedOutput = parseStructuredAgentOutput(agentType, finalMessage);

  return {
    agentId: options.agentId,
    agentType,
    agentMeta: {
      agentId: options.agentId,
      parentId: options.parentId,
      depth: options.depth,
      agentType,
    },
    result,
    cost: result.cost,
    finalMessage,
    childSessionState: sessionState.toJSON(),
    parsedOutput,
  };
}

// ─── Fork Agent ─────────────────────────────────────────────

/** Marker to detect and prevent recursive forking. */
const FORK_BOILERPLATE_TAG = "[FORKED_AGENT]";

export interface ForkAgentOptions extends AgentRunOptions {
  /** Parent's current messages — the forked child inherits these for prompt cache sharing. */
  readonly parentMessages: ReadonlyArray<Message>;
  /** Parent's system prompt — reused as-is for cache prefix alignment. */
  readonly parentSystemPrompt: string;
}

/**
 * Spawn a forked agent that inherits the parent's full conversation context.
 * The child uses the same system prompt prefix as the parent, enabling
 * Anthropic prompt cache sharing (identical cache key prefix).
 *
 * Forked children cannot fork again (depth guard via FORK_BOILERPLATE_TAG).
 */
export async function runForkedAgent(
  query: string,
  options: ForkAgentOptions,
  registry: AgentRegistry,
): Promise<AgentRunResult> {
  // Depth guard: prevent recursive forking
  const hasBoilerplate = options.parentMessages.some(
    (m) => m.content?.includes(FORK_BOILERPLATE_TAG) ?? false,
  );
  if (hasBoilerplate) {
    throw new Error("Forked agents cannot fork again (recursive fork detected)");
  }

  // Build initial messages: parent's history + fork directive
  const initialMessages: Message[] = [...options.parentMessages];
  initialMessages.push({
    role: MessageRole.SYSTEM as const,
    content: `${FORK_BOILERPLATE_TAG} You are a forked worker agent. Your directive:\n\n${query}\n\nFocus on this specific task. Do not attempt to fork additional agents.`,
  });

  // Clone parent's session state
  const sessionState = new SessionState({ persist: false });
  if (options.parentSessionState) {
    seedSessionState(sessionState, options.parentSessionState);
  }

  // Fork uses the parent's full tool set (all categories)
  const childTools = new ToolRegistry();
  for (const tool of options.tools.getAll()) {
    // Exclude delegate to prevent recursive delegation from forks
    if (tool.name === "delegate") continue;
    childTools.register(tool);
  }

  const provider = options.ambient?.providerFactory
    ? options.ambient.providerFactory(options.config, AgentType.GENERAL)
    : options.provider;

  const loop = new TaskLoop({
    provider,
    tools: childTools,
    bus: options.bus,
    approvalGate: options.approvalGate,
    config: options.config,
    systemPrompt: options.parentSystemPrompt,
    repoRoot: options.repoRoot,
    mode: "act",
    initialMessages,
    sessionState,
    injectSessionStateOnFirstTurn: false,
    agentContext: {
      agentId: options.agentId,
      parentAgentId: options.parentId,
      depth: options.depth,
      agentType: AgentType.GENERAL,
      laneLabel: options.laneLabel,
      batchId: options.batchId,
      batchSize: options.batchSize,
    },
  });

  const result = await runWithTimeout(loop, query, options.config.subagentTimeoutMs);
  const finalMessage = extractFinalAssistantMessage(result);

  return {
    agentId: options.agentId,
    agentType: AgentType.GENERAL,
    agentMeta: {
      agentId: options.agentId,
      parentId: options.parentId,
      depth: options.depth,
      agentType: AgentType.GENERAL,
    },
    result,
    cost: result.cost,
    finalMessage,
    childSessionState: sessionState.toJSON(),
    parsedOutput: null,
  };
}

// ─── Helpers ─────────────────────────────────────────────────

/**
 * Create a filtered ToolRegistry containing only tools in the allowed categories.
 */
function filterToolsByCategories(
  registry: ToolRegistry,
  agentType: AgentType,
  config: DevAgentConfig,
  allowedCategories: ReadonlyArray<string>,
): ToolRegistry {
  const filtered = new ToolRegistry();
  const overrides = config.agentPermissionOverrides?.[agentType];

  const allTools = registry.getAll();
  for (const tool of allTools) {
    if (!allowedCategories.includes(tool.category)) continue;
    if (overrides && overrides[tool.category] === "deny") continue;
    filtered.register(tool);
  }

  return filtered;
}

/**
 * Seed a subagent's SessionState with the parent's readonly coverage
 * and tool summaries so the subagent knows what files were already examined.
 * Also copies findings, knowledge, and env facts so the child can continue
 * the parent's analysis without losing structured context.
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
  for (const finding of parent.getFindings()) {
    child.addFinding(finding.title, finding.detail, finding.iteration);
  }
  for (const knowledge of parent.getKnowledge()) {
    child.addKnowledge(knowledge.key, knowledge.content, knowledge.iteration);
  }
  for (const fact of parent.toJSON().envFacts ?? []) {
    child.addEnvFact(fact.key, fact.value);
  }
}

function loadRolePrompt(filename: string): string {
  return readFileSync(new URL(`./prompts/${filename}`, import.meta.url), "utf-8");
}

function allowsChildDelegation(
  agentType: AgentType,
  config: DevAgentConfig,
): boolean {
  const allowed = config.allowedChildAgents?.[agentType];
  return allowed === undefined || allowed.length > 0;
}

function extractFinalAssistantMessage(result: TaskLoopResult): string {
  const assistantMessages = result.messages.filter(
    (message) => message.content && message.role === "assistant",
  );
  return assistantMessages[assistantMessages.length - 1]?.content ?? "(no output)";
}

async function runWithTimeout(
  loop: TaskLoop,
  query: string,
  timeoutMs: number | undefined,
): Promise<TaskLoopResult> {
  if (!timeoutMs || timeoutMs <= 0) {
    return loop.run(query);
  }

  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      loop.run(query),
      new Promise<TaskLoopResult>((_, reject) => {
        timer = setTimeout(() => {
          loop.abort();
          reject(new Error(`Timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
