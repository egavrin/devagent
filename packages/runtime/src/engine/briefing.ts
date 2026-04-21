/**
 * Turn briefing synthesis — creates structured context summaries
 * for fresh TaskLoop instances when continuing a persisted session.
 *
 * Two strategies:
 *   - "heuristic": Fast extraction from messages without LLM call (~1ms)
 *   - "llm": LLM-based structured summary (~2-5s, higher quality)
 *   - "auto": LLM if turn had 5+ tool calls, heuristic otherwise
 *
 * Inspired by:
 *   - Microsoft/Salesforce paper: "Don't pass raw history to executing LLM"
 *   - Codex CLI: Pre-sampling compaction prompt
 *   - OpenCode: Structured compaction template (goal, discoveries, accomplished, files)
 *   - Claude Code: Sub-agents start fresh with focused prompts
 */

import type { PlanStep } from "./plan-tool.js";
import type { Message, LLMProvider } from "../core/index.js";
import { MessageRole } from "../core/index.js";

export interface TurnBriefing {
  readonly turnNumber: number;
  /** Summary of what was accomplished in the prior turn. */
  readonly priorTaskSummary: string;
  /** Key facts, decisions, constraints discovered. */
  readonly activeContext: string;
  /** What the user asked that isn't done yet. */
  readonly pendingWork: string | null;
  /** Files modified, read, or discovered (paths). */
  readonly keyArtifacts: ReadonlyArray<string>;
  /** Structured plan steps from the last update_plan call, if any. */
  readonly planSteps: ReadonlyArray<PlanStep> | null;
}

export type BriefingStrategy = "heuristic" | "llm" | "auto";

export interface SynthesizeBriefingOptions {
  /** Strategy for briefing synthesis. Default: "auto". */
  readonly strategy?: BriefingStrategy;
  /** LLM provider for "llm" and "auto" strategies. */
  readonly provider?: LLMProvider;
  /** Max chars for the briefing text. Default: 6000 (~1500 tokens). */
  readonly maxChars?: number;
}

// ─── Constants ──────────────────────────────────────────────

const DEFAULT_MAX_CHARS = 6000;
const LLM_TOOL_CALL_THRESHOLD = 5;

const SYNTHESIS_PROMPT = `Create a structured briefing for the next agent continuing this work. Be concise and specific.

## Goal
What is the user trying to accomplish? One sentence.

## Key Decisions & Constraints
Important decisions made, user preferences, constraints discovered. Bullet points.

## Accomplished
What was completed. Include specific file paths and function names. Bullet points.

## Plan Status
If a multi-step plan exists, list each step with its status (pending/in_progress/completed).
Format: - [status] description
If no structured plan was used, say "No plan."

## Pending
What remains to be done. Be specific about next steps. Bullet points. If everything is done, say "Nothing pending."

## Relevant Files
Files read, modified, or discovered. Format: \`path\` — role/status. One per line.`;

// ─── Main Function ──────────────────────────────────────────

/**
 * Synthesize a structured briefing from completed turn messages.
 * Called when synthesizing continuation context from a prior run.
 */
export async function synthesizeBriefing(
  messages: ReadonlyArray<Message>,
  turnNumber: number,
  options?: SynthesizeBriefingOptions,
): Promise<TurnBriefing> {
  const strategy = options?.strategy ?? "auto";
  const maxChars = options?.maxChars ?? DEFAULT_MAX_CHARS;
  const context = { messages, turnNumber, options, maxChars };
  return (STRATEGY_HANDLERS[strategy] ?? synthesizeHeuristicStrategy)(context);
}

interface BriefingStrategyContext {
  readonly messages: ReadonlyArray<Message>;
  readonly turnNumber: number;
  readonly options?: SynthesizeBriefingOptions;
  readonly maxChars: number;
}

type BriefingStrategyHandler = (context: BriefingStrategyContext) => Promise<TurnBriefing>;

const STRATEGY_HANDLERS: Readonly<Record<BriefingStrategy, BriefingStrategyHandler>> = {
  heuristic: synthesizeHeuristicStrategy,
  llm: synthesizeLLMStrategy,
  auto: synthesizeAutoStrategy,
};

async function synthesizeHeuristicStrategy(
  context: BriefingStrategyContext,
): Promise<TurnBriefing> {
  return extractHeuristicBriefing(context.messages, context.turnNumber, context.maxChars);
}

async function synthesizeLLMStrategy(
  context: BriefingStrategyContext,
): Promise<TurnBriefing> {
  return context.options?.provider
    ? synthesizeLLMBriefing(
        context.messages,
        context.turnNumber,
        context.options.provider,
        context.maxChars,
      )
    : synthesizeHeuristicStrategy(context);
}

async function synthesizeAutoStrategy(
  context: BriefingStrategyContext,
): Promise<TurnBriefing> {
  return shouldUseLLMBriefing(context.messages, context.options?.provider)
    ? synthesizeAutoLLMBriefing(
        context.messages,
        context.turnNumber,
        context.options.provider,
        context.maxChars,
      )
    : synthesizeHeuristicStrategy(context);
}

function shouldUseLLMBriefing(
  messages: ReadonlyArray<Message>,
  provider: LLMProvider | undefined,
): provider is LLMProvider {
  return Boolean(provider && countToolCalls(messages) >= LLM_TOOL_CALL_THRESHOLD);
}

async function synthesizeAutoLLMBriefing(
  messages: ReadonlyArray<Message>,
  turnNumber: number,
  provider: LLMProvider,
  maxChars: number,
): Promise<TurnBriefing> {
  try {
    return await synthesizeLLMBriefing(messages, turnNumber, provider, maxChars);
  } catch {
    return extractHeuristicBriefing(messages, turnNumber, maxChars);
  }
}

// ─── Heuristic Strategy ─────────────────────────────────────

/**
 * Extract a briefing from messages using heuristic rules — no LLM call.
 * Fast (~1ms) but lower quality than LLM synthesis.
 */
export function extractHeuristicBriefing(
  messages: ReadonlyArray<Message>,
  turnNumber: number,
  maxChars: number = DEFAULT_MAX_CHARS,
): TurnBriefing {
  // 1. Find the user query (first USER message after system)
  const userQuery = findFirstUserQuery(messages);

  // 2. Find the final assistant response (last ASSISTANT without tool calls)
  const finalResponse = findFinalAssistantResponse(messages);

  // 3. Collect file paths from tool call arguments
  const artifacts = extractArtifacts(messages);

  // 4. Collect tool usage summary
  const toolSummary = extractToolSummary(messages);

  // 5. Extract plan steps if any
  const planSteps = extractPlanSteps(messages);

  // Build the briefing sections
  const summaryParts: string[] = [];

  if (userQuery) {
    summaryParts.push(`User asked: ${truncate(userQuery, 200)}`);
  }

  if (toolSummary) {
    summaryParts.push(`Tools used: ${toolSummary}`);
  }

  if (planSteps && planSteps.length > 0) {
    const planSummary = planSteps
      .map((s) => `[${s.status}] ${s.description}`)
      .join("; ");
    summaryParts.push(`Plan: ${planSummary}`);
  }

  if (finalResponse) {
    summaryParts.push(`Result: ${truncate(finalResponse, 500)}`);
  }

  const priorTaskSummary = truncate(summaryParts.join("\n"), maxChars * 0.5);

  // Build active context from tool results (errors, key findings)
  const activeContext = extractActiveContext(messages, maxChars * 0.3);

  // Pending work: check if the final response indicates incomplete work
  const pendingWork = extractPendingWork(messages) ?? derivePendingWorkFromPlanSteps(planSteps);

  return {
    turnNumber,
    priorTaskSummary,
    activeContext,
    pendingWork,
    keyArtifacts: artifacts.slice(0, 20), // Cap at 20 files
    planSteps,
  };
}

// ─── LLM Strategy ───────────────────────────────────────────

/**
 * Use LLM to synthesize a structured briefing from messages.
 * Higher quality but costs one LLM call (~2-5s).
 */
async function synthesizeLLMBriefing(
  messages: ReadonlyArray<Message>,
  turnNumber: number,
  provider: LLMProvider,
  maxChars: number,
): Promise<TurnBriefing> {
  // Build a condensed version of the conversation for the LLM
  const conversationSummary = condenseForSynthesis(messages, maxChars * 2);

  const synthesisMessages: Message[] = [
    {
      role: MessageRole.SYSTEM,
      content: SYNTHESIS_PROMPT,
    },
    {
      role: MessageRole.USER,
      content: `Here is the conversation to summarize:\n\n${conversationSummary}\n\nProduce the structured briefing now.`,
    },
  ];

  let response = "";
  const stream = provider.chat(synthesisMessages, []);
  for await (const chunk of stream) {
    if (chunk.type === "text") {
      response += chunk.content;
    }
  }

  if (!response.trim()) {
    // Empty response — fall back to heuristic
    return extractHeuristicBriefing(messages, turnNumber, maxChars);
  }

  // Parse the structured response into TurnBriefing
  return parseLLMBriefing(response, turnNumber, messages);
}

/**
 * Condense messages for LLM synthesis — keep user queries, assistant responses,
 * tool names, and error messages. Skip raw tool output content.
 */
function condenseForSynthesis(
  messages: ReadonlyArray<Message>,
  maxChars: number,
): string {
  const lines: string[] = [];
  let totalChars = 0;

  for (const msg of messages) {
    if (totalChars >= maxChars) break;
    const line = formatMessageForSynthesis(msg);
    if (!line) continue;
    lines.push(line);
    totalChars += line.length;
  }

  return lines.join("\n");
}

function formatMessageForSynthesis(message: Message): string | null {
  switch (message.role) {
    case MessageRole.USER:
      return `[User]: ${truncate(message.content ?? "", 500)}`;
    case MessageRole.ASSISTANT:
      return formatAssistantForSynthesis(message);
    case MessageRole.TOOL:
      return formatToolResultForSynthesis(message);
    case MessageRole.SYSTEM:
    default:
      return null;
  }
}

function formatAssistantForSynthesis(message: Message): string {
  if (!message.toolCalls || message.toolCalls.length === 0) {
    return `[Assistant]: ${truncate(message.content ?? "", 500)}`;
  }
  const toolNames = message.toolCalls.map((toolCall) => toolCall.name).join(", ");
  const text = message.content?.trim()
    ? `\n  Text: ${truncate(message.content, 200)}`
    : "";
  return `[Assistant]: Called tools: ${toolNames}${text}`;
}

function formatToolResultForSynthesis(message: Message): string {
  const limit = message.content?.startsWith("Error:") ? 200 : 100;
  return `[Tool result]: ${truncate(message.content ?? "", limit)}`;
}

/**
 * Parse LLM structured briefing response into TurnBriefing.
 */
function parseLLMBriefing(
  response: string,
  turnNumber: number,
  messages: ReadonlyArray<Message>,
): TurnBriefing {
  const sections = parseBriefingSections(response);
  const planSteps = parseLLMPlanSteps(sections.planText) ?? extractPlanSteps(messages);
  const pendingWork = getPendingWorkFromLLM(sections.pending, planSteps);
  const allArtifacts = mergeArtifacts(sections.files, messages);

  return {
    turnNumber,
    priorTaskSummary: buildLLMPriorTaskSummary(sections),
    activeContext: sections.decisions,
    pendingWork,
    keyArtifacts: allArtifacts.slice(0, 20),
    planSteps,
  };
}

interface ParsedBriefingSections {
  readonly goal: string;
  readonly decisions: string;
  readonly accomplished: string;
  readonly planText: string;
  readonly pending: string;
  readonly files: string;
}

function parseBriefingSections(response: string): ParsedBriefingSections {
  return {
    goal: extractHeadingSection(response, /## Goal\s*\n([\s\S]*?)(?=\n## |$)/),
    decisions: extractHeadingSection(response, /## Key Decisions[^\n]*\n([\s\S]*?)(?=\n## |$)/),
    accomplished: extractHeadingSection(response, /## Accomplished\s*\n([\s\S]*?)(?=\n## |$)/),
    planText: extractHeadingSection(response, /## Plan Status\s*\n([\s\S]*?)(?=\n## |$)/),
    pending: extractHeadingSection(response, /## Pending\s*\n([\s\S]*?)(?=\n## |$)/),
    files: extractHeadingSection(response, /## Relevant Files\s*\n([\s\S]*?)(?=\n## |$)/),
  };
}

function extractHeadingSection(response: string, pattern: RegExp): string {
  return response.match(pattern)?.[1]?.trim() ?? "";
}

function buildLLMPriorTaskSummary(sections: ParsedBriefingSections): string {
  return [
    sections.goal ? `Goal: ${sections.goal}` : null,
    sections.accomplished ? `Done: ${sections.accomplished}` : null,
  ].filter((part): part is string => part !== null).join("\n");
}

function parseLLMPlanSteps(planText: string): PlanStep[] | null {
  if (!planText || planText.toLowerCase().includes("no plan")) return null;
  const stepLines = planText.split("\n").filter((line) => line.trim().startsWith("- ["));
  return stepLines.length > 0 ? stepLines.map(parsePlanStepLine) : null;
}

function parsePlanStepLine(line: string): PlanStep {
  const match = line.match(/\[(pending|in_progress|completed)\]\s*(.*)/);
  return {
    status: (match?.[1] ?? "pending") as PlanStep["status"],
    description: match?.[2]?.trim() ?? line.trim(),
  };
}

function getPendingWorkFromLLM(
  pending: string,
  planSteps: ReadonlyArray<PlanStep> | null,
): string | null {
  return pending && !pending.toLowerCase().includes("nothing pending")
    ? pending
    : derivePendingWorkFromPlanSteps(planSteps);
}

function mergeArtifacts(
  filesSection: string,
  messages: ReadonlyArray<Message>,
): string[] {
  return [...new Set([...extractFilePathsFromText(filesSection), ...extractArtifacts(messages)])];
}

// ─── Extraction Helpers ─────────────────────────────────────

function findFirstUserQuery(messages: ReadonlyArray<Message>): string | null {
  for (const msg of messages) {
    if (msg.role === MessageRole.USER && msg.content?.trim()) {
      return msg.content;
    }
  }
  return null;
}

/**
 * Find the last non-empty user message content, or a default fallback.
 */
export function findLastUserContent(
  messages: ReadonlyArray<Message>,
  fallback = "(continue)",
): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]!.role === MessageRole.USER && messages[i]!.content?.trim()) {
      return messages[i]!.content!;
    }
  }
  return fallback;
}

function findFinalAssistantResponse(
  messages: ReadonlyArray<Message>,
): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (
      msg.role === MessageRole.ASSISTANT &&
      msg.content?.trim() &&
      (!msg.toolCalls || msg.toolCalls.length === 0)
    ) {
      return msg.content;
    }
  }
  return null;
}
function extractArtifacts(messages: ReadonlyArray<Message>): string[] {
  const paths = new Set<string>();

  for (const msg of messages) {
    for (const tc of msg.toolCalls ?? []) addToolCallArtifactPaths(tc.arguments, paths);
  }

  return Array.from(paths);
}

function addToolCallArtifactPaths(
  args: Readonly<Record<string, unknown>>,
  paths: Set<string>,
): void {
  for (const key of ["path", "file_path", "file", "filename"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) paths.add(value);
  }
}

function extractToolSummary(messages: ReadonlyArray<Message>): string | null {
  const toolCounts = new Map<string, number>();

  for (const msg of messages) {
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        toolCounts.set(tc.name, (toolCounts.get(tc.name) ?? 0) + 1);
      }
    }
  }

  if (toolCounts.size === 0) return null;

  return Array.from(toolCounts.entries())
    .map(([name, count]) => (count > 1 ? `${name}(x${count})` : name))
    .join(", ");
}
function extractPlanSteps(
  messages: ReadonlyArray<Message>,
): PlanStep[] | null {
  // Build set of failed/denied tool call IDs so we skip those plans
  const failedCallIds = collectFailedToolCallIds(messages);

  // Find the LAST *successful* update_plan call (most up-to-date plan state)
  let lastPlan: PlanStep[] | null = null;

  for (const msg of messages) {
    for (const toolCall of msg.toolCalls ?? []) {
      const parsedPlan = parseSuccessfulPlanToolCall(toolCall, failedCallIds);
      if (parsedPlan) lastPlan = parsedPlan;
    }
  }

  return lastPlan;
}

function collectFailedToolCallIds(messages: ReadonlyArray<Message>): Set<string> {
  const failedCallIds = new Set<string>();
  for (const message of messages) {
    if (isFailedToolResult(message)) failedCallIds.add(message.toolCallId);
  }
  return failedCallIds;
}

function isFailedToolResult(
  message: Message,
): message is Message & { readonly toolCallId: string } {
  return Boolean(
    message.role === MessageRole.TOOL &&
      message.toolCallId &&
      message.content?.startsWith("Error: "),
  );
}

function parseSuccessfulPlanToolCall(
  toolCall: NonNullable<Message["toolCalls"]>[number],
  failedCallIds: ReadonlySet<string>,
): PlanStep[] | null {
  if (toolCall.name !== "update_plan" || failedCallIds.has(toolCall.callId)) {
    return null;
  }
  const steps = parsePlanStepsPayload(toolCall.arguments["steps"]);
  return steps ? steps.map(planStepFromPayload) : null;
}

function parsePlanStepsPayload(steps: unknown): Array<Record<string, unknown>> | null {
  if (Array.isArray(steps)) return steps as Array<Record<string, unknown>>;
  if (typeof steps !== "string") return null;
  try {
    const parsed = JSON.parse(steps) as unknown;
    return Array.isArray(parsed) ? parsed as Array<Record<string, unknown>> : null;
  } catch {
    return null;
  }
}

function planStepFromPayload(step: Record<string, unknown>): PlanStep {
  return {
    description: (step["description"] as string) ?? "",
    status: ((step["status"] as string) ?? "pending") as PlanStep["status"],
  };
}

function extractActiveContext(
  messages: ReadonlyArray<Message>,
  maxChars: number,
): string {
  const contextParts: string[] = [];
  let totalChars = 0;

  // Collect error messages and system warnings
  for (const msg of messages) {
    if (totalChars >= maxChars) break;

    if (msg.role === MessageRole.SYSTEM && msg.content) {
      // System injections (doom loop warnings, validation failures, etc.)
      if (
        msg.content.includes("WARNING") ||
        msg.content.includes("VALIDATION FAILED")
      ) {
        const part = truncate(msg.content, 200);
        contextParts.push(part);
        totalChars += part.length;
      }
    }

    if (msg.role === MessageRole.TOOL && msg.content?.startsWith("Error:")) {
      const part = truncate(msg.content, 150);
      contextParts.push(`Tool error: ${part}`);
      totalChars += part.length;
    }
  }

  return contextParts.join("\n");
}

function extractPendingWork(
  messages: ReadonlyArray<Message>,
): string | null {
  // Check if the last assistant message mentions remaining work
  const finalResponse = findFinalAssistantResponse(messages);
  if (!finalResponse) return null;

  // Simple heuristic: look for phrases indicating incomplete work
  const incompletePatterns = [
    /still need[s]? to/i,
    /remain[s]? to be/i,
    /todo[:\s]/i,
    /next step[s]?/i,
    /not yet/i,
    /incomplete/i,
    /in progress/i,
  ];

  for (const pattern of incompletePatterns) {
    if (pattern.test(finalResponse)) {
      // Extract the sentence containing the pattern
      const sentences = finalResponse.split(/[.!?]\s+/);
      const relevant = sentences.filter((s) => pattern.test(s));
      if (relevant.length > 0) {
        return truncate(relevant.join(". "), 300);
      }
    }
  }

  return null;
}

function derivePendingWorkFromPlanSteps(
  planSteps: ReadonlyArray<PlanStep> | null,
): string | null {
  if (!planSteps || planSteps.length === 0) {
    return null;
  }

  const remainingSteps = planSteps.filter((step) => step.status !== "completed");
  if (remainingSteps.length === 0) {
    return null;
  }

  return remainingSteps
    .map((step) => `- [${step.status}] ${step.description}`)
    .join("\n");
}

function countToolCalls(messages: ReadonlyArray<Message>): number {
  let count = 0;
  for (const msg of messages) {
    if (msg.toolCalls) {
      count += msg.toolCalls.length;
    }
  }
  return count;
}

function extractFilePathsFromText(text: string): string[] {
  // Match backtick-wrapped paths and common file path patterns
  const paths = new Set<string>();

  // Backtick-wrapped paths
  const backtickMatches = text.matchAll(/`([^`]+\.[a-z]+[^`]*)`/g);
  for (const match of backtickMatches) {
    if (match[1] && !match[1].includes(" ")) {
      paths.add(match[1]);
    }
  }

  // Bare file paths (starting with ./ or containing /)
  const pathMatches = text.matchAll(
    /(?:^|\s)((?:\.\/|[a-zA-Z_][\w.-]*\/)[^\s,;]+\.[a-z]{1,4})/gm,
  );
  for (const match of pathMatches) {
    if (match[1]) {
      paths.add(match[1]);
    }
  }

  return Array.from(paths);
}

// ─── Utility ────────────────────────────────────────────────

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.substring(0, maxLen - 1) + "…";
}

// ─── Briefing Formatting ────────────────────────────────────

/**
 * Format a TurnBriefing into a string for injection into system prompt.
 */
export function formatBriefing(briefing: TurnBriefing): string {
  const lines: string[] = [];

  lines.push(`Turn: ${briefing.turnNumber}`);

  if (briefing.priorTaskSummary) {
    lines.push(`\nPrevious work:\n${briefing.priorTaskSummary}`);
  }

  if (briefing.planSteps && briefing.planSteps.length > 0) {
    const planLines = briefing.planSteps
      .map((s) => `- [${s.status}] ${s.description}`)
      .join("\n");
    lines.push(`\nPlan status:\n${planLines}`);
  }

  if (briefing.activeContext) {
    lines.push(`\nContext:\n${briefing.activeContext}`);
  }

  if (briefing.pendingWork) {
    lines.push(`\nPending:\n${briefing.pendingWork}`);
  }

  if (briefing.keyArtifacts.length > 0) {
    lines.push(`\nKey files: ${briefing.keyArtifacts.join(", ")}`);
  }

  return lines.join("\n");
}
