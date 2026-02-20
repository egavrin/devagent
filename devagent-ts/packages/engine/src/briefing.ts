/**
 * Turn briefing synthesis — creates structured context summaries
 * for fresh TaskLoop instances between interactive turns.
 *
 * Two strategies:
 *   - "heuristic": Fast extraction from messages without LLM call (~1ms)
 *   - "llm": LLM-based structured summary (~2-5s, higher quality)
 *   - "auto": LLM if turn had 5+ tool calls, heuristic otherwise
 *
 * Inspired by:
 *   - Microsoft/Salesforce paper: "Don't pass raw history to executing LLM"
 *   - Codex CLI: Pre-sampling compaction with checkpoint prompt
 *   - OpenCode: Structured compaction template (goal, discoveries, accomplished, files)
 *   - Claude Code: Sub-agents start fresh with focused prompts
 */

import type { Message, LLMProvider } from "@devagent/core";
import { MessageRole } from "@devagent/core";

// ─── Types ──────────────────────────────────────────────────

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

## Pending
What remains to be done. Be specific about next steps. Bullet points. If everything is done, say "Nothing pending."

## Relevant Files
Files read, modified, or discovered. Format: \`path\` — role/status. One per line.`;

// ─── Main Function ──────────────────────────────────────────

/**
 * Synthesize a structured briefing from completed turn messages.
 * Called at the end of each interactive turn, before the next turn starts.
 */
export async function synthesizeBriefing(
  messages: ReadonlyArray<Message>,
  turnNumber: number,
  options?: SynthesizeBriefingOptions,
): Promise<TurnBriefing> {
  const strategy = options?.strategy ?? "auto";
  const maxChars = options?.maxChars ?? DEFAULT_MAX_CHARS;

  if (strategy === "heuristic") {
    return extractHeuristicBriefing(messages, turnNumber, maxChars);
  }

  if (strategy === "llm" && options?.provider) {
    return synthesizeLLMBriefing(messages, turnNumber, options.provider, maxChars);
  }

  if (strategy === "auto") {
    const toolCallCount = countToolCalls(messages);
    if (toolCallCount >= LLM_TOOL_CALL_THRESHOLD && options?.provider) {
      try {
        return await synthesizeLLMBriefing(
          messages,
          turnNumber,
          options.provider,
          maxChars,
        );
      } catch {
        // Fall back to heuristic if LLM fails
        return extractHeuristicBriefing(messages, turnNumber, maxChars);
      }
    }
    return extractHeuristicBriefing(messages, turnNumber, maxChars);
  }

  // Fallback
  return extractHeuristicBriefing(messages, turnNumber, maxChars);
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

  if (planSteps) {
    summaryParts.push(`Plan: ${planSteps}`);
  }

  if (finalResponse) {
    summaryParts.push(`Result: ${truncate(finalResponse, 500)}`);
  }

  const priorTaskSummary = truncate(summaryParts.join("\n"), maxChars * 0.5);

  // Build active context from tool results (errors, key findings)
  const activeContext = extractActiveContext(messages, maxChars * 0.3);

  // Pending work: check if the final response indicates incomplete work
  const pendingWork = extractPendingWork(messages);

  return {
    turnNumber,
    priorTaskSummary,
    activeContext,
    pendingWork,
    keyArtifacts: artifacts.slice(0, 20), // Cap at 20 files
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

    let line: string;
    switch (msg.role) {
      case MessageRole.SYSTEM:
        // Skip system messages (they're the system prompt)
        continue;

      case MessageRole.USER:
        line = `[User]: ${truncate(msg.content ?? "", 500)}`;
        break;

      case MessageRole.ASSISTANT:
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          const toolNames = msg.toolCalls.map((tc) => tc.name).join(", ");
          line = `[Assistant]: Called tools: ${toolNames}`;
          if (msg.content?.trim()) {
            line += `\n  Text: ${truncate(msg.content, 200)}`;
          }
        } else {
          line = `[Assistant]: ${truncate(msg.content ?? "", 500)}`;
        }
        break;

      case MessageRole.TOOL:
        // Only include errors and brief summaries
        if (msg.content?.startsWith("Error:")) {
          line = `[Tool result]: ${truncate(msg.content, 200)}`;
        } else {
          // Truncate heavily — tool output is the main context bloat
          line = `[Tool result]: ${truncate(msg.content ?? "", 100)}`;
        }
        break;

      default:
        continue;
    }

    lines.push(line);
    totalChars += line.length;
  }

  return lines.join("\n");
}

/**
 * Parse LLM structured briefing response into TurnBriefing.
 */
function parseLLMBriefing(
  response: string,
  turnNumber: number,
  messages: ReadonlyArray<Message>,
): TurnBriefing {
  // Extract sections by heading
  const goalMatch = response.match(/## Goal\s*\n([\s\S]*?)(?=\n## |$)/);
  const decisionsMatch = response.match(
    /## Key Decisions[^\n]*\n([\s\S]*?)(?=\n## |$)/,
  );
  const accomplishedMatch = response.match(
    /## Accomplished\s*\n([\s\S]*?)(?=\n## |$)/,
  );
  const pendingMatch = response.match(/## Pending\s*\n([\s\S]*?)(?=\n## |$)/);
  const filesMatch = response.match(
    /## Relevant Files\s*\n([\s\S]*?)(?=\n## |$)/,
  );

  const goal = goalMatch?.[1]?.trim() ?? "";
  const decisions = decisionsMatch?.[1]?.trim() ?? "";
  const accomplished = accomplishedMatch?.[1]?.trim() ?? "";
  const pending = pendingMatch?.[1]?.trim() ?? "";
  const files = filesMatch?.[1]?.trim() ?? "";

  // Build priorTaskSummary from goal + accomplished
  const summaryParts: string[] = [];
  if (goal) summaryParts.push(`Goal: ${goal}`);
  if (accomplished) summaryParts.push(`Done: ${accomplished}`);
  const priorTaskSummary = summaryParts.join("\n");

  // activeContext from decisions
  const activeContext = decisions;

  // pendingWork
  const pendingWork =
    pending && !pending.toLowerCase().includes("nothing pending")
      ? pending
      : null;

  // keyArtifacts from files section + heuristic extraction
  const fileArtifacts = extractFilePathsFromText(files);
  const heuristicArtifacts = extractArtifacts(messages);
  const allArtifacts = [
    ...new Set([...fileArtifacts, ...heuristicArtifacts]),
  ];

  return {
    turnNumber,
    priorTaskSummary,
    activeContext,
    pendingWork,
    keyArtifacts: allArtifacts.slice(0, 20),
  };
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
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        // Common param names for file paths
        for (const key of ["path", "file_path", "file", "filename"]) {
          const val = tc.arguments[key];
          if (typeof val === "string" && val.trim()) {
            paths.add(val);
          }
        }
        // Pattern for find_files
        if (tc.arguments["pattern"] && typeof tc.arguments["pattern"] === "string") {
          // Don't add patterns — they're not file paths
        }
      }
    }
  }

  return Array.from(paths);
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

function extractPlanSteps(messages: ReadonlyArray<Message>): string | null {
  // Look for update_plan tool calls
  for (const msg of messages) {
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        if (tc.name === "update_plan") {
          const steps = tc.arguments["steps"];
          if (Array.isArray(steps)) {
            return (steps as Array<Record<string, unknown>>)
              .map(
                (s) =>
                  `[${(s["status"] as string) ?? "?"}] ${(s["description"] as string) ?? ""}`,
              )
              .join("; ");
          }
        }
      }
    }
  }
  return null;
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
