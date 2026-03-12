/**
 * Pre-compaction knowledge extractor — extracts structured domain
 * knowledge from the conversation before Phase 2 context compaction.
 *
 * Follows the compaction-judge.ts pattern: LLM call with focused
 * system prompt, JSON response parsing, graceful degradation.
 *
 * The extracted knowledge entries are stored in SessionState and
 * survive all future compactions, preventing the re-read feedback loop.
 */

import type { LLMProvider, Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import type { SessionState } from "./session-state.js";
import {
  collectStreamText,
  parseJudgeResponse,
  formatMessageForJudge,
  buildSessionStateContext,
} from "./llm-judge.js";

// ─── Constants ──────────────────────────────────────────────

/** Max recent messages to include in the extraction context. */
const MAX_RECENT_MESSAGES = 20;

// ─── Types ──────────────────────────────────────────────────

export interface KnowledgeExtractionEntry {
  readonly key: string;
  readonly content: string;
}

export interface KnowledgeExtractionResult {
  readonly entries: KnowledgeExtractionEntry[];
}

// ─── System prompt ──────────────────────────────────────────

export const KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = `You extract structured domain knowledge from a coding session before context compaction.

Your goal: preserve the semantic understanding the assistant has built so it can continue without re-reading files or re-scanning the codebase.

Extract knowledge into these categories (use EXACTLY these keys):

- INVENTORY: Items, patterns, files, or symbols discovered. Include counts and specific names.
  Example: "Found 5 ANI descriptor files: type_a.cpp, type_b.cpp, resolver.cpp, validator.cpp, transform.cpp"

- DECISIONS: Approach chosen and why. Include rejected alternatives if known.
  Example: "Using visitor pattern for AST transformation (rejected: string regex due to nested types)"

- PROGRESS: What's done and what remains. Include specific file paths.
  Example: "Transformed 3/7 files (type_a.cpp, type_b.cpp, resolver.cpp). Remaining: validator.cpp, transform.cpp, tests/test_ani.cpp, build.cmake"

- NEXT_ACTION: The specific next step the assistant should take. Include file, function, or line if possible.
  Example: "Open validator.cpp and apply the same descriptor transformation pattern used in type_a.cpp"

Rules:
- Only include categories where you have concrete information. Omit categories with no data.
- Be specific: include file names, function names, counts, line numbers.
- Keep each entry under 500 characters — density over verbosity.
- Do NOT include speculative or uncertain information.

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"entries": [{"key": "INVENTORY", "content": "..."}, {"key": "DECISIONS", "content": "..."}, ...]}`;

// ─── Extraction function ────────────────────────────────────

/**
 * Extract structured knowledge from the current conversation context
 * before Phase 2 compaction. Returns null on any failure (non-fatal).
 */
export async function extractPreCompactionKnowledge(
  provider: LLMProvider,
  preCompactionSummary: string,
  sessionState: SessionState | null,
  recentMessages: ReadonlyArray<Message>,
  originalTask: string | null,
): Promise<KnowledgeExtractionResult | null> {
  try {
    const parts: string[] = [];

    if (originalTask) {
      parts.push(`## Original task\n${originalTask}`);
    }

    parts.push(`## Pre-compaction summary\n${preCompactionSummary}`);
    parts.push(`## Session state\n${buildSessionStateContext(sessionState)}`);

    // Include last N messages for recent context
    const recentSlice = recentMessages.slice(-MAX_RECENT_MESSAGES);
    if (recentSlice.length > 0) {
      const formatted = recentSlice
        .map((m) => formatMessageForJudge(m))
        .join("\n\n");
      parts.push(`## Recent messages (last ${recentSlice.length})\n${formatted}`);
    }

    parts.push("\nExtract the domain knowledge. Respond with JSON only.");

    const messages = [
      { role: MessageRole.SYSTEM as const, content: KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    const parsed = parseJudgeResponse<{ entries?: unknown[] }>(responseText);

    if (!parsed || !Array.isArray(parsed.entries)) {
      return null;
    }

    // Filter and validate entries
    const validEntries: KnowledgeExtractionEntry[] = [];
    for (const entry of parsed.entries) {
      if (
        typeof entry === "object" &&
        entry !== null &&
        "key" in entry &&
        "content" in entry &&
        typeof (entry as Record<string, unknown>).key === "string" &&
        typeof (entry as Record<string, unknown>).content === "string" &&
        ((entry as Record<string, unknown>).key as string).length > 0 &&
        ((entry as Record<string, unknown>).content as string).length > 0
      ) {
        validEntries.push({
          key: (entry as Record<string, unknown>).key as string,
          content: (entry as Record<string, unknown>).content as string,
        });
      }
    }

    if (validEntries.length === 0) {
      return null;
    }

    return { entries: validEntries };
  } catch {
    return null;
  }
}
