import { formatDuration } from "@devagent/runtime";

import { dim, red, yellow } from "./format-colors.js";

interface RecentToolResult {
  readonly name: string;
  readonly success: boolean;
  readonly durationMs: number;
}

interface EnrichedErrorContext {
  readonly message: string;
  readonly recentTools: ReadonlyArray<RecentToolResult>;
  readonly suggestion: string | null;
}

export function inferErrorSuggestion(
  message: string,
  recentTools: ReadonlyArray<RecentToolResult>,
): string | null {
  const lower = message.toLowerCase();
  return inferMessageSuggestion(lower) ?? inferRepeatedToolFailureSuggestion(recentTools);
}

const ERROR_SUGGESTION_PATTERNS: ReadonlyArray<{
  readonly patterns: ReadonlyArray<string>;
  readonly suggestion: string;
}> = [
  {
    patterns: ["rate limit", "429", "rate_limit"],
    suggestion: "Rate limited — wait a moment or switch to a different provider.",
  },
  {
    patterns: ["context length", "token limit", "maximum context"],
    suggestion: "Context limit reached — use /clear or reduce output length.",
  },
  {
    patterns: ["timeout", "etimedout"],
    suggestion: "Request timed out — check network connectivity.",
  },
];

function inferMessageSuggestion(lowerMessage: string): string | null {
  return ERROR_SUGGESTION_PATTERNS.find((entry) =>
    entry.patterns.some((pattern) => lowerMessage.includes(pattern)),
  )?.suggestion ?? null;
}

function inferRepeatedToolFailureSuggestion(recentTools: ReadonlyArray<RecentToolResult>): string | null {
  const failedTools = recentTools.filter((tool) => !tool.success);
  const lastName = failedTools[failedTools.length - 1]?.name;
  if (!lastName) {
    return null;
  }
  const repeatedFailures = failedTools.filter((tool) => tool.name === lastName);
  return repeatedFailures.length >= 2
    ? `Tool "${lastName}" has failed repeatedly. Try a different approach.`
    : null;
}

export function formatEnrichedError(ctx: EnrichedErrorContext): string {
  const lines: string[] = [];
  lines.push(`${red("✗")} ${red(ctx.message)}`);

  if (ctx.recentTools.length > 0) {
    lines.push(dim("  Recent tool chain:"));
    for (const tool of ctx.recentTools) {
      const icon = tool.success ? "✓" : red("✗");
      const duration = formatDuration(tool.durationMs);
      lines.push(`    ${icon} ${tool.name} (${duration})`);
    }
  }

  if (ctx.suggestion) {
    lines.push(`  ${yellow("Suggestion:")} ${ctx.suggestion}`);
  }

  return lines.join("\n");
}

interface CompactionResultData {
  readonly tokensBefore: number;
  readonly estimatedTokens: number;
  readonly removedCount: number;
  readonly prunedCount?: number;
}

export function formatCompactionResult(data: CompactionResultData): string {
  const parts = [
    formatCompactionTokenChange(data),
    formatCompactionActivity(data),
  ].filter(Boolean);
  return dim(`[context] Compacted: ${parts.join(", ")}`);
}

function formatCompactionTokenChange(data: CompactionResultData): string {
  const kAfter = Math.round(data.estimatedTokens / 1000);
  if (data.tokensBefore <= 0) {
    return `~${kAfter}k tokens remaining`;
  }
  const kBefore = Math.round(data.tokensBefore / 1000);
  const reduction = data.tokensBefore - data.estimatedTokens;
  if (reduction <= 0) {
    return `${kBefore}k → ${kAfter}k tokens (no reduction)`;
  }
  const pct = Math.round((reduction / data.tokensBefore) * 100);
  return `${kBefore}k → ${kAfter}k tokens (${pct}% reduction)`;
}

function formatCompactionActivity(data: CompactionResultData): string {
  const details: string[] = [];
  if (data.removedCount > 0) {
    details.push(`removed ${data.removedCount} msgs`);
  }
  if ((data.prunedCount ?? 0) > 0) {
    details.push(`pruned ${data.prunedCount} outputs`);
  }
  return details.join(", ");
}

export function formatContextGauge(estimatedTokens: number, maxTokens: number): string {
  if (maxTokens <= 0 || estimatedTokens <= 0) return "";
  const kEstimated = Math.round(estimatedTokens / 1000);
  const kMax = Math.round(maxTokens / 1000);
  const ratio = estimatedTokens / maxTokens;
  const label = `[tokens: ${kEstimated}k/${kMax}k]`;
  if (ratio > 0.8) return red(label);
  if (ratio > 0.6) return yellow(label);
  return dim(label);
}

export function formatReasoning(text: string): string | null {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (collapsed.length === 0) return null;

  const sentenceEnd = collapsed.search(/[.!?]\s/);
  const display = sentenceEnd > 0 && sentenceEnd < 120
    ? collapsed.substring(0, sentenceEnd + 1)
    : truncateReasoning(collapsed);

  return `  ${dim("ℹ")} ${dim(display)}`;
}

function truncateReasoning(text: string): string {
  if (text.length <= 120) return text;
  return text.substring(0, 119) + "…";
}
