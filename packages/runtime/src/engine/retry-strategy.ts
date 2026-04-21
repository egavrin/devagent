/**
 * Sophisticated retry strategy for provider errors.
 * Classifies errors by type and applies appropriate retry logic:
 * - RateLimitError (429): respect retry-after, exponential backoff with jitter
 * - OverloadedError (529): max 3 retries, longer backoff
 * - ProviderConnectionError: reconnect retry with exponential backoff
 * - MaxOutputTokensError: increase maxTokens and retry once
 * - Context overflow: delegate to compaction (handled externally)
 *
 * Inspired by claude-code-src withRetry.ts pattern.
 */

import { OverloadedError, MaxOutputTokensError } from "../core/errors.js";
import {
  ProviderError,
  RateLimitError,
  ProviderConnectionError,
  ProviderTimeoutError,
} from "../core/index.js";

// Import new error types — these are added in the same PR
// to avoid circular dependency, we import them directly.

// ─── Constants ──────────────────────────────────────────────

/** Base delay for exponential backoff (ms). */
const BASE_DELAY_MS = 500;

/** Maximum delay between retries (ms). */
const MAX_DELAY_MS = 30_000;

/** Max retries for rate limit errors. */
const MAX_RATE_LIMIT_RETRIES = 5;

/** Max retries for overload (529) errors. */
const MAX_OVERLOAD_RETRIES = 3;

/** Max retries for connection errors. */
const MAX_CONNECTION_RETRIES = 4;

/** Max retries for generic provider errors. */
const MAX_GENERIC_RETRIES = 3;

// ─── Types ──────────────────────────────────────────────────

export interface RetryResult<T> {
  readonly success: boolean;
  readonly value?: T;
  readonly error?: ProviderError;
  /** True if a fallback model should be tried. */
  readonly shouldFallback: boolean;
  /** True if context overflow compaction should be attempted. */
  readonly shouldCompact: boolean;
}

export interface RetryOptions {
  /** Callback invoked before each retry with the error and delay. */
  readonly onRetry?: (error: ProviderError, attempt: number, delayMs: number) => void;
  /** Whether context overflow has already been handled this iteration. */
  readonly overflowCompactionUsed?: boolean;
}

// ─── Retry Strategy ─────────────────────────────────────────

/**
 * Execute an async operation with error-type-aware retry logic.
 * Returns a RetryResult indicating success, failure with fallback hint,
 * or failure with compaction hint.
 */
export async function retryWithStrategy<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {},
): Promise<RetryResult<T>> {
  let lastError: ProviderError | undefined;
  let consecutiveOverloads = 0;

  // Determine max attempts based on error type (resolved per-error)
  for (let attempt = 0; attempt < MAX_RATE_LIMIT_RETRIES + 1; attempt++) {
    try {
      const value = await operation();
      return { success: true, value, shouldFallback: false, shouldCompact: false };
    } catch (err) {
      const next = handleRetryError(err, attempt, consecutiveOverloads, options);
      if (next.action === "throw") throw err;
      lastError = next.error;
      if (next.action === "return") return next.result;
      if (next.action === "break") break;
      consecutiveOverloads = next.consecutiveOverloads;
      options.onRetry?.(next.error, attempt + 1, next.delayMs);
      await sleep(next.delayMs);
    }
  }

  // All retries exhausted — suggest fallback if it was a capacity issue
  const shouldFallback = lastError instanceof OverloadedError ||
    lastError instanceof RateLimitError;

  return {
    success: false,
    error: lastError,
    shouldFallback,
    shouldCompact: false,
  };
}

// ─── Helpers ────────────────────────────────────────────────

type RetryErrorAction =
  | { readonly action: "throw" }
  | { readonly action: "break"; readonly error: ProviderError }
  | { readonly action: "return"; readonly error: ProviderError; readonly result: RetryResult<never> }
  | {
      readonly action: "retry";
      readonly error: ProviderError;
      readonly delayMs: number;
      readonly consecutiveOverloads: number;
    };

function handleRetryError(
  err: unknown,
  attempt: number,
  consecutiveOverloads: number,
  options: RetryOptions,
): RetryErrorAction {
  if (!(err instanceof ProviderError)) return { action: "throw" };
  if (isContextOverflowError(err.message) && !options.overflowCompactionUsed) {
    return {
      action: "return",
      error: err,
      result: { success: false, error: err, shouldFallback: false, shouldCompact: true },
    };
  }
  if (err instanceof MaxOutputTokensError) {
    return {
      action: "return",
      error: err,
      result: { success: false, error: err, shouldFallback: false, shouldCompact: false },
    };
  }

  const { maxRetries, delayMs } = getRetryParams(err, attempt);
  if (attempt >= maxRetries) return { action: "break", error: err };

  const overloads = nextOverloadCount(err, consecutiveOverloads);
  if (overloads >= MAX_OVERLOAD_RETRIES) {
    return {
      action: "return",
      error: err,
      result: { success: false, error: err, shouldFallback: true, shouldCompact: false },
    };
  }
  return { action: "retry", error: err, delayMs, consecutiveOverloads: overloads };
}

function nextOverloadCount(error: ProviderError, current: number): number {
  return error instanceof OverloadedError ? current + 1 : 0;
}

function getRetryParams(
  error: ProviderError,
  attempt: number,
): { maxRetries: number; delayMs: number } {
  if (error instanceof RateLimitError) {
    // Respect retry-after header if available
    const retryAfter = error.retryAfterMs;
    const baseDelay = retryAfter ?? exponentialBackoff(attempt, BASE_DELAY_MS * 2);
    return {
      maxRetries: MAX_RATE_LIMIT_RETRIES,
      delayMs: Math.min(baseDelay, MAX_DELAY_MS),
    };
  }

  if (error instanceof OverloadedError) {
    // Longer backoff for overloaded servers
    return {
      maxRetries: MAX_OVERLOAD_RETRIES,
      delayMs: Math.min(exponentialBackoff(attempt, BASE_DELAY_MS * 4), MAX_DELAY_MS),
    };
  }

  if (error instanceof ProviderConnectionError || error instanceof ProviderTimeoutError) {
    return {
      maxRetries: MAX_CONNECTION_RETRIES,
      delayMs: Math.min(exponentialBackoff(attempt, BASE_DELAY_MS), MAX_DELAY_MS),
    };
  }

  // Generic provider error — limited retries
  return {
    maxRetries: MAX_GENERIC_RETRIES,
    delayMs: Math.min(exponentialBackoff(attempt, BASE_DELAY_MS), MAX_DELAY_MS),
  };
}

/**
 * Exponential backoff with jitter.
 * delay = baseDelay * 2^attempt + random jitter (0-50% of delay)
 */
function exponentialBackoff(attempt: number, baseDelay: number): number {
  const delay = baseDelay * Math.pow(2, attempt);
  const jitter = delay * 0.5 * Math.random();
  return Math.round(delay + jitter);
}

function isContextOverflowError(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("context length") ||
    normalized.includes("maximum context") ||
    normalized.includes("max context") ||
    normalized.includes("token limit") ||
    normalized.includes("too many tokens") ||
    normalized.includes("prompt is too long")
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
