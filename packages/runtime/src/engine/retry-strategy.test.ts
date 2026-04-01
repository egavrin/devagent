import { describe, it, expect, vi, beforeEach } from "vitest";
import { retryWithStrategy } from "./retry-strategy.js";
import type { RetryOptions, RetryResult } from "./retry-strategy.js";
import {
  ProviderError,
  RateLimitError,
  ProviderConnectionError,
  ProviderTimeoutError,
} from "../core/index.js";
import { OverloadedError, MaxOutputTokensError } from "../core/errors.js";

// ─── Helpers ────────────────────────────────────────────────

/** Stub sleep so tests run instantly. */
vi.mock("./retry-strategy.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("./retry-strategy.js")>();
  // We need the real implementation but with mocked timers
  return original;
});

beforeEach(() => {
  vi.useFakeTimers();
});

import { afterEach } from "vitest";
afterEach(() => {
  vi.useRealTimers();
});

/** Run retryWithStrategy while advancing fake timers to unblock sleeps. */
async function runWithTimers<T>(
  op: () => Promise<T>,
  opts?: RetryOptions,
): Promise<RetryResult<T>> {
  const resultPromise = retryWithStrategy(op, opts);
  // Advance timers to flush any pending sleeps
  for (let i = 0; i < 10; i++) {
    await vi.advanceTimersByTimeAsync(60_000);
  }
  return resultPromise;
}

// ─── Tests ──────────────────────────────────────────────────

describe("retryWithStrategy", () => {
  it("returns success immediately on successful operation", async () => {
    const result = await runWithTimers(async () => "hello");

    expect(result.success).toBe(true);
    expect(result.value).toBe("hello");
    expect(result.shouldFallback).toBe(false);
    expect(result.shouldCompact).toBe(false);
  });

  it("retries on RateLimitError with backoff", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      if (attempts < 3) throw new RateLimitError("rate limited", 100);
      return "done";
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(true);
    expect(result.value).toBe("done");
    expect(attempts).toBe(3);
  });

  it("triggers shouldFallback after exhausting rate limit retries", async () => {
    const op = async () => {
      throw new RateLimitError("rate limited");
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(false);
    expect(result.shouldFallback).toBe(true);
    expect(result.shouldCompact).toBe(false);
    expect(result.error).toBeInstanceOf(RateLimitError);
  });

  it("retries OverloadedError up to 3 times then returns shouldFallback", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      throw new OverloadedError("overloaded");
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(false);
    expect(result.shouldFallback).toBe(true);
    expect(result.error).toBeInstanceOf(OverloadedError);
    // Should not exceed MAX_OVERLOAD_RETRIES (3) + 1 initial
    expect(attempts).toBeLessThanOrEqual(4);
  });

  it("retries ProviderConnectionError", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      if (attempts < 3) throw new ProviderConnectionError("ECONNRESET");
      return "reconnected";
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(true);
    expect(result.value).toBe("reconnected");
    expect(attempts).toBe(3);
  });

  it("returns shouldCompact for context overflow error", async () => {
    const op = async () => {
      throw new ProviderError("maximum context length exceeded");
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(false);
    expect(result.shouldCompact).toBe(true);
    expect(result.shouldFallback).toBe(false);
  });

  it("does not return shouldCompact if overflowCompactionUsed", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      throw new ProviderError("maximum context length exceeded");
    };

    const result = await runWithTimers(op, { overflowCompactionUsed: true });

    expect(result.success).toBe(false);
    expect(result.shouldCompact).toBe(false);
    // Should have retried as a generic provider error
    expect(attempts).toBeGreaterThan(1);
  });

  it("returns immediately for MaxOutputTokensError without retry", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      throw new MaxOutputTokensError("max output tokens", "partial");
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(false);
    expect(result.shouldFallback).toBe(false);
    expect(result.shouldCompact).toBe(false);
    expect(result.error).toBeInstanceOf(MaxOutputTokensError);
    expect(attempts).toBe(1);
  });

  it("throws non-provider errors through without catching", async () => {
    const op = async () => {
      throw new TypeError("not a provider error");
    };

    // retryWithStrategy re-throws non-ProviderError errors directly
    await expect(retryWithStrategy(op)).rejects.toThrow(TypeError);
  });

  it("calls onRetry callback with correct params", async () => {
    const onRetry = vi.fn();
    let attempts = 0;
    const op = async () => {
      attempts++;
      if (attempts < 3) throw new RateLimitError("rate limited", 200);
      return "ok";
    };

    await runWithTimers(op, { onRetry });

    expect(onRetry).toHaveBeenCalledTimes(2);
    // First retry: attempt 1
    expect(onRetry.mock.calls[0]![0]).toBeInstanceOf(RateLimitError);
    expect(onRetry.mock.calls[0]![1]).toBe(1);
    expect(typeof onRetry.mock.calls[0]![2]).toBe("number");
    // Second retry: attempt 2
    expect(onRetry.mock.calls[1]![1]).toBe(2);
  });

  it("retries ProviderTimeoutError", async () => {
    let attempts = 0;
    const op = async () => {
      attempts++;
      if (attempts < 2) throw new ProviderTimeoutError("timed out");
      return "ok";
    };

    const result = await runWithTimers(op);

    expect(result.success).toBe(true);
    expect(attempts).toBe(2);
  });
});
