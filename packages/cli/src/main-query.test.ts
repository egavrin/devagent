import {
  ApprovalGate,
  EventBus,
  SessionState,
  ToolRegistry,
  type DevAgentConfig,
  type LLMProvider,
  type StreamChunk,
} from "@devagent/runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { RunSingleQueryOptions } from "./main-types.js";

function createTestConfig(): DevAgentConfig {
  return {
    provider: "openai",
    model: "gpt-5",
    approval: { mode: "default" },
    budget: {
      maxIterations: 10,
      maxContextTokens: 100_000,
      responseHeadroom: 2_000,
      costWarningThreshold: 1.0,
      enableCostTracking: true,
    },
    context: { turnIsolation: false },
    logging: { enabled: false },
  } as unknown as DevAgentConfig;
}

function deferred(): {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function createRunOptions(provider: LLMProvider): RunSingleQueryOptions {
  const config = createTestConfig();
  const bus = new EventBus();
  return {
    query: "inspect",
    provider,
    toolRegistry: new ToolRegistry(),
    bus,
    gate: new ApprovalGate(config.approval, bus),
    config,
    repoRoot: "/tmp",
    mode: "act",
    skills: { list: () => [] } as RunSingleQueryOptions["skills"],
    contextManager: { setSummarizeCallback: () => {} } as RunSingleQueryOptions["contextManager"],
    doubleCheck: {} as RunSingleQueryOptions["doubleCheck"],
    initialMessages: undefined,
    verbosity: "quiet",
    verbosityConfig: { base: "quiet", categories: new Set() },
    sessionState: new SessionState(),
  };
}

function createProvider(): { readonly provider: LLMProvider; readonly chat: ReturnType<typeof vi.fn>; readonly abort: ReturnType<typeof vi.fn> } {
  const chat = vi.fn(async function* (): AsyncIterable<StreamChunk> {
    yield { type: "text", content: "done" };
    yield { type: "done", content: "" };
  });
  const abort = vi.fn();
  return {
    chat,
    abort,
    provider: {
      id: "test-provider",
      chat,
      abort,
    },
  };
}

describe("runTuiQuery cancellation", () => {
  afterEach(() => {
    vi.doUnmock("./prompt-commands.js");
    vi.resetModules();
    vi.restoreAllMocks();
  });

  it("honors cancellation while first-turn query preparation is pending", async () => {
    const prep = deferred();
    vi.doMock("./prompt-commands.js", async (importOriginal) => {
      const actual = await importOriginal<typeof import("./prompt-commands.js")>();
      return {
        ...actual,
        preparePromptCommandQuery: vi.fn(async () => {
          await prep.promise;
          return null;
        }),
      };
    });
    const { provider, chat, abort } = createProvider();
    const { runTuiQuery, abortTuiQuery, resetTuiLoop } = await import("./main-query.js");

    resetTuiLoop();
    const run = runTuiQuery(createRunOptions(provider));
    await Promise.resolve();

    abortTuiQuery();
    prep.resolve();

    const result = await run;
    expect(abort).toHaveBeenCalledTimes(1);
    expect(chat).not.toHaveBeenCalled();
    expect(result.status).toBe("aborted");
    resetTuiLoop();
  });

  it("preserves cancellation during query preparation for an existing TUI loop", async () => {
    const secondPrep = deferred();
    let prepareCallCount = 0;
    vi.doMock("./prompt-commands.js", async (importOriginal) => {
      const actual = await importOriginal<typeof import("./prompt-commands.js")>();
      return {
        ...actual,
        preparePromptCommandQuery: vi.fn(async () => {
          prepareCallCount += 1;
          if (prepareCallCount === 2) await secondPrep.promise;
          return null;
        }),
      };
    });
    const { provider, chat } = createProvider();
    const { runTuiQuery, abortTuiQuery, resetTuiLoop } = await import("./main-query.js");

    resetTuiLoop();
    await expect(runTuiQuery(createRunOptions(provider))).resolves.toMatchObject({ status: "success" });
    expect(chat).toHaveBeenCalledTimes(1);

    const secondRun = runTuiQuery(createRunOptions(provider));
    await Promise.resolve();
    abortTuiQuery();
    secondPrep.resolve();

    await expect(secondRun).resolves.toMatchObject({ status: "aborted" });
    expect(chat).toHaveBeenCalledTimes(1);
    resetTuiLoop();
  });
});
