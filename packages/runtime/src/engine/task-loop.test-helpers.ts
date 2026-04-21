import type { DevAgentConfig, LLMProvider, StreamChunk, ToolSpec } from "../core/index.js";
import { ApprovalMode } from "../core/index.js";

export function createMockProvider(
  responses: Array<StreamChunk[]>,
): LLMProvider {
  let callIndex = 0;
  return {
    id: "mock",
    async *chat(): AsyncIterable<StreamChunk> {
      const chunks = responses[callIndex] ?? [];
      callIndex++;
      for (const chunk of chunks) {
        yield chunk;
      }
    },
    abort() {},
  };
}

export function makeConfig(overrides?: Partial<DevAgentConfig>): DevAgentConfig {
  return {
    provider: "mock",
    model: "mock-model",
    providers: {},
    approval: {
      mode: ApprovalMode.FULL_AUTO,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 10,
      maxContextTokens: 100_000,
      responseHeadroom: 2_000,
      costWarningThreshold: 1.0,
      enableCostTracking: true,
    },
    context: {
      pruningStrategy: "hybrid",
      triggerRatio: 0.8,
      keepRecentMessages: 10,
    },
    ...overrides,
  };
}

export function makeEchoTool(): ToolSpec {
  return {
    name: "echo",
    description: "Echo the input",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        text: { type: "string" },
      },
      required: ["text"],
    },
    resultSchema: { type: "object" },
    handler: async (params) => ({
      success: true,
      output: `Echo: ${params["text"] as string}`,
      error: null,
      artifacts: [],
    }),
  };
}
