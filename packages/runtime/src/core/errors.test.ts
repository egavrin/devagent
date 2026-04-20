import { describe, it, expect } from "vitest";

import {
  DevAgentError,
  ConfigError,
  ProviderError,
  RateLimitError,
  ToolError,
  ToolNotFoundError,
  ApprovalDeniedError,
  BudgetExceededError,
} from "./errors.js";

describe("Error hierarchy", () => {
  it("DevAgentError has code and message", () => {
    const err = new DevAgentError("test error", "TEST");
    expect(err.message).toBe("test error");
    expect(err.code).toBe("TEST");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(DevAgentError);
  });

  it("ConfigError is a DevAgentError", () => {
    const err = new ConfigError("bad config");
    expect(err).toBeInstanceOf(DevAgentError);
    expect(err.code).toBe("CONFIG_ERROR");
  });

  it("RateLimitError carries retryAfterMs", () => {
    const err = new RateLimitError("slow down", 5000);
    expect(err).toBeInstanceOf(ProviderError);
    expect(err).toBeInstanceOf(DevAgentError);
    expect(err.retryAfterMs).toBe(5000);
    expect(err.code).toBe("RATE_LIMIT");
  });

  it("ToolNotFoundError includes tool name in message", () => {
    const err = new ToolNotFoundError("read_file");
    expect(err).toBeInstanceOf(ToolError);
    expect(err.toolName).toBe("read_file");
    expect(err.message).toContain("read_file");
    expect(err.message).toContain("not found");
  });

  it("ApprovalDeniedError includes tool name", () => {
    const err = new ApprovalDeniedError("run_command", "shell execution");
    expect(err.toolName).toBe("run_command");
    expect(err.code).toBe("APPROVAL_DENIED");
  });

  it("BudgetExceededError is a DevAgentError", () => {
    const err = new BudgetExceededError("max iterations reached");
    expect(err).toBeInstanceOf(DevAgentError);
    expect(err.code).toBe("BUDGET_EXCEEDED");
  });
});
