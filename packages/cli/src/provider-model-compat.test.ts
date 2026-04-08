import { beforeAll, describe, expect, it } from "vitest";
import { loadModelRegistry } from "@devagent/runtime";
import {
  formatProviderModelCompatibilityError,
  formatProviderModelCompatibilityHint,
  getProviderModelCompatibilityIssue,
} from "./provider-model-compat.js";

describe("provider/model compatibility", () => {
  beforeAll(() => {
    loadModelRegistry();
  });

  it("reports a mismatch when the model belongs to another provider", () => {
    const issue = getProviderModelCompatibilityIssue("openai", "cortex");
    expect(issue).toEqual({
      model: "cortex",
      configuredProvider: "openai",
      supportedProviders: ["devagent-api"],
    });
    expect(formatProviderModelCompatibilityError(issue!)).toContain(
      'Configured model "cortex" is not registered for provider "openai".',
    );
    expect(formatProviderModelCompatibilityHint(issue!)).toBe(
      'Try "--provider devagent-api --model cortex" for the deployed Devagent API gateway.',
    );
  });

  it("returns no issue for a valid devagent-api and cortex pairing", () => {
    expect(getProviderModelCompatibilityIssue("devagent-api", "cortex")).toBeUndefined();
  });

  it("allows shared model names across providers", () => {
    expect(getProviderModelCompatibilityIssue("openai", "gpt-4.1")).toBeUndefined();
    expect(getProviderModelCompatibilityIssue("github-copilot", "gpt-4.1")).toBeUndefined();
  });
});
