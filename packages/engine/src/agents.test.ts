import { describe, it, expect } from "vitest";
import { AgentType } from "@devagent/core";
import { AgentRegistry } from "./agents.js";

describe("AgentRegistry", () => {
  it("registers all four agent types", () => {
    const registry = new AgentRegistry();
    const allTypes = [
      AgentType.GENERAL,
      AgentType.REVIEWER,
      AgentType.ARCHITECT,
      AgentType.EXPLORE,
    ];

    for (const type of allTypes) {
      expect(registry.has(type)).toBe(true);
    }

    expect(registry.list()).toHaveLength(4);
  });

  it("general agent uses act mode with full tool categories", () => {
    const registry = new AgentRegistry();
    const def = registry.get(AgentType.GENERAL);

    expect(def.name).toBe("General");
    expect(def.defaultMode).toBe("act");
    expect(def.allowedToolCategories).toEqual([
      "readonly",
      "mutating",
      "workflow",
      "external",
    ]);
  });

  it("reviewer agent uses plan mode with readonly tools only", () => {
    const registry = new AgentRegistry();
    const def = registry.get(AgentType.REVIEWER);

    expect(def.name).toBe("Reviewer");
    expect(def.defaultMode).toBe("plan");
    expect(def.allowedToolCategories).toEqual(["readonly"]);
  });

  it("architect agent uses plan mode with readonly tools only", () => {
    const registry = new AgentRegistry();
    const def = registry.get(AgentType.ARCHITECT);

    expect(def.name).toBe("Architect");
    expect(def.defaultMode).toBe("plan");
    expect(def.allowedToolCategories).toEqual(["readonly"]);
  });

  it("explore agent uses act mode with readonly tools only", () => {
    const registry = new AgentRegistry();
    const def = registry.get(AgentType.EXPLORE);

    expect(def.name).toBe("Explore");
    expect(def.defaultMode).toBe("act");
    expect(def.allowedToolCategories).toEqual(["readonly"]);
  });

  it("throws for unknown agent type", () => {
    const registry = new AgentRegistry();
    expect(() => registry.get("unknown" as AgentType)).toThrow(
      "Unknown agent type: unknown",
    );
  });

  it("has() returns false for unknown type", () => {
    const registry = new AgentRegistry();
    expect(registry.has("nonexistent" as AgentType)).toBe(false);
  });

  it("each agent has a non-empty system prompt template", () => {
    const registry = new AgentRegistry();
    for (const def of registry.list()) {
      expect(def.systemPromptTemplate.length).toBeGreaterThan(0);
    }
  });

  it("register() adds a custom agent definition", () => {
    const registry = new AgentRegistry();
    const custom = {
      type: "custom" as AgentType,
      name: "Custom",
      description: "A custom agent",
      systemPromptTemplate: "You are custom.",
      defaultMode: "act" as const,
      allowedToolCategories: ["readonly"],
    };

    registry.register(custom);
    expect(registry.has("custom" as AgentType)).toBe(true);
    expect(registry.get("custom" as AgentType)).toEqual(custom);
    expect(registry.list()).toHaveLength(5);
  });
});
