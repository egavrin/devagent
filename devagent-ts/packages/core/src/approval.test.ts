import { describe, it, expect } from "vitest";
import { ApprovalGate } from "./approval.js";
import type { ApprovalRequest } from "./approval.js";
import type { ApprovalPolicy } from "./types.js";
import { ApprovalMode } from "./types.js";
import { EventBus } from "./events.js";

function makePolicy(overrides?: Partial<ApprovalPolicy>): ApprovalPolicy {
  return {
    mode: ApprovalMode.SUGGEST,
    autoApprovePlan: false,
    autoApproveCode: false,
    autoApproveShell: false,
    auditLog: false,
    toolOverrides: {},
    pathRules: [],
    ...overrides,
  };
}

function makeRequest(overrides?: Partial<ApprovalRequest>): ApprovalRequest {
  return {
    toolName: "write_file",
    toolCategory: "mutating",
    filePath: "/src/index.ts",
    description: "Write to /src/index.ts",
    ...overrides,
  };
}

describe("ApprovalGate", () => {
  describe("Suggest mode", () => {
    it("allows readonly tools", () => {
      const gate = new ApprovalGate(makePolicy());
      const decision = gate.decide(
        makeRequest({ toolCategory: "readonly", toolName: "read_file" }),
      );
      expect(decision).toBe("allow");
    });

    it("asks for mutating tools", () => {
      const gate = new ApprovalGate(makePolicy());
      const decision = gate.decide(makeRequest({ toolCategory: "mutating" }));
      expect(decision).toBe("ask");
    });

    it("asks for workflow tools", () => {
      const gate = new ApprovalGate(makePolicy());
      const decision = gate.decide(
        makeRequest({ toolCategory: "workflow", toolName: "run_command" }),
      );
      expect(decision).toBe("ask");
    });

    it("denies external tools", () => {
      const gate = new ApprovalGate(makePolicy());
      const decision = gate.decide(
        makeRequest({ toolCategory: "external", toolName: "web_search" }),
      );
      expect(decision).toBe("deny");
    });
  });

  describe("Auto-Edit mode", () => {
    it("allows readonly tools", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.AUTO_EDIT }),
      );
      const decision = gate.decide(
        makeRequest({ toolCategory: "readonly", toolName: "read_file" }),
      );
      expect(decision).toBe("allow");
    });

    it("allows mutating tools", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.AUTO_EDIT }),
      );
      const decision = gate.decide(makeRequest({ toolCategory: "mutating" }));
      expect(decision).toBe("allow");
    });

    it("asks for workflow tools (shell commands)", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.AUTO_EDIT }),
      );
      const decision = gate.decide(
        makeRequest({ toolCategory: "workflow", toolName: "run_command" }),
      );
      expect(decision).toBe("ask");
    });

    it("denies external tools", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.AUTO_EDIT }),
      );
      const decision = gate.decide(
        makeRequest({ toolCategory: "external", toolName: "web_search" }),
      );
      expect(decision).toBe("deny");
    });
  });

  describe("Full-Auto mode", () => {
    it("allows all tool categories", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.FULL_AUTO }),
      );

      expect(
        gate.decide(makeRequest({ toolCategory: "readonly" })),
      ).toBe("allow");
      expect(
        gate.decide(makeRequest({ toolCategory: "mutating" })),
      ).toBe("allow");
      expect(
        gate.decide(makeRequest({ toolCategory: "workflow" })),
      ).toBe("allow");
      expect(
        gate.decide(makeRequest({ toolCategory: "external" })),
      ).toBe("allow");
    });
  });

  describe("Per-tool overrides", () => {
    it("overrides mode-based decision", () => {
      const gate = new ApprovalGate(
        makePolicy({
          mode: ApprovalMode.SUGGEST,
          toolOverrides: { write_file: "allow" },
        }),
      );
      const decision = gate.decide(
        makeRequest({ toolName: "write_file", toolCategory: "mutating" }),
      );
      expect(decision).toBe("allow"); // overridden from "ask"
    });

    it("can deny tools that would otherwise be allowed", () => {
      const gate = new ApprovalGate(
        makePolicy({
          mode: ApprovalMode.FULL_AUTO,
          toolOverrides: { run_command: "deny" },
        }),
      );
      const decision = gate.decide(
        makeRequest({ toolName: "run_command", toolCategory: "workflow" }),
      );
      expect(decision).toBe("deny"); // overridden from "allow"
    });
  });

  describe("Per-path rules", () => {
    it("matches exact path", () => {
      const gate = new ApprovalGate(
        makePolicy({
          pathRules: [{ pattern: "/etc/config.toml", action: "deny" }],
        }),
      );
      const decision = gate.decide(
        makeRequest({ filePath: "/etc/config.toml", toolCategory: "mutating" }),
      );
      expect(decision).toBe("deny");
    });

    it("matches glob pattern with *", () => {
      const gate = new ApprovalGate(
        makePolicy({
          mode: ApprovalMode.SUGGEST,
          pathRules: [{ pattern: "/src/*.ts", action: "allow" }],
        }),
      );
      const decision = gate.decide(
        makeRequest({
          filePath: "/src/index.ts",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("allow");
    });

    it("matches glob pattern with **", () => {
      const gate = new ApprovalGate(
        makePolicy({
          mode: ApprovalMode.SUGGEST,
          pathRules: [{ pattern: "/tests/**", action: "allow" }],
        }),
      );
      const decision = gate.decide(
        makeRequest({
          filePath: "/tests/unit/config.test.ts",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("allow");
    });

    it("tool overrides take precedence over path rules", () => {
      const gate = new ApprovalGate(
        makePolicy({
          toolOverrides: { write_file: "deny" },
          pathRules: [{ pattern: "/src/**", action: "allow" }],
        }),
      );
      const decision = gate.decide(
        makeRequest({
          toolName: "write_file",
          filePath: "/src/index.ts",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("deny"); // tool override wins
    });
  });

  describe("check() with event bus", () => {
    it("auto-approves without user input for allow decisions", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.FULL_AUTO }),
        bus,
      );
      const result = await gate.check(makeRequest());
      expect(result.approved).toBe(true);
      expect(result.requiredUserInput).toBe(false);
      expect(result.reason).toBe("auto-approved");
    });

    it("auto-denies without user input for deny decisions", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(makePolicy(), bus);
      const result = await gate.check(
        makeRequest({ toolCategory: "external", toolName: "web_search" }),
      );
      expect(result.approved).toBe(false);
      expect(result.requiredUserInput).toBe(false);
      expect(result.reason).toBe("auto-denied");
    });

    it("emits approval:request and waits for response", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(makePolicy(), bus);

      // Simulate user approving after a tick
      bus.on("approval:request", () => {
        setTimeout(() => {
          bus.emit("approval:response", { id: "test", approved: true });
        }, 1);
      });

      const result = await gate.check(makeRequest({ toolCategory: "mutating" }));
      expect(result.approved).toBe(true);
      expect(result.requiredUserInput).toBe(true);
      expect(result.reason).toBe("user-approved");
    });

    it("handles user denial", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(makePolicy(), bus);

      bus.on("approval:request", () => {
        setTimeout(() => {
          bus.emit("approval:response", { id: "test", approved: false });
        }, 1);
      });

      const result = await gate.check(makeRequest({ toolCategory: "mutating" }));
      expect(result.approved).toBe(false);
      expect(result.requiredUserInput).toBe(true);
      expect(result.reason).toBe("user-denied");
    });

    it("throws when no event bus available for ask decision", async () => {
      const gate = new ApprovalGate(makePolicy()); // no bus
      await expect(
        gate.check(makeRequest({ toolCategory: "mutating" })),
      ).rejects.toThrow("No event bus available");
    });
  });

  describe("audit logging", () => {
    it("emits audit event when auditLog is enabled", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.FULL_AUTO, auditLog: true }),
        bus,
      );

      const events: Array<{ name: string; params: unknown }> = [];
      bus.on("tool:before", (e) => events.push(e));

      await gate.check(makeRequest());
      expect(events).toHaveLength(1);
      expect(events[0]!.name).toBe("audit:write_file");
    });

    it("does not emit audit event when auditLog is disabled", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.FULL_AUTO, auditLog: false }),
        bus,
      );

      const events: Array<{ name: string }> = [];
      bus.on("tool:before", (e) => events.push(e));

      await gate.check(makeRequest());
      expect(events).toHaveLength(0);
    });
  });
});
