import { describe, it, expect } from "vitest";
import { ApprovalGate } from "./approval.js";
import type { ApprovalRequest } from "./approval.js";
import type { ApprovalPolicy } from "./types.js";
import { ApprovalMode, SafetyMode } from "./types.js";
import { EventBus } from "./events.js";

function makePolicy(overrides?: Partial<ApprovalPolicy>): ApprovalPolicy {
  return {
    mode: ApprovalMode.SUGGEST,
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

function makeSafetyPolicy(
  overrides?: Partial<ApprovalPolicy>,
): ApprovalPolicy {
  return {
    mode: SafetyMode.DEFAULT,
    approvalPolicy: "on-request",
    sandboxMode: "workspace-write",
    networkAccess: "off",
    auditLog: false,
    toolOverrides: {},
    pathRules: [],
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

    it("allows state tools (internal agent state)", () => {
      const gate = new ApprovalGate(makePolicy());
      const decision = gate.decide(
        makeRequest({ toolCategory: "state", toolName: "update_plan" }),
      );
      expect(decision).toBe("allow");
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

    it("allows state tools (internal agent state)", () => {
      const gate = new ApprovalGate(
        makePolicy({ mode: ApprovalMode.AUTO_EDIT }),
      );
      const decision = gate.decide(
        makeRequest({ toolCategory: "state", toolName: "update_plan" }),
      );
      expect(decision).toBe("allow");
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
      expect(
        gate.decide(makeRequest({ toolCategory: "state" })),
      ).toBe("allow");
    });

    it("keeps legacy full-auto semantics even when merged safety fields are present", () => {
      const gate = new ApprovalGate(
        makePolicy({
          mode: ApprovalMode.FULL_AUTO,
          approvalPolicy: "on-request",
          sandboxMode: "workspace-write",
          networkAccess: "off",
        }),
      );

      expect(
        gate.decide(makeRequest({ toolCategory: "workflow", toolName: "delegate", filePath: null })),
      ).toBe("allow");
      expect(
        gate.decide(makeRequest({ toolCategory: "external", toolName: "web_search", filePath: null })),
      ).toBe("allow");
    });
  });

  describe("Dynamic mode changes", () => {
    it("updates the active mode and decision logic", () => {
      const gate = new ApprovalGate(makePolicy());

      expect(gate.getMode()).toBe(ApprovalMode.SUGGEST);
      expect(gate.decide(makeRequest({ toolCategory: "mutating" }))).toBe("ask");

      gate.setMode(ApprovalMode.AUTO_EDIT);
      expect(gate.getMode()).toBe(ApprovalMode.AUTO_EDIT);
      expect(gate.decide(makeRequest({ toolCategory: "mutating" }))).toBe("allow");

      gate.setMode(ApprovalMode.FULL_AUTO);
      expect(gate.getMode()).toBe(ApprovalMode.FULL_AUTO);
      expect(gate.decide(makeRequest({ toolCategory: "external", toolName: "web_search" }))).toBe("allow");
    });
  });

  describe("Safety modes", () => {
    it("advanced strict policy asks for mutating, workflow, and external actions", () => {
      const gate = new ApprovalGate(
        makeSafetyPolicy({
          mode: SafetyMode.DEFAULT,
          approvalPolicy: "strict",
          sandboxMode: "read-only",
          networkAccess: "off",
        }),
      );

      expect(
        gate.decide(
          makeRequest({ toolCategory: "mutating", repoRoot: "/repo", filePath: "/repo/src/app.ts" }),
        ),
      ).toBe("ask");
      expect(
        gate.decide(
          makeRequest({
            toolName: "run_command",
            toolCategory: "workflow",
            repoRoot: "/repo",
            filePath: null,
            arguments: { command: "bun test", cwd: "." },
          }),
        ),
      ).toBe("ask");
      expect(
        gate.decide(
          makeRequest({ toolCategory: "external", toolName: "web_search", filePath: null }),
        ),
      ).toBe("ask");
    });

    it("default mode allows mutating tools inside the repo", () => {
      const gate = new ApprovalGate(makeSafetyPolicy());
      const decision = gate.decide(
        makeRequest({
          repoRoot: "/repo",
          filePath: "/repo/src/app.ts",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("allow");
    });

    it("default mode allows safe repo commands", () => {
      const gate = new ApprovalGate(makeSafetyPolicy());
      const decision = gate.decide(
        makeRequest({
          toolName: "run_command",
          toolCategory: "workflow",
          filePath: null,
          repoRoot: "/repo",
          arguments: { command: "bun test", cwd: "." },
        }),
      );
      expect(decision).toBe("allow");
    });

    it("default mode asks for dependency installs", () => {
      const gate = new ApprovalGate(makeSafetyPolicy());
      const decision = gate.decide(
        makeRequest({
          toolName: "run_command",
          toolCategory: "workflow",
          filePath: null,
          repoRoot: "/repo",
          arguments: { command: "npm install zod", cwd: "." },
        }),
      );
      expect(decision).toBe("ask");
    });

    it("default mode asks for sensitive paths", () => {
      const gate = new ApprovalGate(makeSafetyPolicy());
      const decision = gate.decide(
        makeRequest({
          repoRoot: "/repo",
          filePath: "/Users/test/.ssh/config",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("ask");
    });

    it("default mode asks for writes outside the repo", () => {
      const gate = new ApprovalGate(makeSafetyPolicy());
      const decision = gate.decide(
        makeRequest({
          repoRoot: "/repo",
          filePath: "../outside.txt",
          toolCategory: "mutating",
        }),
      );
      expect(decision).toBe("ask");
    });

    it("autopilot mode allows external actions", () => {
      const gate = new ApprovalGate(
        makeSafetyPolicy({
          mode: SafetyMode.AUTOPILOT,
          approvalPolicy: "never",
          sandboxMode: "danger-full-access",
          networkAccess: "on",
        }),
      );
      const decision = gate.decide(
        makeRequest({ toolCategory: "external", toolName: "web_search", filePath: null }),
      );
      expect(decision).toBe("allow");
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

    it("remembers session approvals for matching requests", async () => {
      const bus = new EventBus();
      const gate = new ApprovalGate(makeSafetyPolicy(), bus);
      let requestCount = 0;

      bus.on("approval:request", () => {
        requestCount += 1;
        setTimeout(() => {
          bus.emit("approval:response", {
            id: "test",
            approved: true,
            feedback: "session",
          });
        }, 1);
      });

      const request = makeRequest({
        toolName: "run_command",
        toolCategory: "workflow",
        filePath: null,
        repoRoot: "/repo",
        arguments: { command: "npm install zod", cwd: "." },
      });

      const first = await gate.check(request);
      const second = await gate.check(request);

      expect(first).toMatchObject({
        approved: true,
        requiredUserInput: true,
        reason: "user-approved",
      });
      expect(second).toMatchObject({
        approved: true,
        requiredUserInput: false,
        reason: "auto-approved",
      });
      expect(requestCount).toBe(1);
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
