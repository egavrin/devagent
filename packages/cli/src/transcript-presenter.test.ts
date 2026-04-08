import { describe, expect, it } from "vitest";
import {
  presentApprovalRequestEvent,
  presentApprovalResponseEvent,
  presentContextCompactingEvent,
  presentSummaryToolMessage,
  presentToolAfterEvent,
} from "./transcript-presenter.js";

describe("transcript-presenter", () => {
  it("emits tool plus file-edit parts for tool:after file edits", () => {
    const parts = presentToolAfterEvent({
      name: "write_file",
      callId: "call-1",
      durationMs: 12,
      fileEditHiddenCount: 1,
      fileEdits: [{
        path: "src/new.ts",
        kind: "create",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
        truncated: false,
      }],
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
      },
    }, 2, 10);

    expect(parts[0]?.kind).toBe("tool");
    expect(parts[1]?.kind).toBe("file-edit");
    expect(parts[2]?.kind).toBe("file-edit-overflow");
  });

  it("emits command-result parts for run_command metadata", () => {
    const parts = presentToolAfterEvent({
      name: "run_command",
      callId: "call-cmd-1",
      durationMs: 45,
      result: {
        success: false,
        output: "Exit code: 1",
        error: "Command exited with code 1",
        artifacts: [],
        metadata: {
          commandResult: {
            command: "npm test",
            cwd: ".",
            exitCode: 1,
            timedOut: false,
            warningOnly: false,
            stdoutPreview: "stdout line",
            stderrPreview: "stderr line",
            stdoutTruncated: false,
            stderrTruncated: false,
          },
        },
      },
    }, 2, 10);

    expect(parts[0]?.kind).toBe("tool");
    expect(parts[1]?.kind).toBe("command-result");
    if (parts[1]?.kind === "command-result") {
      expect(parts[1].data.command).toBe("npm test");
      expect(parts[1].data.statusLine).toContain("Exited with code 1");
    }
  });

  it("emits validation and diagnostic parts from validation metadata", () => {
    const parts = presentToolAfterEvent({
      name: "write_file",
      callId: "call-validate-1",
      durationMs: 12,
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          validationResult: {
            passed: false,
            diagnosticErrors: ["src/new.ts: Unexpected token"],
            testPassed: false,
            testOutputPreview: "1 failed",
            testSummary: {
              framework: "vitest",
              passed: 3,
              failed: 1,
              failureMessages: ["fails"],
            },
          },
        },
      },
    }, 2, 10);

    expect(parts[1]?.kind).toBe("validation-result");
    expect(parts[2]?.kind).toBe("diagnostic-list");
  });

  it("emits progress parts for context compaction", () => {
    const part = presentContextCompactingEvent({
      estimatedTokens: 96000,
      maxTokens: 128000,
    });
    expect(part.kind).toBe("progress");
    if (part.kind === "progress") {
      expect(part.data.detail).toContain("96k / 128k");
    }
  });

  it("emits distinct approval request and response parts", () => {
    const request = presentApprovalRequestEvent({
      id: "approval-1",
      action: "edit",
      toolName: "write_file",
      details: "Write src/new.ts",
    });
    const response = presentApprovalResponseEvent({
      id: "approval-1",
      approved: true,
    });

    expect(request.kind).toBe("approval");
    expect(response.kind).toBe("status");
    if (request.kind === "approval") {
      expect(request.data.action).toBe("edit");
    }
  });

  it("maps delegate summary tool messages to typed status parts", () => {
    const part = presentSummaryToolMessage({
      role: "tool",
      content: "Subagent root-sub-1 docs/spec completed (4.5s)",
      toolCallId: "call-1",
      toolName: "delegate",
      summaryOnly: true,
    });

    expect(part.kind).toBe("status");
    if (part.kind === "status") {
      expect(part.data.title).toBe("delegate");
      expect(part.data.lines[0]).toContain("completed");
    }
  });
});
