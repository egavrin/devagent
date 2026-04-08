import { describe, expect, it } from "vitest";

import { TranscriptComposer } from "./transcript-composer.js";
import {
  makeFinalOutputPart,
  makeInfoPart,
  makeTurnSummaryPart,
} from "./transcript-presenter.js";

describe("TranscriptComposer", () => {
  it("keeps standalone info rows outside turns", () => {
    const composer = new TranscriptComposer();

    composer.appendStandalone("info-1", makeInfoPart("info", ["Ready."]));

    const nodes = composer.getNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0]).toMatchObject({
      id: "info-1",
      kind: "part",
      part: { kind: "info" },
    });
  });

  it("composes one completed turn from user activity and summary", () => {
    const composer = new TranscriptComposer();

    composer.startTurn("turn-1", "Fix the test", 1_000);
    composer.appendPart("tool-1", {
      kind: "tool",
      event: {
        id: "call-1",
        name: "write_file",
        summary: "src/app.ts",
        iteration: 1,
        maxIterations: 8,
        status: "success",
        durationMs: 20,
      },
    });
    composer.appendPart("edit-1", {
      kind: "file-edit",
      data: {
        toolId: "call-1",
        summary: "Added 1 line",
        fileEdit: {
          path: "src/app.ts",
          kind: "update",
          additions: 1,
          deletions: 0,
          unifiedDiff: "@@ -1,1 +1,2 @@\n line\n+tail",
          truncated: false,
        },
      },
    });
    composer.appendPart("final-1", makeFinalOutputPart("Done."));
    composer.completeTurn(
      "summary-1",
      makeTurnSummaryPart({
        iterations: 2,
        toolCalls: 1,
        cost: 0.01,
        elapsedMs: 500,
      }),
      { status: "completed", finishedAt: 1_500 },
    );

    const nodes = composer.getNodes();
    expect(nodes).toHaveLength(1);
    expect(nodes[0]?.kind).toBe("turn");
    if (nodes[0]?.kind !== "turn") {
      throw new Error("expected a completed turn");
    }
    expect(nodes[0].turn.userText).toBe("Fix the test");
    expect(nodes[0].turn.entries.map((entry) => entry.part.kind)).toEqual(["tool", "file-edit", "final-output"]);
    expect(nodes[0].turn.metrics.toolCalls).toBe(1);
    expect(nodes[0].turn.metrics.filesChanged).toBe(1);
    expect(nodes[0].turn.metrics.validationFailed).toBe(false);
    expect(nodes[0].turn.summary?.iterations).toBe(2);
  });

  it("attaches progress, approval, and validation rows to the active turn", () => {
    const composer = new TranscriptComposer();

    composer.startTurn("turn-1", "Update config", 2_000);
    composer.appendPart("progress-1", {
      kind: "progress",
      data: { title: "compacting context", detail: "96k / 128k tokens" },
    });
    composer.appendPart("approval-1", {
      kind: "approval",
      data: {
        id: "approval-1",
        action: "edit",
        toolName: "write_file",
        details: "Write src/config.ts",
        status: "pending",
      },
    });
    composer.appendPart("validation-1", {
      kind: "validation-result",
      data: {
        toolId: "call-1",
        passed: false,
        summary: "Validation failed",
        diagnosticCount: 1,
      },
    });
    composer.completeTurn(
      "summary-1",
      makeTurnSummaryPart({
        iterations: 1,
        toolCalls: 0,
        cost: 0,
        elapsedMs: 250,
      }),
      { finishedAt: 2_250 },
    );

    const turnNode = composer.getNodes()[0];
    expect(turnNode?.kind).toBe("turn");
    if (turnNode?.kind !== "turn") {
      throw new Error("expected turn node");
    }
    expect(turnNode.turn.entries.map((entry) => entry.part.kind)).toEqual([
      "progress",
      "approval",
      "validation-result",
    ]);
    expect(turnNode.turn.metrics.validationFailed).toBe(true);
  });

  it("keeps turns separate across consecutive user submissions", () => {
    const composer = new TranscriptComposer();

    composer.startTurn("turn-1", "First request", 1_000);
    composer.appendPart("tool-1", {
      kind: "tool",
      event: {
        id: "call-1",
        name: "read_file",
        summary: "src/a.ts",
        iteration: 1,
        maxIterations: 8,
        status: "success",
      },
    });
    composer.startTurn("turn-2", "Second request", 2_000);
    composer.appendPart("tool-2", {
      kind: "tool",
      event: {
        id: "call-2",
        name: "write_file",
        summary: "src/b.ts",
        iteration: 2,
        maxIterations: 8,
        status: "success",
      },
    });
    composer.completeTurn(
      "summary-2",
      makeTurnSummaryPart({
        iterations: 1,
        toolCalls: 1,
        cost: 0,
        elapsedMs: 100,
      }),
      { finishedAt: 2_100 },
    );

    const nodes = composer.getNodes();
    expect(nodes).toHaveLength(2);
    expect(nodes[0]?.kind).toBe("turn");
    expect(nodes[1]?.kind).toBe("turn");
    if (nodes[0]?.kind !== "turn" || nodes[1]?.kind !== "turn") {
      throw new Error("expected completed turns");
    }
    expect(nodes[0].turn.userText).toBe("First request");
    expect(nodes[0].turn.entries).toHaveLength(1);
    expect(nodes[1].turn.userText).toBe("Second request");
    expect(nodes[1].turn.entries).toHaveLength(1);
  });
});
