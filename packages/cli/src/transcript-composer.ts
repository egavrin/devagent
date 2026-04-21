import type {
  PresentedTurnSummary,
  TranscriptPart,
} from "./transcript-presenter.js";

export type PresentedTurnStatus = "running" | "completed" | "error" | "budget_exceeded";

export interface TurnEntry {
  readonly id: string;
  readonly part: TranscriptPart;
}

export interface TurnMetrics {
  readonly toolCalls: number;
  readonly filesChanged: number;
  readonly validationFailed: boolean;
  readonly iterations: number;
  readonly cost: number;
  readonly elapsedMs: number;
}

export interface PresentedTurn {
  readonly id: string;
  readonly userText: string;
  readonly startedAt: number;
  readonly finishedAt?: number;
  readonly status: PresentedTurnStatus;
  readonly entries: ReadonlyArray<TurnEntry>;
  readonly summary?: PresentedTurnSummary;
  readonly metrics: TurnMetrics;
}

export type TranscriptNode =
  | { readonly id: string; readonly kind: "part"; readonly part: TranscriptPart }
  | { readonly id: string; readonly kind: "turn"; readonly turn: PresentedTurn };

interface ActiveTurn {
  readonly id: string;
  readonly userText: string;
  readonly startedAt: number;
  readonly entries: ReadonlyArray<TurnEntry>;
}
function createMetrics(
  entries: ReadonlyArray<TurnEntry>,
  summary: PresentedTurnSummary | undefined,
  startedAt: number,
  finishedAt: number,
): TurnMetrics {
  const entryMetrics = entries.map((entry) => getEntryMetrics(entry.part));

  return {
    toolCalls: sumMetric(entryMetrics, "toolCalls"),
    filesChanged: sumMetric(entryMetrics, "filesChanged"),
    validationFailed: entryMetrics.some((metrics) => metrics.validationFailed),
    iterations: summary?.iterations ?? 0,
    cost: summary?.cost ?? 0,
    elapsedMs: summary?.elapsedMs ?? Math.max(0, finishedAt - startedAt),
  };
}

interface EntryMetrics {
  readonly toolCalls: number;
  readonly filesChanged: number;
  readonly validationFailed: boolean;
}

function getEntryMetrics(part: TranscriptPart): EntryMetrics {
  return {
    toolCalls: getPartToolCallCount(part),
    filesChanged: getPartFileChangeCount(part),
    validationFailed: part.kind === "validation-result" && !part.data.passed,
  };
}

function getPartToolCallCount(part: TranscriptPart): number {
  if (part.kind === "tool") {
    return isFinishedToolStatus(part.event.status) ? 1 : 0;
  }
  if (part.kind === "tool-group") {
    return isFinishedToolStatus(part.event.status) ? part.event.count : 0;
  }
  return 0;
}

function isFinishedToolStatus(status: string): boolean {
  return status === "success" || status === "error";
}

function getPartFileChangeCount(part: TranscriptPart): number {
  if (part.kind === "file-edit") return 1;
  if (part.kind === "file-edit-overflow") return part.data.hiddenCount;
  return 0;
}

function sumMetric(metrics: ReadonlyArray<EntryMetrics>, field: "toolCalls" | "filesChanged"): number {
  return metrics.reduce((total, entry) => total + entry[field], 0);
}

function inferCompletedStatus(
  entries: ReadonlyArray<TurnEntry>,
): PresentedTurnStatus {
  for (const entry of entries) {
    if (entry.part.kind === "error") return "error";
  }
  return "completed";
}
function partBelongsToTurn(part: TranscriptPart): boolean {
  return TURN_PART_KINDS.has(part.kind);
}

const TURN_PART_KINDS = new Set<TranscriptPart["kind"]>([
  "tool",
  "tool-group",
  "file-edit",
  "file-edit-overflow",
  "command-result",
  "validation-result",
  "diagnostic-list",
  "status",
  "progress",
  "approval",
  "reasoning",
  "plan",
  "error",
  "final-output",
  "info",
]);

export class TranscriptComposer {
  private readonly nodes: TranscriptNode[] = [];
  private activeTurn: ActiveTurn | null = null;

  startTurn(id: string, userText: string, startedAt: number): void {
    if (this.activeTurn) {
      this.finishActiveTurn(undefined, { status: inferCompletedStatus(this.activeTurn.entries), finishedAt: startedAt });
    }
    this.activeTurn = {
      id,
      userText,
      startedAt,
      entries: [],
    };
  }

  appendPart(id: string, part: TranscriptPart): void {
    if (part.kind === "user") {
      this.startTurn(id, part.data.text, Date.now());
      return;
    }
    if (part.kind === "turn-summary") {
      this.completeTurn(id, part);
      return;
    }

    if (this.activeTurn && partBelongsToTurn(part)) {
      this.activeTurn = {
        ...this.activeTurn,
        entries: [...this.activeTurn.entries, { id, part }],
      };
      return;
    }

    this.appendStandalone(id, part);
  }

  appendStandalone(id: string, part: TranscriptPart): void {
    this.nodes.push({ id, kind: "part", part });
  }

  completeTurn(
    id: string,
    summaryPart: Extract<TranscriptPart, { readonly kind: "turn-summary" }>,
    options?: {
      readonly status?: PresentedTurnStatus;
      readonly finishedAt?: number;
    },
  ): void {
    if (!this.activeTurn) {
      this.appendStandalone(id, summaryPart);
      return;
    }
    this.finishActiveTurn(summaryPart.data, options);
  }

  getActiveTurn(): PresentedTurn | null {
    if (!this.activeTurn) return null;
    const finishedAt = Date.now();
    return {
      id: this.activeTurn.id,
      userText: this.activeTurn.userText,
      startedAt: this.activeTurn.startedAt,
      status: "running",
      entries: this.activeTurn.entries,
      metrics: createMetrics(this.activeTurn.entries, undefined, this.activeTurn.startedAt, finishedAt),
    };
  }

  getNodes(): ReadonlyArray<TranscriptNode> {
    const active = this.getActiveTurn();
    return active
      ? [...this.nodes, { id: active.id, kind: "turn", turn: active }]
      : [...this.nodes];
  }

  private finishActiveTurn(
    summary: PresentedTurnSummary | undefined,
    options?: {
      readonly status?: PresentedTurnStatus;
      readonly finishedAt?: number;
    },
  ): void {
    if (!this.activeTurn) return;
    const finishedAt = options?.finishedAt ?? Date.now();
    const status = options?.status ?? inferCompletedStatus(this.activeTurn.entries);
    const completedTurn: PresentedTurn = {
      id: this.activeTurn.id,
      userText: this.activeTurn.userText,
      startedAt: this.activeTurn.startedAt,
      finishedAt,
      status,
      entries: this.activeTurn.entries,
      ...(summary ? { summary } : {}),
      metrics: createMetrics(this.activeTurn.entries, summary, this.activeTurn.startedAt, finishedAt),
    };
    this.nodes.push({ id: completedTurn.id, kind: "turn", turn: completedTurn });
    this.activeTurn = null;
  }
}
