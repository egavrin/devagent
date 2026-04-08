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
  let toolCalls = 0;
  let filesChanged = 0;
  let validationFailed = false;

  for (const entry of entries) {
    switch (entry.part.kind) {
      case "tool":
        if (entry.part.event.status === "success" || entry.part.event.status === "error") {
          toolCalls++;
        }
        break;
      case "tool-group":
        if (entry.part.event.status === "success" || entry.part.event.status === "error") {
          toolCalls += entry.part.event.count;
        }
        break;
      case "file-edit":
        filesChanged++;
        break;
      case "file-edit-overflow":
        filesChanged += entry.part.data.hiddenCount;
        break;
      case "validation-result":
        if (!entry.part.data.passed) validationFailed = true;
        break;
      default:
        break;
    }
  }

  return {
    toolCalls,
    filesChanged,
    validationFailed,
    iterations: summary?.iterations ?? 0,
    cost: summary?.cost ?? 0,
    elapsedMs: summary?.elapsedMs ?? Math.max(0, finishedAt - startedAt),
  };
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
  switch (part.kind) {
    case "tool":
    case "tool-group":
    case "file-edit":
    case "file-edit-overflow":
    case "command-result":
    case "validation-result":
    case "diagnostic-list":
    case "status":
    case "progress":
    case "approval":
    case "reasoning":
    case "plan":
    case "error":
    case "final-output":
    case "info":
      return true;
    case "turn-summary":
    case "user":
      return false;
    default:
      return false;
  }
}

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
