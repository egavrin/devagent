/**
 * Status metrics tracker — tracks cost, tokens, iteration for display.
 * No longer renders its own line — instead provides formatted strings
 * that the Spinner and tool headers embed inline.
 *
 * Previous version used ANSI cursor save/restore + move-to-bottom,
 * which caused blank lines and fought with other output. This version
 * is pure data — no terminal writes.
 */

// ─── Types ──────────────────────────────────────────────────

interface StatusLineState {
  model: string;
  cost: number;
  inputTokens: number;
  maxContextTokens: number;
  iteration: number;
  maxIterations: number;
  approvalMode: string;
}

// ─── StatusLine ─────────────────────────────────────────────

/**
 * Tracks session metrics and formats them for inline display.
 * Call update() to refresh state, formatInline() to get a string
 * for embedding in spinner messages or tool headers.
 */
export class StatusLine {
  private state: StatusLineState;

  constructor(model: string, approvalMode: string) {
    this.state = {
      model,
      cost: 0,
      inputTokens: 0,
      maxContextTokens: 0,
      iteration: 0,
      maxIterations: 0,
      approvalMode,
    };
  }

  /** Update state. */
  update(partial: Partial<StatusLineState>): void {
    Object.assign(this.state, partial);
  }

  /**
   * Format metrics as a compact inline string for spinner/header embedding.
   * Returns: `$0.05 │ 45k/128k (35%)`
   */
  formatInline(): string {
    const parts: string[] = [];

    if (this.state.cost > 0) {
      parts.push(`$${this.state.cost.toFixed(4)}`);
    }

    if (this.state.maxContextTokens > 0) {
      const tokensK = formatTokens(this.state.inputTokens);
      const maxK = formatTokens(this.state.maxContextTokens);
      const pct = Math.round((this.state.inputTokens / this.state.maxContextTokens) * 100);
      parts.push(`${tokensK}/${maxK} (${pct}%)`);
    }

    return parts.join(" │ ");
  }

  /**
   * Format a short suffix for the spinner message.
   * Returns: ` │ $0.05 │ 35%` or empty string if no data.
   */
  formatSpinnerSuffix(): string {
    const inline = this.formatInline();
    return inline ? ` │ ${inline}` : "";
  }

  /** No-op methods for backward compatibility (removed cursor rendering). */
  clear(): void { /* no-op */ }
  suspend(): void { /* no-op */ }
  resume(): void { /* no-op */ }

  get cost(): number { return this.state.cost; }
  get iteration(): number { return this.state.iteration; }
  get maxIterations(): number { return this.state.maxIterations; }
}

// ─── Helpers ────────────────────────────────────────────────

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}
