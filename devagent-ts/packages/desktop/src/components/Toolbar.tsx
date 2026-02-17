import type { AppMode, CostState } from "../types";

interface ToolbarProps {
  readonly mode: AppMode;
  readonly onModeChange: (mode: AppMode) => void;
  readonly isStreaming: boolean;
  readonly costState: CostState;
  readonly workingDir: string;
}

function formatTokens(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatCost(cost: number): string {
  if (cost < 0.01) return cost > 0 ? "<$0.01" : "$0.00";
  return `$${cost.toFixed(2)}`;
}

function shortenDir(dir: string): string {
  const parts = dir.split("/").filter(Boolean);
  if (parts.length <= 2) return dir;
  return `.../${parts.slice(-2).join("/")}`;
}

export function Toolbar({
  mode,
  onModeChange,
  isStreaming,
  costState,
  workingDir,
}: ToolbarProps): React.JSX.Element {
  return (
    <header className="toolbar">
      <div className="toolbar-left">
        <h1 className="toolbar-title">DevAgent</h1>
        <div className="toolbar-dir" title={workingDir}>
          <span className="toolbar-dir-icon">📁</span>
          <span className="toolbar-dir-path">{shortenDir(workingDir)}</span>
        </div>
      </div>

      <div className="toolbar-center">
        <div className="toolbar-mode">
          <button
            className={`mode-btn ${mode === "plan" ? "mode-btn-active" : ""}`}
            onClick={() => onModeChange("plan")}
            disabled={isStreaming}
          >
            Plan
          </button>
          <button
            className={`mode-btn ${mode === "act" ? "mode-btn-active" : ""}`}
            onClick={() => onModeChange("act")}
            disabled={isStreaming}
          >
            Act
          </button>
        </div>
        {isStreaming && <span className="toolbar-status">Processing...</span>}
      </div>

      <div className="toolbar-right">
        <div className="toolbar-cost" title={`Input: ${costState.inputTokens} | Output: ${costState.outputTokens}`}>
          <span className="toolbar-cost-tokens">
            ↑{formatTokens(costState.inputTokens)} ↓{formatTokens(costState.outputTokens)}
          </span>
          <span className="toolbar-cost-divider">·</span>
          <span className="toolbar-cost-amount">{formatCost(costState.totalCost)}</span>
        </div>
      </div>
    </header>
  );
}
