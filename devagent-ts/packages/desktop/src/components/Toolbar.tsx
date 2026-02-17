import type { AppMode, AppView } from "../types";

interface ToolbarProps {
  readonly mode: AppMode;
  readonly onModeChange: (mode: AppMode) => void;
  readonly view: AppView;
  readonly onViewChange: (view: AppView) => void;
  readonly isStreaming: boolean;
}

export function Toolbar({
  mode,
  onModeChange,
  view,
  onViewChange,
  isStreaming,
}: ToolbarProps): React.JSX.Element {
  return (
    <header className="toolbar">
      <div className="toolbar-left">
        <h1 className="toolbar-title">DevAgent</h1>
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
      </div>

      <div className="toolbar-center">
        {isStreaming && <span className="toolbar-status">Processing...</span>}
      </div>

      <div className="toolbar-right">
        <button
          className={`view-btn ${view === "chat" ? "view-btn-active" : ""}`}
          onClick={() => onViewChange("chat")}
        >
          Chat
        </button>
        <button
          className={`view-btn ${view === "diff" ? "view-btn-active" : ""}`}
          onClick={() => onViewChange("diff")}
        >
          Diffs
        </button>
        <button
          className={`view-btn ${view === "settings" ? "view-btn-active" : ""}`}
          onClick={() => onViewChange("settings")}
        >
          Settings
        </button>
      </div>
    </header>
  );
}
