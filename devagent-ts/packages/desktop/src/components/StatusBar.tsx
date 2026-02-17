interface StatusBarProps {
  readonly connected: boolean;
  readonly ready: boolean;
  readonly streaming: boolean;
  readonly error: string | null;
  readonly toolCount: number;
  readonly workingDir: string;
}

export function StatusBar({
  connected,
  ready,
  streaming,
  error,
  toolCount,
  workingDir,
}: StatusBarProps): React.JSX.Element {
  const statusDot = connected && ready ? "status-dot-connected" : "status-dot-disconnected";
  const statusText = error
    ? error
    : streaming
      ? "Processing..."
      : connected && ready
        ? "Connected"
        : connected
          ? "Starting engine..."
          : "Disconnected";

  return (
    <footer className="status-bar">
      <div className="status-bar-left">
        <span className={`status-dot ${statusDot}`} />
        <span className="status-text">{statusText}</span>
        {workingDir && (
          <>
            <span className="status-separator">|</span>
            <span className="status-dir" title={workingDir}>{workingDir}</span>
          </>
        )}
      </div>
      <div className="status-bar-right">
        {toolCount > 0 && (
          <span className="status-tools">{toolCount} tool call{toolCount !== 1 ? "s" : ""}</span>
        )}
      </div>
    </footer>
  );
}
