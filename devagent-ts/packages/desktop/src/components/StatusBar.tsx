interface StatusBarProps {
  readonly connected: boolean;
  readonly ready: boolean;
  readonly streaming: boolean;
  readonly error: string | null;
  readonly toolCount: number;
}

export function StatusBar({
  connected,
  ready,
  streaming,
  error,
  toolCount,
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
      </div>
      <div className="status-bar-right">
        {toolCount > 0 && (
          <span className="status-tools">{toolCount} tool call{toolCount !== 1 ? "s" : ""}</span>
        )}
      </div>
    </footer>
  );
}
