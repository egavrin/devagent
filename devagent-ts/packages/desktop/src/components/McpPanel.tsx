/**
 * McpPanel — card-per-server layout for MCP servers.
 *
 * Shows status dot (green/yellow/red), tool count, expandable tool list,
 * restart button per server.
 */

import { useState, useCallback } from "react";
import type { McpServerInfo, McpToolInfo } from "../types";

interface McpPanelProps {
  readonly servers: ReadonlyArray<McpServerInfo>;
  readonly loading: boolean;
  readonly onRefresh: () => void;
  readonly onRestart: (name: string) => void;
}

function statusDotClass(status: McpServerInfo["status"]): string {
  switch (status) {
    case "running":
      return "mcp-status-running";
    case "stopped":
      return "mcp-status-stopped";
    case "error":
      return "mcp-status-error";
  }
}

function McpToolRow({ tool }: { readonly tool: McpToolInfo }): React.JSX.Element {
  return (
    <div className="mcp-tool-row">
      <span className="mcp-tool-name">{tool.name}</span>
      {tool.description && (
        <span className="mcp-tool-desc">{tool.description}</span>
      )}
    </div>
  );
}

function McpServerCard({
  server,
  onRestart,
}: {
  readonly server: McpServerInfo;
  readonly onRestart: (name: string) => void;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(false);

  const handleRestart = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onRestart(server.name);
    },
    [onRestart, server.name],
  );

  return (
    <div className="mcp-server-card">
      <div className="mcp-server-header" onClick={() => setExpanded(!expanded)}>
        <div className="mcp-server-info">
          <span className={`mcp-status-dot ${statusDotClass(server.status)}`} />
          <span className="mcp-server-name">{server.name}</span>
          <span className="mcp-tool-count">
            {server.toolCount} tool{server.toolCount !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="mcp-server-actions">
          <button
            className="mcp-restart-btn"
            onClick={handleRestart}
            title="Restart server"
          >
            ↻
          </button>
          <span className="mcp-expand-toggle">{expanded ? "▼" : "▶"}</span>
        </div>
      </div>
      {server.error && (
        <div className="mcp-server-error">{server.error}</div>
      )}
      {expanded && server.tools.length > 0 && (
        <div className="mcp-server-tools">
          <div className="mcp-tools-header">Available Tools</div>
          {server.tools.map((tool) => (
            <McpToolRow key={tool.name} tool={tool} />
          ))}
        </div>
      )}
      {expanded && server.tools.length === 0 && (
        <div className="mcp-no-tools">No tools registered.</div>
      )}
    </div>
  );
}

export function McpPanel({
  servers,
  loading,
  onRefresh,
  onRestart,
}: McpPanelProps): React.JSX.Element {
  const runningCount = servers.filter((s) => s.status === "running").length;

  if (servers.length === 0 && !loading) {
    return (
      <div className="mcp-panel">
        <div className="mcp-header">
          <h2>MCP Servers</h2>
          <button className="panel-refresh-btn" onClick={onRefresh}>Refresh</button>
        </div>
        <div className="panel-empty">
          <span className="panel-empty-icon">🔌</span>
          <h3>No MCP Servers Configured</h3>
          <p>
            Add servers to <code>.devagent/mcp.json</code> in your project.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mcp-panel">
      <div className="mcp-header">
        <h2>
          MCP Servers ({runningCount}/{servers.length} running)
        </h2>
        <button className="panel-refresh-btn" onClick={onRefresh} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>
      <div className="mcp-server-list">
        {servers.map((server) => (
          <McpServerCard
            key={server.name}
            server={server}
            onRestart={onRestart}
          />
        ))}
      </div>
    </div>
  );
}
