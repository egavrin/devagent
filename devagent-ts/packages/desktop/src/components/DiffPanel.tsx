/**
 * DiffPanel — unified diff viewer with green/red line highlighting.
 *
 * Renders file diffs collected from tool executions (write_file, replace_in_file).
 * Each file shows a unified diff with per-file accept/reject controls.
 */

import { useState, useCallback } from "react";
import type { FileDiff, DiffHunk, DiffLine } from "../types";

interface DiffPanelProps {
  readonly diffs: ReadonlyArray<FileDiff>;
  readonly hasPending: boolean;
  readonly onAccept: (id: string) => void;
  readonly onReject: (id: string) => void;
  readonly onAcceptAll: () => void;
  readonly onClearResolved: () => void;
}

function fileName(filePath: string): string {
  const parts = filePath.split("/");
  return parts.length > 2
    ? `.../${parts.slice(-2).join("/")}`
    : filePath;
}

function statusBadge(status: FileDiff["status"]): string {
  switch (status) {
    case "pending":
      return "diff-badge-pending";
    case "accepted":
      return "diff-badge-accepted";
    case "rejected":
      return "diff-badge-rejected";
  }
}

function DiffLineRow({ line }: { readonly line: DiffLine }): React.JSX.Element {
  const lineClass =
    line.type === "add"
      ? "diff-line-add"
      : line.type === "remove"
        ? "diff-line-remove"
        : "diff-line-context";

  const prefix =
    line.type === "add" ? "+" : line.type === "remove" ? "-" : " ";

  return (
    <tr className={`diff-line ${lineClass}`}>
      <td className="diff-line-num diff-line-num-old">
        {line.oldLineNumber ?? ""}
      </td>
      <td className="diff-line-num diff-line-num-new">
        {line.newLineNumber ?? ""}
      </td>
      <td className="diff-line-prefix">{prefix}</td>
      <td className="diff-line-content">
        <pre>{line.content}</pre>
      </td>
    </tr>
  );
}

function DiffHunkView({ hunk }: { readonly hunk: DiffHunk }): React.JSX.Element {
  return (
    <div className="diff-hunk">
      <div className="diff-hunk-header">{hunk.header}</div>
      <table className="diff-table">
        <tbody>
          {hunk.lines.map((line, idx) => (
            <DiffLineRow key={idx} line={line} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiffFileCard({
  diff,
  onAccept,
  onReject,
}: {
  readonly diff: FileDiff;
  readonly onAccept: (id: string) => void;
  readonly onReject: (id: string) => void;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(true);

  const addCount = diff.hunks.reduce(
    (sum, h) => sum + h.lines.filter((l) => l.type === "add").length,
    0,
  );
  const removeCount = diff.hunks.reduce(
    (sum, h) => sum + h.lines.filter((l) => l.type === "remove").length,
    0,
  );

  return (
    <div className={`diff-file-card ${diff.status !== "pending" ? "diff-file-resolved" : ""}`}>
      <div className="diff-file-header" onClick={() => setExpanded(!expanded)}>
        <div className="diff-file-info">
          <span className="diff-file-toggle">{expanded ? "▼" : "▶"}</span>
          <span className="diff-file-name" title={diff.filePath}>
            {fileName(diff.filePath)}
          </span>
          <span className="diff-file-stats">
            {addCount > 0 && <span className="diff-stat-add">+{addCount}</span>}
            {removeCount > 0 && <span className="diff-stat-remove">-{removeCount}</span>}
          </span>
          <span className={`diff-badge ${statusBadge(diff.status)}`}>
            {diff.status}
          </span>
        </div>
        {diff.status === "pending" && (
          <div className="diff-file-actions" onClick={(e) => e.stopPropagation()}>
            <button
              className="diff-btn diff-btn-accept"
              onClick={() => onAccept(diff.id)}
              title="Accept changes"
            >
              Accept
            </button>
            <button
              className="diff-btn diff-btn-reject"
              onClick={() => onReject(diff.id)}
              title="Reject changes"
            >
              Reject
            </button>
          </div>
        )}
      </div>
      {expanded && (
        <div className="diff-file-body">
          {diff.hunks.map((hunk, idx) => (
            <DiffHunkView key={idx} hunk={hunk} />
          ))}
        </div>
      )}
    </div>
  );
}

export function DiffPanel({
  diffs,
  hasPending,
  onAccept,
  onReject,
  onAcceptAll,
  onClearResolved,
}: DiffPanelProps): React.JSX.Element {
  const [filter, setFilter] = useState<"all" | "pending" | "accepted" | "rejected">("all");

  const handleFilterChange = useCallback((f: typeof filter) => {
    setFilter(f);
  }, []);

  const filteredDiffs =
    filter === "all" ? diffs : diffs.filter((d) => d.status === filter);

  if (diffs.length === 0) {
    return (
      <div className="diff-panel">
        <div className="diff-empty">
          <span className="diff-empty-icon">📝</span>
          <h2>No Diffs Yet</h2>
          <p>File changes will appear here when the agent edits files.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="diff-panel">
      <div className="diff-toolbar">
        <div className="diff-filters">
          {(["all", "pending", "accepted", "rejected"] as const).map((f) => (
            <button
              key={f}
              className={`diff-filter-btn ${filter === f ? "diff-filter-active" : ""}`}
              onClick={() => handleFilterChange(f)}
            >
              {f === "all" ? `All (${diffs.length})` : `${f} (${diffs.filter((d) => d.status === f).length})`}
            </button>
          ))}
        </div>
        <div className="diff-bulk-actions">
          {hasPending && (
            <button className="diff-btn diff-btn-accept" onClick={onAcceptAll}>
              Accept All
            </button>
          )}
          {diffs.some((d) => d.status !== "pending") && (
            <button className="diff-btn diff-btn-clear" onClick={onClearResolved}>
              Clear Resolved
            </button>
          )}
        </div>
      </div>
      <div className="diff-list">
        {filteredDiffs.map((diff) => (
          <DiffFileCard
            key={diff.id}
            diff={diff}
            onAccept={onAccept}
            onReject={onReject}
          />
        ))}
        {filteredDiffs.length === 0 && (
          <div className="diff-empty-filter">
            No {filter} diffs.
          </div>
        )}
      </div>
    </div>
  );
}
