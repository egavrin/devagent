/**
 * MemoryPanel — cross-session memory viewer with category filters and search.
 */

import { useState, useCallback } from "react";
import type { MemoryEntry, MemoryCategory } from "../types";

interface MemoryPanelProps {
  readonly memories: ReadonlyArray<MemoryEntry>;
  readonly summary: Record<MemoryCategory, number> | null;
  readonly loading: boolean;
  readonly onSearch: (query: string, category?: MemoryCategory) => void;
  readonly onDelete: (id: string) => void;
  readonly onRefresh: () => void;
}

const CATEGORIES: ReadonlyArray<{ id: MemoryCategory | "all"; label: string; color: string }> = [
  { id: "all", label: "All", color: "var(--text-accent)" },
  { id: "pattern", label: "Patterns", color: "#3498db" },
  { id: "decision", label: "Decisions", color: "#2ecc71" },
  { id: "mistake", label: "Mistakes", color: "#e74c3c" },
  { id: "preference", label: "Preferences", color: "#f39c12" },
  { id: "context", label: "Context", color: "#9b59b6" },
];

function categoryColor(cat: MemoryCategory): string {
  const found = CATEGORIES.find((c) => c.id === cat);
  return found?.color ?? "var(--text-muted)";
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function MemoryCard({
  memory,
  onDelete,
}: {
  readonly memory: MemoryEntry;
  readonly onDelete: (id: string) => void;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="memory-card">
      <div className="memory-card-header" onClick={() => setExpanded(!expanded)}>
        <div className="memory-card-info">
          <span
            className="memory-category-dot"
            style={{ background: categoryColor(memory.category) }}
          />
          <span className="memory-key">{memory.key}</span>
          <span className="memory-category-label">{memory.category}</span>
        </div>
        <div className="memory-card-meta">
          <span className="memory-relevance" title="Relevance score">
            {(memory.relevance * 100).toFixed(0)}%
          </span>
          <span className="memory-date">{formatDate(memory.updatedAt)}</span>
        </div>
      </div>
      {expanded && (
        <div className="memory-card-body">
          <div className="memory-content">{memory.content}</div>
          <div className="memory-card-footer">
            <div className="memory-tags">
              {memory.tags.map((tag) => (
                <span key={tag} className="memory-tag">{tag}</span>
              ))}
            </div>
            <div className="memory-card-actions">
              <span className="memory-access-count">
                {memory.accessCount} access{memory.accessCount !== 1 ? "es" : ""}
              </span>
              <button
                className="memory-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(memory.id);
                }}
                title="Delete this memory"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function MemoryPanel({
  memories,
  summary,
  loading,
  onSearch,
  onDelete,
  onRefresh,
}: MemoryPanelProps): React.JSX.Element {
  const [activeCategory, setActiveCategory] = useState<MemoryCategory | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");

  const handleCategoryChange = useCallback(
    (cat: MemoryCategory | "all") => {
      setActiveCategory(cat);
      if (cat === "all") {
        onSearch(searchQuery);
      } else {
        onSearch(searchQuery, cat);
      }
    },
    [onSearch, searchQuery],
  );

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (activeCategory === "all") {
        onSearch(searchQuery);
      } else {
        onSearch(searchQuery, activeCategory);
      }
    },
    [onSearch, searchQuery, activeCategory],
  );

  if (memories.length === 0 && !loading && !summary) {
    return (
      <div className="memory-panel">
        <div className="memory-header">
          <h2>Memory</h2>
          <button className="panel-refresh-btn" onClick={onRefresh}>Refresh</button>
        </div>
        <div className="panel-empty">
          <span className="panel-empty-icon">🧠</span>
          <h3>No Memories Yet</h3>
          <p>Memories are created during conversations and persist across sessions.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-panel">
      <div className="memory-header">
        <h2>Memory</h2>
        <button className="panel-refresh-btn" onClick={onRefresh} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {summary && (
        <div className="memory-summary">
          {CATEGORIES.filter((c) => c.id !== "all").map((cat) => {
            const count = summary[cat.id as MemoryCategory] ?? 0;
            return (
              <div key={cat.id} className="memory-summary-item" style={{ borderLeftColor: cat.color }}>
                <span className="memory-summary-count">{count}</span>
                <span className="memory-summary-label">{cat.label}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="memory-filters">
        <div className="memory-category-tabs">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              className={`memory-tab ${activeCategory === cat.id ? "memory-tab-active" : ""}`}
              onClick={() => handleCategoryChange(cat.id as MemoryCategory | "all")}
              style={
                activeCategory === cat.id
                  ? { borderBottomColor: cat.color, color: cat.color }
                  : undefined
              }
            >
              {cat.label}
            </button>
          ))}
        </div>
        <form className="memory-search" onSubmit={handleSearch}>
          <input
            type="text"
            className="memory-search-input"
            placeholder="Search memories..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </form>
      </div>

      <div className="memory-list">
        {memories.map((m) => (
          <MemoryCard key={m.id} memory={m} onDelete={onDelete} />
        ))}
        {memories.length === 0 && !loading && (
          <div className="memory-empty-filter">
            No memories found for this filter.
          </div>
        )}
      </div>
    </div>
  );
}
