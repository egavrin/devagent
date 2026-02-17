import type { AppView } from "../types";

interface SidebarProps {
  readonly view: AppView;
  readonly onViewChange: (view: AppView) => void;
  readonly diffCount?: number;
}

export function Sidebar({
  view,
  onViewChange,
  diffCount,
}: SidebarProps): React.JSX.Element {
  const items: ReadonlyArray<{ id: AppView; label: string; icon: string; badge?: number }> = [
    { id: "chat", label: "Chat", icon: "💬" },
    { id: "diff", label: "Diffs", icon: "📝", badge: diffCount },
    { id: "skills", label: "Skills", icon: "🧩" },
    { id: "mcp", label: "MCP", icon: "🔌" },
    { id: "memory", label: "Memory", icon: "🧠" },
    { id: "settings", label: "Settings", icon: "⚙️" },
  ];

  return (
    <nav className="sidebar">
      {items.map((item) => (
        <button
          key={item.id}
          className={`sidebar-item ${view === item.id ? "sidebar-item-active" : ""}`}
          onClick={() => onViewChange(item.id)}
          title={item.label}
        >
          <span className="sidebar-icon">{item.icon}</span>
          <span className="sidebar-label">{item.label}</span>
          {item.badge != null && item.badge > 0 && (
            <span className="sidebar-badge">{item.badge}</span>
          )}
        </button>
      ))}
    </nav>
  );
}
