import type { AppView } from "../types";

interface SidebarProps {
  readonly view: AppView;
  readonly onViewChange: (view: AppView) => void;
}

export function Sidebar({
  view,
  onViewChange,
}: SidebarProps): React.JSX.Element {
  const items: ReadonlyArray<{ id: AppView; label: string; icon: string }> = [
    { id: "chat", label: "Chat", icon: "💬" },
    { id: "diff", label: "Diffs", icon: "📝" },
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
        </button>
      ))}
    </nav>
  );
}
