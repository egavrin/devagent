import { useState, useCallback, useEffect } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { Sidebar } from "./components/Sidebar";
import { SettingsPanel } from "./components/SettingsPanel";
import { Toolbar } from "./components/Toolbar";
import { StatusBar } from "./components/StatusBar";
import { ApprovalDialog } from "./components/ApprovalDialog";
import { DiffPanel } from "./components/DiffPanel";
import { SkillsPanel } from "./components/SkillsPanel";
import { McpPanel } from "./components/McpPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import { useEngineBridge } from "./hooks/useEngineBridge";
import { useDiffs } from "./hooks/useDiffs";
import { useSkills } from "./hooks/useSkills";
import { useMcp } from "./hooks/useMcp";
import { useMemory } from "./hooks/useMemory";
import { useCommands } from "./hooks/useCommands";
import type { AppMode, AppView } from "./types";

export function App(): React.JSX.Element {
  const [mode, setMode] = useState<AppMode>("act");
  const [view, setView] = useState<AppView>("chat");

  const bridge = useEngineBridge();
  const diffsHook = useDiffs(bridge);
  const skillsHook = useSkills(bridge);
  const mcpHook = useMcp(bridge);
  const memoryHook = useMemory(bridge);
  const commandsHook = useCommands(bridge);

  const handleSendMessage = useCallback(
    (content: string) => {
      bridge.sendQuery(content, mode);
    },
    [bridge, mode],
  );

  const handleModeChange = useCallback(
    (newMode: AppMode) => {
      setMode(newMode);
    },
    [],
  );

  const handleProviderChange = useCallback(
    (provider: string, model: string, apiKey?: string) => {
      bridge.setProvider(provider, model, apiKey);
    },
    [bridge],
  );

  const handleApprovalChange = useCallback(
    (approvalMode: string) => {
      bridge.setApproval(approvalMode);
    },
    [bridge],
  );

  const pendingDiffCount = diffsHook.diffs.filter((d) => d.status === "pending").length;

  // Global keyboard shortcuts for sidebar navigation
  useEffect(() => {
    const VIEWS: ReadonlyArray<AppView> = ["chat", "diff", "skills", "mcp", "memory", "settings"];

    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle when not focused on inputs
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT") {
        return;
      }

      // Alt+1..6 to switch views
      if (e.altKey && e.key >= "1" && e.key <= "6") {
        e.preventDefault();
        const idx = parseInt(e.key, 10) - 1;
        const newView = VIEWS[idx];
        if (newView) setView(newView);
        return;
      }

      // Escape while streaming to abort
      if (e.key === "Escape" && bridge.state.streaming) {
        bridge.abort();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [bridge]);

  return (
    <div className="app">
      <Toolbar
        mode={mode}
        onModeChange={handleModeChange}
        isStreaming={bridge.state.streaming}
        costState={bridge.costState}
        workingDir={bridge.workingDir}
      />
      <div className="app-content">
        <Sidebar
          view={view}
          onViewChange={setView}
          diffCount={pendingDiffCount}
        />
        <main className="main-panel">
          {view === "chat" && (
            <ChatPanel
              messages={bridge.messages}
              isStreaming={bridge.state.streaming}
              onSendMessage={handleSendMessage}
              onClear={bridge.clearMessages}
              onAbort={bridge.abort}
              commands={commandsHook.commands}
              onExecuteCommand={commandsHook.executeCommand}
            />
          )}
          {view === "diff" && (
            <DiffPanel
              diffs={diffsHook.diffs}
              hasPending={diffsHook.hasPending}
              onAccept={diffsHook.acceptDiff}
              onReject={diffsHook.rejectDiff}
              onAcceptAll={diffsHook.acceptAll}
              onClearResolved={diffsHook.clearResolved}
            />
          )}
          {view === "skills" && (
            <SkillsPanel
              skills={skillsHook.skills}
              loading={skillsHook.loading}
              onRefresh={skillsHook.refresh}
              onLoadSkill={skillsHook.loadSkill}
              loadedInstructions={skillsHook.loadedInstructions}
            />
          )}
          {view === "mcp" && (
            <McpPanel
              servers={mcpHook.servers}
              loading={mcpHook.loading}
              onRefresh={mcpHook.refresh}
              onRestart={mcpHook.restartServer}
            />
          )}
          {view === "memory" && (
            <MemoryPanel
              memories={memoryHook.memories}
              summary={memoryHook.summary}
              loading={memoryHook.loading}
              onSearch={memoryHook.search}
              onDelete={memoryHook.deleteMemory}
              onRefresh={memoryHook.refresh}
            />
          )}
          {view === "settings" && (
            <SettingsPanel
              onProviderChange={handleProviderChange}
              onApprovalChange={handleApprovalChange}
            />
          )}
        </main>
      </div>
      <StatusBar
        connected={bridge.state.connected}
        ready={bridge.state.ready}
        streaming={bridge.state.streaming}
        error={bridge.state.error}
        toolCount={bridge.tools.length}
        workingDir={bridge.workingDir}
      />
      <ApprovalDialog
        requests={bridge.approvalRequests}
        onRespond={bridge.respondApproval}
      />
    </div>
  );
}
