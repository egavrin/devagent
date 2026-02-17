import { useState, useCallback } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { Sidebar } from "./components/Sidebar";
import { SettingsPanel } from "./components/SettingsPanel";
import { Toolbar } from "./components/Toolbar";
import { StatusBar } from "./components/StatusBar";
import { ApprovalDialog } from "./components/ApprovalDialog";
import { useEngineBridge } from "./hooks/useEngineBridge";
import type { AppMode, AppView } from "./types";

export function App(): React.JSX.Element {
  const [mode, setMode] = useState<AppMode>("act");
  const [view, setView] = useState<AppView>("chat");

  const bridge = useEngineBridge();

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

  return (
    <div className="app">
      <Toolbar
        mode={mode}
        onModeChange={handleModeChange}
        view={view}
        onViewChange={setView}
        isStreaming={bridge.state.streaming}
      />
      <div className="app-content">
        <Sidebar view={view} onViewChange={setView} />
        <main className="main-panel">
          {view === "chat" && (
            <ChatPanel
              messages={bridge.messages}
              isStreaming={bridge.state.streaming}
              onSendMessage={handleSendMessage}
              onClear={bridge.clearMessages}
              onAbort={bridge.abort}
            />
          )}
          {view === "settings" && (
            <SettingsPanel
              onProviderChange={handleProviderChange}
              onApprovalChange={handleApprovalChange}
            />
          )}
          {view === "diff" && (
            <div className="placeholder-panel">
              <h2>Diff Viewer</h2>
              <p>File diffs will appear here during edit operations.</p>
            </div>
          )}
        </main>
      </div>
      <StatusBar
        connected={bridge.state.connected}
        ready={bridge.state.ready}
        streaming={bridge.state.streaming}
        error={bridge.state.error}
        toolCount={bridge.tools.length}
      />
      <ApprovalDialog
        requests={bridge.approvalRequests}
        onRespond={bridge.respondApproval}
      />
    </div>
  );
}
