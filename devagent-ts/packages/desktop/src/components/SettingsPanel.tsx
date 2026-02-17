/**
 * SettingsPanel — restructured with sections:
 * Provider, Approval Mode, Budget, Context Strategy, ArkTS.
 */

import { useState, useCallback } from "react";

interface SettingsPanelProps {
  readonly onProviderChange: (provider: string, model: string, apiKey?: string) => void;
  readonly onApprovalChange: (mode: string) => void;
}

const PROVIDERS: ReadonlyArray<{ id: string; name: string; models: ReadonlyArray<string> }> = [
  {
    id: "anthropic",
    name: "Anthropic",
    models: ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-3-20250414"],
  },
  {
    id: "openai",
    name: "OpenAI",
    models: ["gpt-5.2-2025-12-11", "gpt-4o-2024-08-06", "gpt-4o-mini"],
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    models: ["deepseek-chat", "deepseek-reasoner"],
  },
];

const APPROVAL_MODES: ReadonlyArray<{ id: string; label: string; desc: string }> = [
  { id: "suggest", label: "Suggest", desc: "Show diffs, ask before writing files or running commands" },
  { id: "auto-edit", label: "Auto-Edit", desc: "Auto-approve file writes, ask before commands" },
  { id: "full-auto", label: "Full-Auto", desc: "Auto-approve all operations (use with caution)" },
];

const CONTEXT_STRATEGIES: ReadonlyArray<{ id: string; label: string; desc: string }> = [
  { id: "sliding_window", label: "Sliding Window", desc: "Keep recent messages, drop old ones" },
  { id: "summarize", label: "Summarize", desc: "Summarize old messages via LLM call" },
  { id: "hybrid", label: "Hybrid", desc: "Summarize early, slide recent" },
];

export function SettingsPanel({
  onProviderChange,
  onApprovalChange,
}: SettingsPanelProps): React.JSX.Element {
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("claude-sonnet-4-20250514");
  const [apiKey, setApiKey] = useState("");
  const [approval, setApproval] = useState("suggest");
  const [saved, setSaved] = useState(false);
  const [maxIterations, setMaxIterations] = useState(20);
  const [costWarning, setCostWarning] = useState(1.0);
  const [contextStrategy, setContextStrategy] = useState("sliding_window");
  const [triggerRatio, setTriggerRatio] = useState(0.8);

  const currentProvider = PROVIDERS.find((p) => p.id === provider);

  const handleSaveProvider = useCallback(() => {
    onProviderChange(provider, model, apiKey || undefined);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }, [provider, model, apiKey, onProviderChange]);

  const handleApprovalChange = useCallback(
    (newMode: string) => {
      setApproval(newMode);
      onApprovalChange(newMode);
    },
    [onApprovalChange],
  );

  return (
    <div className="settings-panel">
      <h2>Settings</h2>

      {/* ── Provider ─────────────────────── */}
      <div className="settings-section">
        <h3>LLM Provider</h3>
        <div className="settings-field">
          <label htmlFor="provider-select">Provider</label>
          <select
            id="provider-select"
            value={provider}
            onChange={(e) => {
              setProvider(e.target.value);
              const newProvider = PROVIDERS.find((p) => p.id === e.target.value);
              if (newProvider?.models[0]) {
                setModel(newProvider.models[0]);
              }
            }}
          >
            {PROVIDERS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-field">
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {currentProvider?.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-field">
          <label htmlFor="api-key-input">API Key</label>
          <input
            id="api-key-input"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="settings-input"
          />
          <span className="settings-hint">
            Or set via DEVAGENT_API_KEY environment variable
          </span>
        </div>

        <button
          className="settings-save-btn"
          onClick={handleSaveProvider}
        >
          {saved ? "Saved!" : "Apply Provider Settings"}
        </button>
      </div>

      {/* ── Approval Mode ────────────────── */}
      <div className="settings-section">
        <h3>Approval Mode</h3>
        <div className="settings-approval-cards">
          {APPROVAL_MODES.map((mode) => (
            <button
              key={mode.id}
              className={`settings-approval-card ${approval === mode.id ? "settings-approval-active" : ""}`}
              onClick={() => handleApprovalChange(mode.id)}
            >
              <span className="settings-approval-radio">
                {approval === mode.id ? "◉" : "○"}
              </span>
              <div className="settings-approval-content">
                <span className="settings-approval-label">{mode.label}</span>
                <span className="settings-approval-desc">{mode.desc}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Budget ───────────────────────── */}
      <div className="settings-section">
        <h3>Budget</h3>
        <div className="settings-row">
          <div className="settings-field">
            <label htmlFor="max-iterations">Max Iterations</label>
            <input
              id="max-iterations"
              type="number"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value, 10) || 1)}
              min={1}
              max={100}
              className="settings-input settings-input-small"
            />
            <span className="settings-hint">Max tool-call iterations per query</span>
          </div>
          <div className="settings-field">
            <label htmlFor="cost-warning">Cost Warning ($)</label>
            <input
              id="cost-warning"
              type="number"
              value={costWarning}
              onChange={(e) => setCostWarning(parseFloat(e.target.value) || 0)}
              min={0}
              step={0.1}
              className="settings-input settings-input-small"
            />
            <span className="settings-hint">Warn when session cost exceeds this</span>
          </div>
        </div>
      </div>

      {/* ── Context Strategy ─────────────── */}
      <div className="settings-section">
        <h3>Context Strategy</h3>
        <div className="settings-field">
          <label htmlFor="context-strategy">Pruning Strategy</label>
          <select
            id="context-strategy"
            value={contextStrategy}
            onChange={(e) => setContextStrategy(e.target.value)}
          >
            {CONTEXT_STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label} — {s.desc}
              </option>
            ))}
          </select>
        </div>
        <div className="settings-field">
          <label htmlFor="trigger-ratio">Trigger Ratio</label>
          <div className="settings-range-row">
            <input
              id="trigger-ratio"
              type="range"
              min={0.5}
              max={0.95}
              step={0.05}
              value={triggerRatio}
              onChange={(e) => setTriggerRatio(parseFloat(e.target.value))}
              className="settings-range"
            />
            <span className="settings-range-value">{(triggerRatio * 100).toFixed(0)}%</span>
          </div>
          <span className="settings-hint">
            Start pruning when context fills to this percentage
          </span>
        </div>
      </div>

      {/* ── ArkTS ────────────────────────── */}
      <div className="settings-section">
        <h3>ArkTS</h3>
        <div className="settings-field">
          <label className="settings-toggle-label">
            <input type="checkbox" defaultChecked={false} className="settings-checkbox" />
            Enable ArkTS validation
          </label>
          <span className="settings-hint">
            Validate generated code against ArkTS constraints
          </span>
        </div>
      </div>
    </div>
  );
}
