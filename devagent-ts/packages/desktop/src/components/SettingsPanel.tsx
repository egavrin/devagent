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

export function SettingsPanel({
  onProviderChange,
  onApprovalChange,
}: SettingsPanelProps): React.JSX.Element {
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("claude-sonnet-4-20250514");
  const [apiKey, setApiKey] = useState("");
  const [approval, setApproval] = useState("suggest");
  const [saved, setSaved] = useState(false);

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

      <div className="settings-section">
        <h3>Approval Mode</h3>
        <div className="settings-field">
          <label>
            <input
              type="radio"
              name="approval"
              value="suggest"
              checked={approval === "suggest"}
              onChange={() => handleApprovalChange("suggest")}
            />
            Suggest — show diffs, ask before writing
          </label>
          <label>
            <input
              type="radio"
              name="approval"
              value="auto-edit"
              checked={approval === "auto-edit"}
              onChange={() => handleApprovalChange("auto-edit")}
            />
            Auto-Edit — auto-approve file writes
          </label>
          <label>
            <input
              type="radio"
              name="approval"
              value="full-auto"
              checked={approval === "full-auto"}
              onChange={() => handleApprovalChange("full-auto")}
            />
            Full-Auto — auto-approve everything
          </label>
        </div>
      </div>

      <div className="settings-section">
        <h3>Budget</h3>
        <div className="settings-field">
          <label htmlFor="max-iterations">Max Iterations</label>
          <input
            id="max-iterations"
            type="number"
            defaultValue={20}
            min={1}
            max={100}
            className="settings-input settings-input-small"
          />
        </div>
      </div>
    </div>
  );
}
