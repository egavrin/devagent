/**
 * SubagentPanel — displays active/completed subagent work.
 */

import { Box, Text } from "ink";
import React, { useState, useEffect } from "react";

export interface SubagentState {
  readonly agentId: string;
  readonly agentType: string;
  readonly laneLabel?: string | null;
  readonly status: "running" | "completed" | "error";
  readonly iteration: number;
  readonly startedAt: number;
  readonly activity: string;
  readonly quality?: { score: number; completeness: string };
  readonly error?: string;
}

interface SubagentPanelProps {
  readonly agents: ReadonlyMap<string, SubagentState>;
}

export function SubagentPanel({ agents }: SubagentPanelProps): React.ReactElement | null {
  if (agents.size === 0) return null;

  return (
    <Box flexDirection="column" marginTop={1}>
      {[...agents.values()].map((agent) => (
        <SubagentRow key={agent.agentId} agent={agent} />
      ))}
    </Box>
  );
}
function SubagentRow({ agent }: { agent: SubagentState }): React.ReactElement {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (agent.status !== "running") return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [agent.status]);

  const elapsedStr = formatElapsed(agent, now);
  const statusIcon = getStatusIcon(agent.status);
  const statusColor = getStatusColor(agent.status);
  const label = formatAgentLabel(agent);
  const qualityStr = formatQuality(agent);

  return (
    <Box>
      <Text color={statusColor}>  {statusIcon} </Text>
      <Text dimColor>Subagent {agent.agentId.slice(-6)} </Text>
      <Text bold>{label}</Text>
      <Text dimColor>
        {" "}iter {agent.iteration} {elapsedStr}
        {qualityStr}
        {agent.error ? ` — ${agent.error.slice(0, 40)}` : ""}
        {agent.activity ? ` │ ${agent.activity}` : ""}
      </Text>
    </Box>
  );
}

function formatElapsed(agent: SubagentState, now: number): string {
  const elapsed = ((agent.status === "running" ? now : Date.now()) - agent.startedAt) / 1000;
  return elapsed >= 60
    ? `${Math.floor(elapsed / 60)}m ${Math.round(elapsed % 60)}s`
    : `${elapsed.toFixed(0)}s`;
}

function getStatusIcon(status: SubagentState["status"]): string {
  if (status === "completed") return "✓";
  if (status === "error") return "✗";
  return "⠋";
}

function getStatusColor(status: SubagentState["status"]): "green" | "red" | "yellow" {
  if (status === "completed") return "green";
  if (status === "error") return "red";
  return "yellow";
}

function formatAgentLabel(agent: SubagentState): string {
  return agent.laneLabel ? `${agent.agentType} ${agent.laneLabel}` : agent.agentType;
}

function formatQuality(agent: SubagentState): string {
  return agent.quality
    ? ` (${agent.quality.score.toFixed(2)}, ${agent.quality.completeness})`
    : "";
}
