/**
 * SkillsPanel — list discovered skills grouped by source (project/global).
 */

import { useState, useCallback } from "react";
import type { SkillInfo } from "../types";

interface SkillsPanelProps {
  readonly skills: ReadonlyArray<SkillInfo>;
  readonly loading: boolean;
  readonly onRefresh: () => void;
  readonly onLoadSkill: (name: string) => void;
  readonly loadedInstructions: Record<string, string>;
}

function SkillCard({
  skill,
  instructions,
  onLoad,
}: {
  readonly skill: SkillInfo;
  readonly instructions: string | undefined;
  readonly onLoad: (name: string) => void;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(false);

  const handleToggle = useCallback(() => {
    if (!expanded && !instructions) {
      onLoad(skill.name);
    }
    setExpanded(!expanded);
  }, [expanded, instructions, onLoad, skill.name]);

  return (
    <div className="skill-card">
      <div className="skill-card-header" onClick={handleToggle}>
        <div className="skill-card-info">
          <span className="skill-card-toggle">{expanded ? "▼" : "▶"}</span>
          <span className="skill-card-name">{skill.name}</span>
          <span className={`skill-source-badge skill-source-${skill.source}`}>
            {skill.source}
          </span>
        </div>
      </div>
      {skill.description && (
        <div className="skill-card-desc">{skill.description}</div>
      )}
      {expanded && (
        <div className="skill-card-body">
          {instructions ? (
            <pre className="skill-instructions">{instructions}</pre>
          ) : (
            <div className="skill-loading">Loading instructions...</div>
          )}
        </div>
      )}
    </div>
  );
}

export function SkillsPanel({
  skills,
  loading,
  onRefresh,
  onLoadSkill,
  loadedInstructions,
}: SkillsPanelProps): React.JSX.Element {
  const projectSkills = skills.filter((s) => s.source === "project");
  const globalSkills = skills.filter((s) => s.source === "global");

  if (skills.length === 0 && !loading) {
    return (
      <div className="skills-panel">
        <div className="skills-header">
          <h2>Skills</h2>
          <button className="panel-refresh-btn" onClick={onRefresh}>Refresh</button>
        </div>
        <div className="panel-empty">
          <span className="panel-empty-icon">🧩</span>
          <h3>No Skills Found</h3>
          <p>
            Add Markdown files to <code>.devagent/skills/</code> in your project
            or <code>~/.config/devagent/skills/</code> globally.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="skills-panel">
      <div className="skills-header">
        <h2>Skills ({skills.length})</h2>
        <button className="panel-refresh-btn" onClick={onRefresh} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>
      <div className="skills-list">
        {projectSkills.length > 0 && (
          <div className="skills-group">
            <h3 className="skills-group-title">Project Skills</h3>
            {projectSkills.map((skill) => (
              <SkillCard
                key={skill.name}
                skill={skill}
                instructions={loadedInstructions[skill.name]}
                onLoad={onLoadSkill}
              />
            ))}
          </div>
        )}
        {globalSkills.length > 0 && (
          <div className="skills-group">
            <h3 className="skills-group-title">Global Skills</h3>
            {globalSkills.map((skill) => (
              <SkillCard
                key={skill.name}
                skill={skill}
                instructions={loadedInstructions[skill.name]}
                onLoad={onLoadSkill}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
