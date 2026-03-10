import { existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync } from "fs";
import { join } from "path";

export interface ArtifactMetadata {
  runId: string;
  phase: string;
  timestamp: string;
  type: "input" | "output" | "events" | "summary";
  path: string;
}

export class ArtifactStore {
  private baseDir: string;

  constructor(baseDir?: string) {
    this.baseDir = baseDir ?? join(
      process.env.HOME ?? "~",
      ".config", "devagent", "artifacts"
    );
  }

  getRunDir(runId: string): string {
    const dir = join(this.baseDir, runId);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    return dir;
  }

  save(runId: string, name: string, data: unknown): string {
    const dir = this.getRunDir(runId);
    const filePath = join(dir, name);
    const content = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    writeFileSync(filePath, content, "utf-8");
    return filePath;
  }

  load(runId: string, name: string): unknown | null {
    const filePath = join(this.getRunDir(runId), name);
    if (!existsSync(filePath)) {
      return null;
    }
    const raw = readFileSync(filePath, "utf-8");
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }

  listRuns(): string[] {
    if (!existsSync(this.baseDir)) {
      return [];
    }
    return readdirSync(this.baseDir, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort()
      .reverse();
  }

  listArtifacts(runId: string): string[] {
    const dir = join(this.baseDir, runId);
    if (!existsSync(dir)) {
      return [];
    }
    return readdirSync(dir, { withFileTypes: true })
      .filter((f) => f.isFile())
      .map((f) => f.name)
      .sort();
  }
}
