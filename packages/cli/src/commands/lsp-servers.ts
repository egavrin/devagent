import { execSync } from "node:child_process";

export interface LspServerDefinition {
  readonly command: string;
  readonly label: string;
  readonly install: string;
  readonly npmPackages: ReadonlyArray<string> | null;
}

export const LSP_SERVERS: LspServerDefinition[] = [
  { command: "typescript-language-server", label: "TypeScript/JavaScript", install: "npm i -g typescript-language-server typescript", npmPackages: ["typescript-language-server", "typescript"] },
  { command: "pyright-langserver", label: "Python (Pyright)", install: "npm i -g pyright", npmPackages: ["pyright"] },
  { command: "clangd", label: "C/C++ (clangd)", install: "apt install clangd / brew install llvm", npmPackages: null },
  { command: "rust-analyzer", label: "Rust", install: "rustup component add rust-analyzer", npmPackages: null },
  { command: "bash-language-server", label: "Bash/Shell", install: "npm i -g bash-language-server", npmPackages: ["bash-language-server"] },
];

export function commandExists(cmd: string): boolean {
  try {
    execSync("which " + cmd, { encoding: "utf-8", timeout: 3000, stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}
