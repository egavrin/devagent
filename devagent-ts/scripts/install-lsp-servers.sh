#!/usr/bin/env bash
#
# Install LSP servers for DevAgent multi-language validation.
#
# Supported languages: TypeScript, JavaScript, Python, C/C++, Rust, Bash, ArkTS
#
# Usage:
#   ./scripts/install-lsp-servers.sh          # Install all available
#   ./scripts/install-lsp-servers.sh ts       # Install TypeScript only
#   ./scripts/install-lsp-servers.sh ts py    # Install TypeScript + Python
#
# Prerequisites:
#   - bun (or npm) for JS/TS-based servers
#   - pip for Python-based servers
#   - rustup for rust-analyzer
#   - Xcode CLI tools for clangd (macOS)
#
set -euo pipefail

# ─── Helpers ───────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
DIM='\033[2m'
NC='\033[0m' # No Color

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }
dim()  { echo -e "${DIM}$1${NC}"; }

has_cmd() { command -v "$1" &>/dev/null; }

# Detect package manager for global JS installs
JS_PM=""
if has_cmd bun; then
  JS_PM="bun"
elif has_cmd npm; then
  JS_PM="npm"
else
  warn "Neither bun nor npm found. Cannot install JS-based LSP servers."
fi

install_global_js() {
  local pkg="$1"
  if [ -z "$JS_PM" ]; then
    err "Cannot install $pkg — no JS package manager available"
    return 1
  fi
  if [ "$JS_PM" = "bun" ]; then
    bun install -g "$pkg"
  else
    npm install -g "$pkg"
  fi
}

# ─── Server Installers ────────────────────────────────────

install_typescript() {
  echo ""
  echo "── TypeScript / JavaScript LSP ──"
  if has_cmd typescript-language-server; then
    ok "typescript-language-server already installed ($(which typescript-language-server))"
  else
    dim "Installing typescript-language-server + typescript..."
    install_global_js typescript-language-server
    install_global_js typescript
    if has_cmd typescript-language-server; then
      ok "typescript-language-server installed"
    else
      err "typescript-language-server installation failed"
      return 1
    fi
  fi
}

install_python() {
  echo ""
  echo "── Python LSP (pyright) ──"
  if has_cmd pyright-langserver; then
    ok "pyright-langserver already installed ($(which pyright-langserver))"
  elif has_cmd pyright; then
    ok "pyright already installed ($(which pyright)) — pyright-langserver may be available"
  else
    if has_cmd pip3; then
      dim "Installing pyright via pip3..."
      pip3 install --user pyright
    elif has_cmd pip; then
      dim "Installing pyright via pip..."
      pip install --user pyright
    elif [ -n "$JS_PM" ]; then
      dim "Installing pyright via $JS_PM (global)..."
      install_global_js pyright
    else
      err "Cannot install pyright — no pip or JS package manager available"
      return 1
    fi

    if has_cmd pyright-langserver || has_cmd pyright; then
      ok "pyright installed"
    else
      err "pyright installation failed"
      return 1
    fi
  fi
}

install_clangd() {
  echo ""
  echo "── C / C++ LSP (clangd) ──"
  if has_cmd clangd; then
    ok "clangd already installed ($(which clangd))"
    dim "Version: $(clangd --version 2>&1 | head -1)"
  else
    if [[ "$(uname)" == "Darwin" ]]; then
      dim "clangd is included with Xcode Command Line Tools."
      dim "Install with: xcode-select --install"
      warn "clangd not found — install Xcode CLI tools"
    elif has_cmd apt-get; then
      dim "Installing clangd via apt..."
      sudo apt-get install -y clangd
    elif has_cmd dnf; then
      dim "Installing clangd via dnf..."
      sudo dnf install -y clang-tools-extra
    elif has_cmd brew; then
      dim "Installing clangd via brew..."
      brew install llvm
    else
      err "Cannot install clangd — install manually from https://clangd.llvm.org/"
      return 1
    fi
  fi
}

install_rust() {
  echo ""
  echo "── Rust LSP (rust-analyzer) ──"
  if has_cmd rust-analyzer; then
    ok "rust-analyzer already installed ($(which rust-analyzer))"
  elif has_cmd rustup; then
    dim "Installing rust-analyzer via rustup..."
    rustup component add rust-analyzer
    if has_cmd rust-analyzer; then
      ok "rust-analyzer installed"
    else
      err "rust-analyzer installation failed"
      return 1
    fi
  else
    err "rustup not found. Install Rust: https://rustup.rs/"
    return 1
  fi
}

install_bash() {
  echo ""
  echo "── Bash LSP ──"
  if has_cmd bash-language-server; then
    ok "bash-language-server already installed ($(which bash-language-server))"
  else
    dim "Installing bash-language-server..."
    install_global_js bash-language-server
    if has_cmd bash-language-server; then
      ok "bash-language-server installed"
    else
      err "bash-language-server installation failed"
      return 1
    fi
  fi
}

# ─── Main ─────────────────────────────────────────────────

TARGETS=("$@")
ALL_TARGETS=(ts py clangd rust bash)

if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=("${ALL_TARGETS[@]}")
  echo "Installing all LSP servers..."
else
  echo "Installing selected LSP servers: ${TARGETS[*]}"
fi

FAILED=0

for target in "${TARGETS[@]}"; do
  case "$target" in
    ts|typescript|javascript|js)
      install_typescript || ((FAILED++)) || true
      ;;
    py|python)
      install_python || ((FAILED++)) || true
      ;;
    c|cpp|clangd)
      install_clangd || ((FAILED++)) || true
      ;;
    rust|rs)
      install_rust || ((FAILED++)) || true
      ;;
    bash|sh|shellscript)
      install_bash || ((FAILED++)) || true
      ;;
    *)
      warn "Unknown target: $target (valid: ts, py, clangd, rust, bash)"
      ;;
  esac
done

echo ""
echo "── Summary ──"

check_server() {
  local name="$1"
  local cmd="$2"
  if has_cmd "$cmd"; then
    ok "$name: $(which "$cmd")"
  else
    err "$name: not found"
  fi
}

check_server "TypeScript/JS" "typescript-language-server"
check_server "Python"        "pyright-langserver"
check_server "C/C++"         "clangd"
check_server "Rust"          "rust-analyzer"
check_server "Bash"          "bash-language-server"

echo ""
if [ $FAILED -eq 0 ]; then
  ok "All LSP servers ready!"
else
  warn "$FAILED server(s) could not be installed. See errors above."
fi
