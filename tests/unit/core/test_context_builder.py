from pathlib import Path

from ai_dev_agent.core.context import ContextBuilder, get_context_builder


def test_python_file_count_skips_ignored_directories(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    src_dir = workspace / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')", encoding="utf-8")

    hidden_dir = workspace / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "hidden.py").write_text("print('hidden')", encoding="utf-8")

    git_dir = workspace / ".git"
    git_dir.mkdir()
    (git_dir / "config.py").write_text("print('git')", encoding="utf-8")

    venv_dir = workspace / "venv"
    venv_dir.mkdir()
    (venv_dir / "ignored.py").write_text("print('venv')", encoding="utf-8")

    builder = ContextBuilder(workspace)
    context = builder.build_project_context()

    assert context["python_files_count"] == 1


def test_build_system_context_windows(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.platform.mac_ver", lambda: ("", ("", "", ""))
    )
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.version", lambda: "10.0.12345")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.release", lambda: "10")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.machine", lambda: "AMD64")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.processor", lambda: "Intel")
    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.platform.python_version", lambda: "3.11.0"
    )
    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.shutil.which",
        lambda tool: r"C:\Tools\python.exe" if tool == "python" else None,
    )

    context = builder.build_system_context()

    assert context["shell"] == r"C:\Windows\System32\cmd.exe"
    assert context["shell_type"] == "windows"
    assert context["path_separator"] == "\\"
    assert context["command_separator"] == "&"
    assert context["is_unix"] is False
    assert "python" in context["available_tools"]


def test_build_project_context_with_outline(monkeypatch, tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (workspace / ".git" / "refs" / "heads" / "main").write_text("abc123", encoding="utf-8")

    (workspace / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (workspace / "package.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.generate_repo_outline",
        lambda *args, **kwargs: "OUTLINE",
    )

    builder = ContextBuilder(workspace)
    context = builder.build_project_context(include_outline=True)

    assert context["has_git"] is True
    assert context["git_branch"] == "main"
    assert context["has_pyproject_toml"] is True
    assert context["has_package_json"] is True
    assert context["project_outline"] == "OUTLINE"


def test_load_gitignore_entries_and_gitignore_checks(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "logs").mkdir()
    (workspace / ".gitignore").write_text(
        """
        # comment
        /build/
        logs
        data.txt
        data/*.log
        !keep.txt
        """,
        encoding="utf-8",
    )

    builder = ContextBuilder(workspace)
    ignored_dirs, ignored_files = builder._load_gitignore_entries()

    assert Path("build") in ignored_dirs
    assert Path("logs") in ignored_dirs
    assert Path("data.txt") in ignored_files
    # pattern with wildcard should be skipped
    assert all("*.log" not in str(entry) for entry in ignored_dirs)
    assert all("*.log" not in str(entry) for entry in ignored_files)

    inside_logs = Path("logs/output.log")
    assert builder._is_gitignored(inside_logs, ignored_dirs, ignored_files) is True
    assert builder._is_gitignored(Path("src/main.py"), ignored_dirs, ignored_files) is False


def test_build_tool_context_includes_system(monkeypatch, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    monkeypatch.setattr(
        ContextBuilder,
        "build_system_context",
        lambda self: {"os": "TestOS", "shell": "/bin/zsh"},
    )

    tool_context = builder.build_tool_context("shell", extra="value")
    assert tool_context["tool_name"] == "shell"
    assert tool_context["os"] == "TestOS"
    assert tool_context["extra"] == "value"


def test_get_context_builder_singleton(tmp_path):
    workspace_a = tmp_path / "a"
    workspace_b = tmp_path / "b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    builder_a = get_context_builder(workspace_a)
    builder_a_again = get_context_builder(workspace_a)
    builder_b = get_context_builder(workspace_b)

    assert builder_a is builder_a_again
    assert builder_b is not builder_a


def test_build_system_context_generic_platform(monkeypatch, tmp_path):
    workspace = tmp_path / "generic"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.system", lambda: "FreeBSD")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.release", lambda: "14.0")
    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.platform.version", lambda: "14.0-RELEASE"
    )
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.machine", lambda: "arm64")
    monkeypatch.setattr("ai_dev_agent.core.context.builder.platform.processor", lambda: "")
    monkeypatch.setattr(
        "ai_dev_agent.core.context.builder.platform.python_version", lambda: "3.11.1"
    )
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.delenv("COMSPEC", raising=False)
    monkeypatch.setattr("ai_dev_agent.core.context.builder.shutil.which", lambda tool: None)

    ctx = builder.build_system_context()

    assert ctx["os"] == "FreeBSD"
    assert ctx["os_version"] == "14.0"
    assert ctx["shell"] == "/bin/sh"
    assert ctx["is_unix"] is False
    assert ctx["shell_type"] == "windows"
    assert ctx["command_separator"] == "&"
    assert ctx["path_separator"] == "\\"


def test_load_gitignore_entries_handles_read_errors(monkeypatch, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    gitignore = workspace / ".gitignore"
    gitignore.write_text("build/\n", encoding="utf-8")

    original_read_text = Path.read_text

    def failing_read_text(self, *args, **kwargs):
        if self == gitignore:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read_text)

    builder = ContextBuilder(workspace)
    ignored_dirs, ignored_files = builder._load_gitignore_entries()

    assert ignored_dirs == set()
    assert ignored_files == set()


def test_build_agent_context_combines_sources(monkeypatch, tmp_path):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    monkeypatch.setattr(ContextBuilder, "build_system_context", lambda self: {"os": "TestOS"})
    monkeypatch.setattr(
        ContextBuilder, "build_project_context", lambda self: {"workspace_name": "agent"}
    )

    ctx = builder.build_agent_context("design", extra="value")

    assert ctx["agent_type"] == "design"
    assert ctx["os"] == "TestOS"
    assert ctx["workspace_name"] == "agent"
    assert ctx["extra"] == "value"


def test_build_full_context_toggles(monkeypatch, tmp_path):
    workspace = tmp_path / "full"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    monkeypatch.setattr(ContextBuilder, "build_system_context", lambda self: {"os": "TestOS"})
    monkeypatch.setattr(ContextBuilder, "build_project_context", lambda self: {"workspace": "full"})

    ctx = builder.build_full_context(include_system=False, include_project=True)
    assert "os" not in ctx and ctx["workspace"] == "full"

    ctx_all = builder.build_full_context(include_system=True, include_project=False)
    assert ctx_all == {"os": "TestOS"}
