from ai_dev_agent.core.context import ContextBuilder


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
