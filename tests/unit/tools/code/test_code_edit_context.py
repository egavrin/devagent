import textwrap
from pathlib import Path

from ai_dev_agent.tools.code.code_edit import context as context_module
from ai_dev_agent.tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions


def _write(repo: Path, relative: str, content: str) -> Path:
    file_path = repo / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return file_path


def test_context_extraction_multiple_languages(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    python_body = """
        from functools import wraps

        def audit(func):
            @wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs)
            return inner

        class InvoiceService:
            def __init__(self, provider: str) -> None:
                self.provider = provider

            @audit
            def generate(self, amount: float) -> str:
                return f"invoice-{amount:.2f}"
    """
    typescript_body = """
        export interface OrderService<T> {
          fulfill(input: T): Promise<void>;
        }

        export type OrderId = string | number;

        export function createOrder<T>(initial: T): T {
          return initial;
        }
    """
    go_body = """
        package runtime

        type Processor interface {
            Execute() error
        }

        func NewProcessor() Processor {
            return nil
        }
    """

    _write(repo, "services/invoice.py", python_body)
    _write(repo, "services/orders.ts", typescript_body)
    _write(repo, "runtime/processor.go", go_body)

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(
            include_related_files=False,
            include_structure_summary=True,
            use_repo_map=False,
            max_files=6,
        ),
    )

    contexts = gatherer.gather_contexts(
        ["services/invoice.py", "services/orders.ts", "runtime/processor.go"]
    )
    outlines = {}
    for ctx in contexts:
        try:
            rel = ctx.path.relative_to(repo).as_posix()
        except ValueError:
            rel = ctx.path.name
        outlines[rel] = tuple(ctx.structure_outline)

    assert "services/invoice.py" in outlines
    assert any("class InvoiceService" in line for line in outlines["services/invoice.py"])
    assert any("function generate" in line for line in outlines["services/invoice.py"])

    assert "services/orders.ts" in outlines
    assert any("interface OrderService" in line for line in outlines["services/orders.ts"])
    assert any("type OrderId" in line for line in outlines["services/orders.ts"])
    assert any("function createOrder" in line for line in outlines["services/orders.ts"])

    assert "runtime/processor.go" in outlines
    assert any("func NewProcessor" in line for line in outlines["runtime/processor.go"])

    assert "__project_structure__.md" in outlines

    summary_ctx = next(ctx for ctx in contexts if ctx.reason == "project_structure_summary")
    assert summary_ctx.content.startswith("# Project Structure")

    # Exercise size limiting logic by constraining the maximum allowed total bytes.
    original_limit = gatherer.options.max_total_size
    gatherer.options.max_total_size = 1  # far smaller than any real file
    limited = gatherer._apply_size_limits(contexts)
    assert len(limited) == 1
    gatherer.options.max_total_size = original_limit


def test_context_symbol_resolution(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    main_body = """
        from libs.helpers import resolve_helper
        from services.contracts import PipelineResolver

        class MainFlow:
            def run(self) -> str:
                return resolve_helper(PipelineResolver().build())
    """
    helper_body = """
        def resolve_helper(payload: str) -> str:
            return f"resolved:{payload}"
    """
    contract_body = """
        export class PipelineResolver {
          build(): string {
            return "artifact"
          }
        }
    """

    _write(repo, "app/main.py", main_body)
    _write(repo, "libs/helpers.py", helper_body)
    _write(repo, "services/contracts.ts", contract_body)

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(
            include_related_files=True,
            include_structure_summary=False,
            use_repo_map=False,
            keyword_match_limit=4,
            max_files=6,
        ),
    )

    contexts = gatherer.gather_contexts(
        ["app/main.py"],
        task_description="Investigate PipelineResolver and resolve_helper usage across modules.",
        keywords=["resolve_helper", "PipelineResolver"],
    )

    rel_paths = {ctx.path.relative_to(repo).as_posix(): ctx for ctx in contexts}
    assert "app/main.py" in rel_paths
    assert "libs/helpers.py" in rel_paths  # discovered via keyword match
    assert rel_paths["libs/helpers.py"].reason.startswith("keyword_match")
    assert "services/contracts.ts" in rel_paths
    assert rel_paths["services/contracts.ts"].reason.startswith("keyword_match")

    all_symbols = {symbol for ctx in contexts for symbol in ctx.symbols}
    assert {"MainFlow", "PipelineResolver", "resolve_helper"} <= all_symbols

    # Force fallback search to exercise Python resolution path
    gatherer._rg_available = False
    gatherer._git_available = False

    search_hits = gatherer.search_files("PipelineResolver")
    assert sorted(search_hits) == ["app/main.py", "services/contracts.ts"]

    symbol_hits = gatherer.find_symbol_references("resolve_helper")
    assert {path for path, _line in symbol_hits} == {"app/main.py", "libs/helpers.py"}

    class DummyResult:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        if cmd[0] == "rg" and "--files-with-matches" in cmd:
            return DummyResult("app/main.py\nlibs/helpers.py\n")
        if cmd[0] == "rg" and "--line-number" in cmd:
            return DummyResult(
                "app/main.py:5:return resolve_helper(PipelineResolver().build())\n"
                "libs/helpers.py:1:def resolve_helper(payload: str) -> str:\n"
            )
        if cmd[0] == "git" and "-l" in cmd:
            return DummyResult("services/contracts.ts\n")
        if cmd[0] == "git" and "-n" in cmd:
            return DummyResult("services/contracts.ts:1:export class PipelineResolver {\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(context_module.subprocess, "run", fake_run)

    gatherer._rg_available = True
    gatherer._git_available = False
    search_hits_rg = gatherer.search_files("resolve_helper", file_types=["py"])
    assert search_hits_rg == ["app/main.py", "libs/helpers.py"]
    symbol_hits_rg = gatherer.find_symbol_references("resolve_helper")
    assert symbol_hits_rg == [("app/main.py", 5), ("libs/helpers.py", 1)]

    gatherer._rg_available = False
    gatherer._git_available = True
    search_hits_git = gatherer.search_files("PipelineResolver", file_types=["ts"])
    assert search_hits_git == ["services/contracts.ts"]
    symbol_hits_git = gatherer.find_symbol_references("PipelineResolver")
    assert symbol_hits_git == [("services/contracts.ts", 1)]

    gatherer._rg_available = False
    gatherer._git_available = False

    _write(repo, "docs/additional.py", "def extra_symbol():\n    return True\n")

    class FakeRepoMap:
        def __init__(self):
            self.calls = []

        def get_ranked_files(self, mentioned_files, mentioned_symbols, limit):
            self.calls.append((set(mentioned_files), set(mentioned_symbols), limit))
            return [("docs/additional.py", 0.88)]

    gatherer.repo_map = FakeRepoMap()
    gatherer.options.use_repo_map = True

    contexts_with_repo = gatherer.gather_contexts(
        ["app/main.py"],
        task_description="PipelineResolver should coordinate extraSymbol across stages",
        keywords=["resolve_helper"],
        chat_files=[repo / "app/main.py", tmp_path / "outside.py"],
    )

    rel_paths_repo = {
        ctx.path.relative_to(repo).as_posix(): ctx.reason
        for ctx in contexts_with_repo
        if ctx.path.exists()
    }
    assert "docs/additional.py" in rel_paths_repo
    assert rel_paths_repo["docs/additional.py"].startswith("repomap_rank")

    mentioned_symbols = gatherer.repo_map.calls[-1][1]
    assert "PipelineResolver" in mentioned_symbols
    assert "extraSymbol" in mentioned_symbols


def test_context_tree_sitter_integration(tmp_path, monkeypatch, caplog):
    repo = tmp_path / "repo"
    repo.mkdir()

    good_body = """
        export function compute(value: number): number {
          return value * 2;
        }
    """
    bad_body = """
        export function broken(value: number): number {
          return value *
    """

    _write(repo, "ts/good.ts", good_body)
    _write(repo, "ts/bad.ts", bad_body)

    class TrackingAnalyzer:
        def __init__(self, *_args, **_kwargs):
            self.calls: list[str] = []
            self.max_lines_per_file = 8

        def summarize_content(self, rel_path: str, content: str):
            self.calls.append(rel_path)
            if rel_path.endswith("bad.ts"):
                raise RuntimeError("tree-sitter parse failure")
            return [f"   1: function processed_{Path(rel_path).stem}"]

    monkeypatch.setattr(context_module, "TreeSitterProjectAnalyzer", TrackingAnalyzer)

    gatherer = ContextGatherer(
        repo,
        ContextGatheringOptions(include_structure_summary=False, include_related_files=False),
    )

    analyzer = gatherer._structure_analyzer
    caplog.set_level("WARNING")

    contexts = gatherer.gather_contexts(["ts/good.ts", "ts/bad.ts"])

    assert sorted(analyzer.calls) == ["ts/bad.ts", "ts/good.ts"]
    outlines = {ctx.path.relative_to(repo).as_posix(): ctx.structure_outline for ctx in contexts}

    assert outlines["ts/good.ts"] == ["   1: function processed_good"]
    assert outlines["ts/bad.ts"] == []  # tree-sitter failure should not break context gathering

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("tree-sitter parse failure" in message for message in log_messages)
