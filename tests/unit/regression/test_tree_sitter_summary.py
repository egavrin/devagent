from ai_dev_agent.tools.code.code_edit.tree_sitter_analysis import TreeSitterProjectAnalyzer


def test_python_outline_identifies_key_symbols(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)

    content = """
    class Alpha:
        def beta(self):
            return 1


    async def gamma():
        return 2
    """.strip()

    outline = analyzer.summarize_content("pkg/sample.py", content)

    assert any("class Alpha" in entry for entry in outline)
    assert any("function beta" in entry or "fn beta" in entry for entry in outline)
    assert any("function gamma" in entry for entry in outline)


def test_typescript_outline_catches_exports(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path)

    content = """
    export class Example {}
    export function run() {}
    interface Options {}
    type Alias = string;
    enum Color { Red, Blue }
    """.strip()

    outline = analyzer.summarize_content("src/module.ts", content)

    assert any("class Example" in entry for entry in outline)
    assert any("function run" in entry for entry in outline)
    assert any("interface Options" in entry for entry in outline)
    assert any("type Alias" in entry for entry in outline)
    assert any("enum Color" in entry for entry in outline)


def test_project_summary_produces_markdown(tmp_path):
    analyzer = TreeSitterProjectAnalyzer(tmp_path, max_files=1, max_lines_per_file=2)

    summary = analyzer.build_project_summary(
        [
            (
                "pkg/sample.py",
                """
                class Foo:
                    pass


                def bar():
                    return 42
                """.strip(),
            )
        ]
    )

    assert summary is not None
    assert "# Project Structure" in summary
    assert "### pkg/sample.py" in summary
    assert "Foo" in summary
    assert "bar" in summary
