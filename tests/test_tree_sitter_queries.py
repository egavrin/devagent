"""Tests for tree-sitter query functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_dev_agent.core.tree_sitter.queries import (
    AST_QUERY_TEMPLATES,
    LANGUAGE_ALIASES,
    SUMMARY_QUERY_TEMPLATES,
    build_capture_query,
    build_field_capture_query,
    get_ast_query,
    get_scm_file_path,
    get_summary_queries,
    iter_ast_queries,
    load_query_from_file,
    normalise_language,
)


class TestLanguageNormalization:
    """Test language name normalization."""

    def test_normalise_language_with_alias(self):
        """Test normalizing language names using aliases."""
        assert normalise_language("c++") == "cpp"
        assert normalise_language("c#") == "csharp"
        assert normalise_language("c-sharp") == "csharp"
        assert normalise_language("c_sharp") == "csharp"
        assert normalise_language("py") == "python"
        assert normalise_language("js") == "javascript"
        assert normalise_language("ts") == "typescript"

    def test_normalise_language_without_alias(self):
        """Test normalizing language names without aliases."""
        assert normalise_language("python") == "python"
        assert normalise_language("java") == "java"
        assert normalise_language("rust") == "rust"
        assert normalise_language("go") == "go"

    def test_normalise_language_case_insensitive(self):
        """Test that normalization is case insensitive."""
        assert normalise_language("C++") == "cpp"
        assert normalise_language("PY") == "python"
        assert normalise_language("Python") == "python"
        assert normalise_language("JAVA") == "java"


class TestASTQueries:
    """Test AST query functions."""

    def test_get_ast_query_valid(self):
        """Test getting a valid AST query."""
        query = get_ast_query("python", "find_classes")
        assert query == "(class_definition name: (identifier) @name)"

        query = get_ast_query("cpp", "find_functions")
        assert query == "(function_definition declarator: (function_declarator) @func)"

    def test_get_ast_query_with_alias(self):
        """Test getting AST query with language alias."""
        query = get_ast_query("c++", "find_classes")
        assert query == "(class_declaration name: (type_identifier) @name)"

        query = get_ast_query("py", "find_functions")
        assert query == "(function_definition name: (identifier) @name)"

    def test_get_ast_query_invalid_language(self):
        """Test getting AST query for invalid language."""
        query = get_ast_query("unknown_language", "find_classes")
        assert query is None

    def test_get_ast_query_invalid_name(self):
        """Test getting AST query with invalid name."""
        query = get_ast_query("python", "invalid_query")
        assert query is None

    def test_iter_ast_queries_valid_language(self):
        """Test iterating AST queries for valid language."""
        queries = list(iter_ast_queries("python"))
        assert len(queries) > 0

        # Check that we get tuples of (name, query)
        for name, query in queries:
            assert isinstance(name, str)
            assert isinstance(query, str)
            assert name.startswith("find_")

    def test_iter_ast_queries_with_alias(self):
        """Test iterating AST queries with language alias."""
        queries = list(iter_ast_queries("c++"))
        assert len(queries) > 0

        # Should get C++ queries
        names = [name for name, _ in queries]
        assert "find_classes" in names
        assert "find_templates" in names

    def test_iter_ast_queries_invalid_language(self):
        """Test iterating AST queries for invalid language."""
        queries = list(iter_ast_queries("unknown_language"))
        assert queries == []


class TestSummaryQueries:
    """Test summary query functions."""

    def test_get_summary_queries_valid_language(self):
        """Test getting summary queries for valid language."""
        queries = get_summary_queries("python")
        assert isinstance(queries, dict)
        assert "classes" in queries
        assert "functions" in queries
        assert queries["classes"] == "(class_definition name: (identifier) @name)"

    def test_get_summary_queries_with_alias(self):
        """Test getting summary queries with language alias."""
        queries = get_summary_queries("py")
        assert isinstance(queries, dict)
        assert "classes" in queries
        assert "functions" in queries

    def test_get_summary_queries_invalid_language(self):
        """Test getting summary queries for invalid language."""
        queries = get_summary_queries("unknown_language")
        assert queries == {}


class TestQueryBuilders:
    """Test query builder helper functions."""

    def test_build_capture_query_default(self):
        """Test building capture query with default capture."""
        query = build_capture_query("function_definition")
        assert query == "(function_definition) @node"

    def test_build_capture_query_custom(self):
        """Test building capture query with custom capture."""
        query = build_capture_query("class_definition", "class")
        assert query == "(class_definition) @class"

    def test_build_field_capture_query_default(self):
        """Test building field capture query with default capture."""
        query = build_field_capture_query("function_definition", "name")
        assert query == "(function_definition name: (_) @node)"

    def test_build_field_capture_query_custom(self):
        """Test building field capture query with custom capture."""
        query = build_field_capture_query("class_definition", "name", "class_name")
        assert query == "(class_definition name: (_) @class_name)"


class TestSCMFileOperations:
    """Test .scm file operations."""

    def test_get_scm_file_path_existing(self):
        """Test getting path for existing .scm file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock queries directory
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            # Create a test .scm file
            scm_file = queries_dir / "python-tags.scm"
            scm_file.write_text("test query")

            # Mock the QUERIES_DIR
            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                path = get_scm_file_path("python")
                assert path == scm_file
                assert path.exists()

    def test_get_scm_file_path_with_alias(self):
        """Test getting path with language alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            # Create file with canonical name
            scm_file = queries_dir / "cpp-tags.scm"
            scm_file.write_text("test query")

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                # Should find cpp-tags.scm when asking for c++
                path = get_scm_file_path("c++")
                assert path == scm_file

    def test_get_scm_file_path_alternate_names(self):
        """Test finding .scm file with alternate naming patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            # Test alternate name pattern: {lang}.scm
            scm_file = queries_dir / "python.scm"
            scm_file.write_text("test query")

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                path = get_scm_file_path("python")
                assert path == scm_file

    def test_get_scm_file_path_not_found(self):
        """Test getting path when .scm file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                path = get_scm_file_path("nonexistent")
                assert path is None

    def test_load_query_from_file_existing(self):
        """Test loading query from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            # Create a test .scm file with query content
            scm_file = queries_dir / "python-tags.scm"
            test_query = "(function_definition) @function"
            scm_file.write_text(test_query)

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                content = load_query_from_file("python")
                assert content == test_query

    def test_load_query_from_file_not_found(self):
        """Test loading query when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                content = load_query_from_file("nonexistent")
                assert content is None

    def test_load_query_from_file_with_alias(self):
        """Test loading query using language alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queries_dir = Path(tmpdir) / "queries"
            queries_dir.mkdir()

            # Create file with canonical name
            scm_file = queries_dir / "cpp-tags.scm"
            test_query = "(class_declaration) @class"
            scm_file.write_text(test_query)

            with patch("ai_dev_agent.core.tree_sitter.queries.QUERIES_DIR", queries_dir):
                # Should load cpp-tags.scm when asking for c++
                content = load_query_from_file("c++")
                assert content == test_query


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_all_languages_have_queries(self):
        """Test that all languages in AST templates are also in summary templates."""
        ast_languages = set(AST_QUERY_TEMPLATES.keys())
        summary_languages = set(SUMMARY_QUERY_TEMPLATES.keys())

        # These should have significant overlap
        common_languages = ast_languages & summary_languages
        assert len(common_languages) >= 8  # At least 8 common languages

    def test_query_format_validity(self):
        """Test that all queries follow expected format."""
        for lang, queries in AST_QUERY_TEMPLATES.items():
            for name, query in queries.items():
                # All queries should start with parenthesis
                assert query.startswith("("), f"Invalid query format for {lang}.{name}"
                # All queries should contain @ for captures
                assert "@" in query, f"No capture in query for {lang}.{name}"

    def test_aliases_map_to_valid_languages(self):
        """Test that all aliases map to languages with queries."""
        for alias, canonical in LANGUAGE_ALIASES.items():
            # At least one of the template dicts should have the canonical language
            has_ast = canonical in AST_QUERY_TEMPLATES
            has_summary = canonical in SUMMARY_QUERY_TEMPLATES
            assert has_ast or has_summary, f"Alias {alias} maps to {canonical} which has no queries"
