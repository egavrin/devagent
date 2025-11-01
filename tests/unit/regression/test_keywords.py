"""Comprehensive tests for the keywords utility module."""

from ai_dev_agent.core.utils.keywords import extract_keywords


class TestExtractKeywords:
    """Test suite for extract_keywords function."""

    def test_empty_text(self):
        """Test with empty text."""
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_basic_extraction(self):
        """Test basic keyword extraction."""
        text = "implement the user authentication system for the application"
        keywords = extract_keywords(text)

        # Should filter out stopwords and short words
        assert "implement" not in keywords  # stopword
        assert "the" not in keywords  # stopword
        assert "for" not in keywords  # stopword
        assert "user" in keywords
        assert "authentication" in keywords
        assert "system" in keywords
        assert "application" in keywords

    def test_limit_parameter(self):
        """Test the limit parameter works correctly."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

        # Test different limits
        keywords_3 = extract_keywords(text, limit=3)
        assert len(keywords_3) == 3
        assert keywords_3 == ["alpha", "beta", "gamma"]

        keywords_5 = extract_keywords(text, limit=5)
        assert len(keywords_5) == 5

        keywords_20 = extract_keywords(text, limit=20)
        assert len(keywords_20) <= 20

    def test_duplicate_removal(self):
        """Test that duplicates are removed."""
        text = "test test testing tested tester test"
        keywords = extract_keywords(text)

        # Should have unique keywords only
        assert "test" in keywords or "testing" in keywords
        assert len(keywords) == len(set(keywords))  # No duplicates

    def test_case_insensitive_deduplication(self):
        """Test case-insensitive deduplication."""
        text = "Python python PYTHON PyThOn"
        keywords = extract_keywords(text)

        # Should only have one version (lowercase)
        assert len(keywords) == 1
        assert keywords[0] == "python"

    def test_filters_short_words(self):
        """Test that words with 2 or fewer characters are filtered."""
        text = "a bb ccc dd eee ff ggg hi ij"
        keywords = extract_keywords(text)

        # Should filter out words with <= 2 characters
        assert "a" not in keywords
        assert "bb" not in keywords
        assert "hi" not in keywords
        assert "ij" not in keywords
        assert "ccc" in keywords
        assert "eee" in keywords
        assert "ggg" in keywords

    def test_include_special_terms(self):
        """Test including special terms when enabled."""
        text = "the simple text with test and api endpoint"

        # Without special terms
        keywords_normal = extract_keywords(text, include_special_terms=False)
        assert "test" not in keywords_normal  # Common word
        assert "api" not in keywords_normal
        assert "simple" in keywords_normal
        assert "text" in keywords_normal
        assert "endpoint" in keywords_normal

        # With special terms
        keywords_special = extract_keywords(text, include_special_terms=True, limit=10)
        # Special terms should be added after normal keywords
        assert "test" in keywords_special or "api" in keywords_special

    def test_special_terms_pattern_matching(self):
        """Test that special terms are correctly identified."""
        text = "pytest unittest graphql database json http rest api testing"
        keywords = extract_keywords(text, include_special_terms=True, limit=20)

        # All these should be recognized as special terms
        special_terms = [
            "pytest",
            "unittest",
            "graphql",
            "database",
            "json",
            "http",
            "rest",
            "api",
            "testing",
        ]
        found_specials = [k for k in keywords if k in special_terms]
        assert len(found_specials) > 0

    def test_extra_stopwords(self):
        """Test with additional stopwords."""
        text = "python javascript typescript golang rust programming languages"

        # Without extra stopwords
        keywords_normal = extract_keywords(text)
        assert "python" in keywords_normal
        assert "javascript" in keywords_normal

        # With extra stopwords
        extra_stops = ["python", "javascript"]
        keywords_filtered = extract_keywords(text, extra_stopwords=extra_stops)
        assert "python" not in keywords_filtered
        assert "javascript" not in keywords_filtered
        assert "typescript" in keywords_filtered
        assert "golang" in keywords_filtered

    def test_identifier_pattern(self):
        """Test that the identifier pattern works correctly."""
        text = (
            "snake_case camelCase PascalCase kebab-case dot.case 123numbers _underscore __dunder__"
        )
        keywords = extract_keywords(text, limit=20)

        # Valid identifiers
        assert "snake_case" in keywords
        assert "camelcase" in keywords  # lowercase version
        assert "pascalcase" in keywords  # lowercase version
        assert "_underscore" in keywords or "underscore" in keywords
        assert "__dunder__" in keywords or "dunder" in keywords

        # Invalid identifiers (with special chars)
        # Note: kebab-case will be split, dot.case will be split
        # 123numbers won't match as it starts with numbers

    def test_preserves_order(self):
        """Test that keywords are returned in order of appearance."""
        text = "first second third fourth fifth"
        keywords = extract_keywords(text)

        assert keywords == ["first", "second", "third", "fourth", "fifth"]

    def test_mixed_content(self):
        """Test with realistic mixed content."""
        text = """
        Fix the authentication bug in the user login function.
        The issue occurs when users try to reset their password.
        We need to update the validation logic and add better error handling.
        """

        keywords = extract_keywords(text, limit=10)

        # Should extract meaningful terms
        assert "authentication" in keywords
        assert "bug" in keywords
        assert "user" in keywords
        assert "login" in keywords
        assert "issue" in keywords or "occurs" in keywords
        assert "password" in keywords or "reset" in keywords

        # Should not include stopwords
        assert "the" not in keywords
        assert "in" not in keywords
        assert "to" not in keywords

    def test_code_related_text(self):
        """Test with code-related text."""
        text = "refactor the DatabaseConnection class to use async await patterns"
        keywords = extract_keywords(text, include_special_terms=True)

        assert "refactor" in keywords
        assert "databaseconnection" in keywords
        assert "async" in keywords
        assert "await" in keywords
        assert "patterns" in keywords

    def test_numbers_and_special_chars(self):
        """Test handling of numbers and special characters."""
        text = "version 2.0.1 requires python3.9+ and node@16"
        keywords = extract_keywords(text)

        # Pure numbers are not identifiers
        # But alphanumeric identifiers should work
        assert "version" in keywords
        assert "requires" in keywords
        # "python3" might not match as identifier pattern looks for letter start
        assert "node" in keywords

    def test_very_long_text(self):
        """Test with very long text to ensure performance."""
        # Create a long text with many unique words
        words = [f"word{i}" for i in range(1000)]
        text = " ".join(words)

        keywords = extract_keywords(text, limit=10)

        # Should return exactly 10 keywords
        assert len(keywords) == 10
        # Should be the first 10 unique words
        assert keywords == [f"word{i}" for i in range(10)]

    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "implement über feature für python código"
        keywords = extract_keywords(text)

        # ASCII identifiers should be extracted
        assert "feature" in keywords
        assert "python" in keywords
        # Non-ASCII might not match the identifier pattern

    def test_all_stopwords(self):
        """Test text containing only stopwords."""
        text = "the and or but in on at to for of with by"
        keywords = extract_keywords(text)

        assert keywords == []

    def test_limit_exceeds_available_keywords(self):
        """Test when limit exceeds available keywords."""
        text = "alpha beta gamma"
        keywords = extract_keywords(text, limit=100)

        # Should return only available keywords
        assert len(keywords) == 3
        assert keywords == ["alpha", "beta", "gamma"]
