"""Tests to improve coverage for simple utility modules."""

import logging

from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.logger import (
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


class TestLogger:
    """Test logger utilities."""

    def test_get_logger_basic(self):
        """Test getting a basic logger."""
        logger = get_logger(__name__)
        assert logger.name == __name__
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_instance(self):
        """Test that same logger name returns same instance."""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        assert logger1 is logger2

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        configure_logging(level="INFO")  # Pass as string, not int
        get_logger("config_test")
        # Logger should inherit the root configuration
        assert logging.getLogger().level == logging.INFO

    def test_configure_logging_with_structured(self):
        """Test logging configuration with structured output."""
        configure_logging(level="DEBUG", structured=True)
        # Verify configuration was applied
        assert logging.getLogger().level == logging.DEBUG
        # Check that structured formatter is used
        handler = logging.getLogger().handlers[0]
        from ai_dev_agent.core.utils.logger import StructuredFormatter

        assert isinstance(handler.formatter, StructuredFormatter)

    def test_correlation_id(self):
        """Test correlation ID functionality."""
        # Save original correlation ID
        original_id = get_correlation_id()

        try:
            # Reset to default first
            set_correlation_id(None)
            assert get_correlation_id() == "-"

            # Set and get correlation ID
            set_correlation_id("test-123")
            assert get_correlation_id() == "test-123"

            # Reset correlation ID
            set_correlation_id(None)
            assert get_correlation_id() == "-"
        finally:
            # Restore original correlation ID
            set_correlation_id(original_id if original_id != "-" else None)

    def test_get_logger_default_name(self):
        """Test getting logger with default name."""
        logger = get_logger(None)
        assert logger.name == "ai_dev_agent"

        # Also test with no arguments (uses default)
        logger2 = get_logger()
        assert logger2.name == "ai_dev_agent"


class TestKeywords:
    """Test keyword extraction functionality."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        text = "Create a test for the authentication module"
        keywords = extract_keywords(text)
        assert isinstance(keywords, list)
        assert "authentication" in keywords  # "test" and "module" are special terms

    def test_extract_keywords_with_identifiers(self):
        """Test extracting programming identifiers."""
        text = "Fix the getUserById function in UserService class"
        keywords = extract_keywords(text, include_special_terms=True)
        assert "getuserbyid" in keywords  # Extracted as lowercase
        assert "userservice" in keywords
        # function and class are special terms
        assert "function" in keywords or "class" in keywords

    def test_extract_keywords_special_terms(self):
        """Test extraction of special programming terms."""
        text = "Build a REST API with GraphQL support and JSON responses"
        keywords = extract_keywords(text, include_special_terms=True)
        # Keywords are returned lowercase
        assert (
            "rest" in keywords or "api" in keywords or "graphql" in keywords or "json" in keywords
        )

    def test_extract_keywords_with_limit(self):
        """Test keyword extraction with limit."""
        text = "one two three four five six seven eight nine ten eleven twelve"
        keywords = extract_keywords(text, limit=5)
        assert len(keywords) <= 5

    def test_extract_keywords_stopwords(self):
        """Test that stopwords are filtered."""
        text = "the quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text)
        # "the", "over" are stopwords
        assert "the" not in keywords
        assert "quick" in keywords or "brown" in keywords or "lazy" in keywords

    def test_extract_keywords_custom_stopwords(self):
        """Test with custom stopwords."""
        text = "test the quick authentication system"
        keywords = extract_keywords(text, extra_stopwords=["quick", "system"])
        assert "quick" not in keywords
        assert "system" not in keywords
        assert "authentication" in keywords

    def test_extract_keywords_empty_string(self):
        """Test extraction from empty string."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_extract_keywords_none_safe(self):
        """Test extraction handles None-like input safely."""
        keywords = extract_keywords("")
        assert keywords == []


class TestSimpleUtilities:
    """Test other simple utilities."""

    def test_constants_module(self):
        """Test that constants module has expected values."""
        from ai_dev_agent.core.utils.constants import (
            DEFAULT_IGNORED_REPO_DIRS,
            DEFAULT_MAX_CONTEXT_TOKENS,
            DEFAULT_RESPONSE_HEADROOM,
            MAX_HISTORY_ENTRIES,
        )

        # Verify constants exist and have reasonable values
        assert isinstance(DEFAULT_IGNORED_REPO_DIRS, frozenset)
        assert ".git" in DEFAULT_IGNORED_REPO_DIRS
        assert isinstance(MAX_HISTORY_ENTRIES, int)
        assert MAX_HISTORY_ENTRIES > 0
        assert isinstance(DEFAULT_MAX_CONTEXT_TOKENS, int)
        assert DEFAULT_MAX_CONTEXT_TOKENS > 0
        assert isinstance(DEFAULT_RESPONSE_HEADROOM, int)

    def test_approval_policy(self):
        """Test approval policy module."""
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        # Test that ApprovalPolicy is a dataclass with expected fields
        policy = ApprovalPolicy()
        assert hasattr(policy, "auto_approve_plan")
        assert hasattr(policy, "auto_approve_code")
        assert hasattr(policy, "auto_approve_shell")
        assert hasattr(policy, "auto_approve_adr")
        assert hasattr(policy, "emergency_override")
        assert hasattr(policy, "audit_file")

        # Test default values
        assert not policy.auto_approve_plan
        assert not policy.auto_approve_code
        assert not policy.emergency_override

    def test_approval_manager(self):
        """Test approval manager functionality."""
        from ai_dev_agent.core.approval.approvals import ApprovalManager
        from ai_dev_agent.core.approval.policy import ApprovalPolicy

        # Create policy with some auto-approvals
        policy = ApprovalPolicy(
            auto_approve_plan=True, auto_approve_code=False, auto_approve_shell=False
        )
        manager = ApprovalManager(policy)

        # Test auto-approval based on policy
        assert manager.maybe_auto("plan")
        assert not manager.maybe_auto("code")
        assert not manager.maybe_auto("shell")
        assert not manager.maybe_auto("unknown")

        # Test with emergency override
        emergency_policy = ApprovalPolicy(emergency_override=True)
        emergency_manager = ApprovalManager(emergency_policy)
        assert emergency_manager.maybe_auto("anything")
