"""Tests for the prompt loader."""

import shutil
import tempfile
from pathlib import Path

import pytest

from ai_dev_agent.prompts.loader import PromptLoader


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary prompts directory for testing."""
    temp_dir = tempfile.mkdtemp()
    prompts_dir = Path(temp_dir) / "prompts"
    prompts_dir.mkdir()

    # Create test prompt files
    agents_dir = prompts_dir / "agents"
    agents_dir.mkdir()
    (agents_dir / "test_agent.md").write_text("# Test Agent\nTask: {task}")

    system_dir = prompts_dir / "system"
    system_dir.mkdir()
    (system_dir / "base_context.md").write_text("System context: {workspace}")

    formats_dir = prompts_dir / "formats"
    formats_dir.mkdir()
    (formats_dir / "test_format.md").write_text("Format specification")

    yield prompts_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestPromptLoader:
    """Test the PromptLoader class."""

    def test_init_with_custom_dir(self, temp_prompts_dir):
        """Test initialization with custom directory."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        assert loader.prompts_dir == temp_prompts_dir.resolve()

    def test_init_creates_missing_dir(self):
        """Explicit directories without prompts should raise immediately."""
        non_existent = Path("/tmp/test_prompts_" + str(id(self)))
        with pytest.raises(FileNotFoundError):
            PromptLoader(prompts_dir=non_existent)

    def test_load_prompt(self, temp_prompts_dir):
        """Test loading a prompt file."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.load_prompt("agents/test_agent.md")
        assert "# Test Agent" in content
        assert "Task: {task}" in content

    def test_load_prompt_without_extension(self, temp_prompts_dir):
        """Test loading prompt without .md extension."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.load_prompt("agents/test_agent")
        assert "# Test Agent" in content

    def test_load_prompt_not_found(self, temp_prompts_dir):
        """Test loading non-existent prompt raises error."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_prompt("agents/nonexistent.md")

    def test_render_prompt(self, temp_prompts_dir):
        """Test rendering prompt with context."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.render_prompt(
            "agents/test_agent.md", context={"task": "implement feature X"}
        )
        assert "# Test Agent" in content
        assert "Task: implement feature X" in content
        assert "{task}" not in content

    def test_render_prompt_no_context(self, temp_prompts_dir):
        """Test rendering without context returns template as-is."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.render_prompt("agents/test_agent.md")
        assert "Task: {task}" in content

    def test_load_agent_prompt(self, temp_prompts_dir):
        """Test loading agent-specific prompt."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.load_agent_prompt("test_agent", context={"task": "write tests"})
        assert "Task: write tests" in content

    def test_load_system_prompt(self, temp_prompts_dir):
        """Test loading system prompt."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.load_system_prompt(
            "base_context", context={"workspace": "/home/user/project"}
        )
        assert "System context: /home/user/project" in content

    def test_load_format_prompt(self, temp_prompts_dir):
        """Test loading format specification."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        content = loader.load_format_prompt("test_format")
        assert "Format specification" in content

    def test_compose_prompt(self, temp_prompts_dir):
        """Test composing multiple prompts."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        composed = loader.compose_prompt(
            ["system/base_context.md", ("agents/test_agent.md", {"task": "test task"})]
        )
        assert "System context: {workspace}" in composed
        assert "Task: test task" in composed
        assert "---" in composed  # Separator

    def test_list_prompts(self, temp_prompts_dir):
        """Test listing available prompts."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        all_prompts = loader.list_prompts()
        assert "agents/test_agent.md" in all_prompts
        assert "system/base_context.md" in all_prompts
        assert "formats/test_format.md" in all_prompts

    def test_list_prompts_by_category(self, temp_prompts_dir):
        """Test listing prompts filtered by category."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)
        agent_prompts = loader.list_prompts(category="agents")
        assert "agents/test_agent.md" in agent_prompts
        assert "system/base_context.md" not in agent_prompts

    def test_caching(self, temp_prompts_dir):
        """Test that prompts are cached."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)

        # Load prompt twice
        content1 = loader.load_prompt("agents/test_agent.md")
        content2 = loader.load_prompt("agents/test_agent.md")

        # Should return same content
        assert content1 == content2

        # Check cache
        cached_path = str((temp_prompts_dir / "agents/test_agent.md").resolve())
        assert cached_path in loader._cache

    def test_clear_cache(self, temp_prompts_dir):
        """Test clearing the cache."""
        loader = PromptLoader(prompts_dir=temp_prompts_dir)

        # Load and cache a prompt
        loader.load_prompt("agents/test_agent.md")
        assert len(loader._cache) > 0

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0
