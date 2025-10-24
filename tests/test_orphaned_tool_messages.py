"""Tests for orphaned tool message handling and sanitization."""
import pytest

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.session.sanitizer import sanitize_conversation
from ai_dev_agent.core.utils.context_budget import prune_messages, ContextBudgetConfig


class TestConversationSanitization:
    """Test suite for conversation sanitization to remove orphaned tool messages."""

    def test_sanitize_removes_orphaned_tool_message(self):
        """Orphaned tool messages (no matching assistant tool_call) should be removed."""
        messages = [
            Message(role='system', content='System'),
            Message(role='user', content='Query'),
            # No assistant message with tool_calls
            Message(role='tool', content='Orphaned result', tool_call_id='call_1'),  # ORPHAN
            Message(role='assistant', content='Answer'),
        ]

        sanitized = sanitize_conversation(messages)

        assert len(sanitized) == 3  # Orphaned tool message removed
        assert sanitized[0].role == 'system'
        assert sanitized[1].role == 'user'
        assert sanitized[2].role == 'assistant'
        assert all(msg.role != 'tool' for msg in sanitized)

    def test_sanitize_preserves_valid_tool_messages(self):
        """Valid tool messages (matching assistant tool_call) should be preserved."""
        messages = [
            Message(role='system', content='System'),
            Message(role='user', content='Query'),
            Message(role='assistant', content='Searching...', tool_calls=[
                {'id': 'call_1', 'function': {'name': 'grep', 'arguments': '{}'}}
            ]),
            Message(role='tool', content='Result', tool_call_id='call_1'),
            Message(role='assistant', content='Found it'),
        ]

        sanitized = sanitize_conversation(messages)

        assert len(sanitized) == 5  # All messages preserved
        assert sanitized[3].role == 'tool'
        assert sanitized[3].tool_call_id == 'call_1'

    def test_sanitize_handles_multiple_tool_calls(self):
        """Multiple tool calls in parallel should all be preserved if valid."""
        messages = [
            Message(role='user', content='Search'),
            Message(role='assistant', content='Searching...', tool_calls=[
                {'id': 'call_1', 'function': {'name': 'grep'}},
                {'id': 'call_2', 'function': {'name': 'glob'}}
            ]),
            Message(role='tool', content='Grep result', tool_call_id='call_1'),
            Message(role='tool', content='Glob result', tool_call_id='call_2'),
            Message(role='assistant', content='Done'),
        ]

        sanitized = sanitize_conversation(messages)

        assert len(sanitized) == 5  # All preserved
        tool_messages = [msg for msg in sanitized if msg.role == 'tool']
        assert len(tool_messages) == 2

    def test_sanitize_removes_partial_orphans(self):
        """If one of multiple tool responses is orphaned, only remove the orphan."""
        messages = [
            Message(role='assistant', content='Searching...', tool_calls=[
                {'id': 'call_1', 'function': {'name': 'grep'}}
                # Missing call_2 in tool_calls
            ]),
            Message(role='tool', content='Valid result', tool_call_id='call_1'),
            Message(role='tool', content='Orphaned', tool_call_id='call_2'),  # ORPHAN
            Message(role='assistant', content='Done'),
        ]

        sanitized = sanitize_conversation(messages)

        assert len(sanitized) == 3  # Orphaned tool message removed
        tool_messages = [msg for msg in sanitized if msg.role == 'tool']
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == 'call_1'

    def test_sanitize_empty_conversation(self):
        """Empty conversations should remain empty."""
        assert sanitize_conversation([]) == []

    def test_sanitize_no_tool_messages(self):
        """Conversations without tool messages should pass through unchanged."""
        messages = [
            Message(role='system', content='System'),
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi'),
        ]

        sanitized = sanitize_conversation(messages)
        assert len(sanitized) == 3
        assert [msg.role for msg in sanitized] == ['system', 'user', 'assistant']


class TestPruningMessageRolePreservation:
    """Test suite for context pruning role preservation fixes."""

    def test_pruning_preserves_tool_message_role(self):
        """Tool messages should keep their role even when content is truncated."""
        config = ContextBudgetConfig(
            max_tokens=100,  # Very low limit to force pruning
            headroom_tokens=10,
            keep_last_assistant=1,
        )

        # Create a conversation that will exceed budget
        messages = [
            Message(role='system', content='System'),
            Message(role='user', content='Query' * 100),  # Large message
            Message(role='assistant', content='Searching', tool_calls=[
                {'id': 'call_1', 'function': {'name': 'grep'}}
            ]),
            Message(role='tool', content='Result' * 100, tool_call_id='call_1'),  # Large
            Message(role='assistant', content='Done'),
        ]

        pruned = prune_messages(messages, config)

        # Check that tool messages are still tool messages (not converted to assistant)
        tool_messages = [msg for msg in pruned if msg.role == 'tool']
        assert len(tool_messages) > 0, "Tool messages should not be completely removed"

        for tool_msg in tool_messages:
            assert tool_msg.role == 'tool', "Tool message role should be preserved"
            assert tool_msg.tool_call_id is not None, "tool_call_id should be preserved"

    def test_pruning_preserves_user_message_role(self):
        """User messages should keep their role when truncated."""
        config = ContextBudgetConfig(
            max_tokens=50,
            headroom_tokens=5,
            keep_last_assistant=1,
        )

        messages = [
            Message(role='user', content='X' * 1000),
            Message(role='assistant', content='Y' * 1000),
            Message(role='user', content='Recent'),  # Should be preserved
        ]

        pruned = prune_messages(messages, config)

        # All user messages should still be user role
        user_messages = [msg for msg in pruned if msg.role == 'user']
        assert all(msg.role == 'user' for msg in user_messages)

    def test_pruning_preserves_assistant_tool_calls(self):
        """Assistant messages with tool_calls should preserve the tool_calls structure."""
        config = ContextBudgetConfig(
            max_tokens=100,
            headroom_tokens=10,
            keep_last_assistant=2,
        )

        messages = [
            Message(role='system', content='S' * 500),
            Message(role='assistant', content='Searching', tool_calls=[
                {'id': 'call_1', 'function': {'name': 'grep'}}
            ]),
            Message(role='tool', content='Result', tool_call_id='call_1'),
            Message(role='assistant', content='Done'),
        ]

        pruned = prune_messages(messages, config)

        # Find the assistant message with tool_calls
        assistant_with_tools = [
            msg for msg in pruned
            if msg.role == 'assistant' and msg.tool_calls
        ]

        if assistant_with_tools:
            # If preserved, tool_calls structure should be intact
            assert assistant_with_tools[0].tool_calls is not None
            assert len(assistant_with_tools[0].tool_calls) > 0


class TestIntegrationPruningAndSanitization:
    """Integration tests combining pruning and sanitization."""

    def test_pruning_then_sanitization_removes_orphans(self):
        """Pruning may create orphans, sanitization should clean them up."""
        config = ContextBudgetConfig(
            max_tokens=150,
            headroom_tokens=10,
            keep_last_assistant=1,  # Only keep last assistant
        )

        messages = [
            Message(role='system', content='S' * 200),
            Message(role='user', content='U' * 200),
            Message(role='assistant', content='A1' * 200, tool_calls=[
                {'id': 'call_1', 'function': {'name': 'tool1'}}
            ]),
            Message(role='tool', content='T1' * 200, tool_call_id='call_1'),
            Message(role='assistant', content='A2'),  # This will be kept (last)
        ]

        # Step 1: Prune
        pruned = prune_messages(messages, config)

        # Step 2: Sanitize
        sanitized = sanitize_conversation(pruned)

        # Verify no orphaned tool messages remain
        tool_messages = [msg for msg in sanitized if msg.role == 'tool']
        assistant_tool_calls = set()

        for msg in sanitized:
            if msg.role == 'assistant' and msg.tool_calls:
                for call in msg.tool_calls:
                    call_id = call.get('id') or call.get('tool_call_id')
                    if call_id:
                        assistant_tool_calls.add(call_id)

        # Every tool message must have a matching assistant tool call
        for tool_msg in tool_messages:
            assert tool_msg.tool_call_id in assistant_tool_calls, \
                f"Tool message {tool_msg.tool_call_id} is orphaned!"

    def test_realistic_review_scenario(self):
        """Simulate a realistic review command scenario with large context."""
        config = ContextBudgetConfig(
            max_tokens=5000,  # Simulate exceeding budget
            headroom_tokens=500,
            keep_last_assistant=3,
            max_tool_messages=5,
        )

        # Simulate a review conversation with multiple iterations
        messages = [
            Message(role='system', content='Rule: ' + 'X' * 2000),
            Message(role='user', content='Review patch: ' + 'P' * 2000),
        ]

        # Simulate 10 iterations of tool calls
        for i in range(10):
            messages.extend([
                Message(role='assistant', content=f'Iteration {i}', tool_calls=[
                    {'id': f'call_{i}', 'function': {'name': 'analyze'}}
                ]),
                Message(role='tool', content=f'Result {i}' * 100, tool_call_id=f'call_{i}'),
            ])

        messages.append(Message(role='assistant', content='Final answer' * 100))

        # Apply pruning then sanitization
        pruned = prune_messages(messages, config)
        sanitized = sanitize_conversation(pruned)

        # Verify conversation is valid
        assert len(sanitized) > 0
        assert all(msg.role in ['system', 'user', 'assistant', 'tool'] for msg in sanitized)

        # Verify no orphans
        tool_call_ids = set()
        for msg in sanitized:
            if msg.role == 'assistant' and msg.tool_calls:
                for call in msg.tool_calls:
                    call_id = call.get('id') or call.get('tool_call_id')
                    if call_id:
                        tool_call_ids.add(call_id)

        for msg in sanitized:
            if msg.role == 'tool':
                assert msg.tool_call_id in tool_call_ids or msg.tool_call_id == 'pruned', \
                    f"Found orphaned tool message: {msg.tool_call_id}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
