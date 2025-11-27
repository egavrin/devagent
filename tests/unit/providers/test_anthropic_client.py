"""Tests for the Anthropic client implementation."""

from __future__ import annotations

import json

from ai_dev_agent.providers.llm.anthropic import AnthropicClient
from ai_dev_agent.providers.llm.base import Message


def test_anthropic_complete_converts_response(monkeypatch):
    client = AnthropicClient(api_key="k", model="claude-3-5-sonnet-20241022")

    captured_payload: dict = {}

    def fake_post(payload, extra_headers=None):
        captured_payload.update(payload)
        return {"content": [{"type": "text", "text": "Hello world"}]}

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.complete([Message(role="user", content="Hi")], temperature=0.2, max_tokens=42)

    assert "messages" in captured_payload
    assert captured_payload["max_tokens"] == 42
    assert result == "Hello world"


def test_anthropic_invoke_tools_transforms_schema(monkeypatch):
    client = AnthropicClient(api_key="k", model="claude-3-opus-20240229")

    captured_payload: dict = {}

    def fake_post(payload, extra_headers=None):
        captured_payload.update(payload)
        return {
            "content": [
                {"type": "text", "text": "Using tool"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "echo",
                    "input": {"value": "hi"},
                },
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools(
        [Message(role="user", content="Call tool")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                    "description": "Echo input",
                },
            }
        ],
    )

    assert captured_payload["tools"][0]["name"] == "echo"
    assert captured_payload["tools"][0]["input_schema"]["properties"]["value"]["type"] == "string"
    assert result.calls[0].name == "echo"
    assert result.calls[0].arguments == {"value": "hi"}
