import pytest

from ai_dev_agent.providers.llm import create_client
from ai_dev_agent.providers.llm.anthropic import AnthropicClient
from ai_dev_agent.providers.llm.base import (
    LLMRateLimitError,
    LLMResponseError,
    Message,
    RetryConfig,
)
from ai_dev_agent.providers.llm.deepseek import DeepSeekClient
from ai_dev_agent.providers.llm.openai import OpenAIClient
from ai_dev_agent.providers.llm.openrouter import OpenRouterClient


def test_deepseek_backoff_jitter_range(monkeypatch):
    config = RetryConfig(
        initial_delay=1.0, max_delay=10.0, backoff_multiplier=2.0, jitter_ratio=0.25
    )
    client = DeepSeekClient(api_key="test", model="demo", retry_config=config)

    captured = {}

    def fake_uniform(low: float, high: float) -> float:
        captured["low"] = low
        captured["high"] = high
        return high

    monkeypatch.setattr("ai_dev_agent.providers.llm.deepseek.random.uniform", fake_uniform)

    delay = client._calculate_delay(3)

    assert delay == pytest.approx(5.0)
    assert captured["low"] == pytest.approx(3.0)
    assert captured["high"] == pytest.approx(5.0)


def test_deepseek_error_mapping():
    client = DeepSeekClient(api_key="test", model="demo")

    rate_error = client._error_from_status(429, "Too Many Requests")
    server_error = client._error_from_status(500, "Server error")

    assert isinstance(rate_error, LLMRateLimitError)
    assert isinstance(server_error, LLMResponseError)


def test_deepseek_tool_call_parsing(monkeypatch):
    client = DeepSeekClient(api_key="test", model="demo")

    payload_captured = {}

    def fake_post(payload, extra_headers=None):
        payload_captured["tools"] = payload.get("tools")
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "find",
                                    "arguments": '{\n  "query": "*.py"\n}',
                                },
                            }
                        ],
                        "content": "Searching for *.py",
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools(
        [Message(role="user", content="search for python files")],
        tools=[
            {"type": "function", "function": {"name": "find", "parameters": {"type": "object"}}}
        ],
        temperature=0.0,
    )

    assert payload_captured["tools"][0]["function"]["name"] == "find"
    assert result.message_content == "Searching for *.py"
    assert len(result.calls) == 1
    assert result.calls[0].name == "find"
    assert result.calls[0].arguments == {"query": "*.py"}


def test_openrouter_tool_call_parsing_with_provider(monkeypatch):
    client = OpenRouterClient(
        api_key="test",
        model="demo",
        provider_only=("Cerebras",),
        provider_config={"allow": ["Cerebras"]},
        default_headers={"HTTP-Referer": "https://example.com"},
    )

    captured: dict = {}

    def fake_post(payload, extra_headers=None):
        captured["provider"] = payload.get("provider")
        captured["headers"] = client._build_headers(extra_headers)  # type: ignore[attr-defined]
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"value": "hello"}',
                                },
                            }
                        ],
                        "content": None,
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.invoke_tools(
        [Message(role="user", content="say hi")],
        tools=[
            {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}
        ],
    )

    assert captured["provider"] == {"allow": ["Cerebras"], "only": ["Cerebras"]}
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert "HTTP-Referer" in captured["headers"]
    assert len(result.calls) == 1
    assert result.calls[0].arguments == {"value": "hello"}


def test_create_client_openrouter_aliases():
    openrouter_client = create_client(
        provider="openrouter",
        api_key="k",
        model="m",
    )
    assert isinstance(openrouter_client, OpenRouterClient)

    cerebras_client = create_client(
        provider="cerebras",
        api_key="k",
        model="m",
    )
    assert isinstance(cerebras_client, OpenRouterClient)
    payload = cerebras_client._prepare_payload([Message(role="user", content="hi")], 0.0, None)
    assert payload["provider"]["only"] == ["Cerebras"]


def test_create_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_client(provider="unknown", api_key="k", model="m")


def test_create_client_openai_and_anthropic():
    openai_client = create_client(provider="openai", api_key="k", model="gpt-4o")
    assert isinstance(openai_client, OpenAIClient)
    assert openai_client.base_url.endswith("api.openai.com/v1")

    anthropic_client = create_client(
        provider="anthropic", api_key="k", model="claude-3-5-sonnet-20241022"
    )
    assert isinstance(anthropic_client, AnthropicClient)
    assert anthropic_client.base_url.endswith("api.anthropic.com")
