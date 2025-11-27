"""Tests for the OpenAI client implementation."""

from ai_dev_agent.providers.llm.base import Message
from ai_dev_agent.providers.llm.openai import OpenAIClient


def test_openai_prepare_payload_and_complete(monkeypatch):
    client = OpenAIClient(api_key="k", model="gpt-4o")

    captured = {}

    def fake_post(payload, extra_headers=None):
        captured.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": "Done",
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)

    result = client.complete(
        [Message(role="user", content="hello")], temperature=0.2, max_tokens=16
    )

    assert captured["model"] == "gpt-4o"
    assert captured["messages"][0]["role"] == "user"
    assert captured["max_tokens"] == 16
    assert captured["temperature"] == 0.2
    assert result == "Done"
