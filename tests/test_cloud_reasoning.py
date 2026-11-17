"""Cloud providers and ``reasoning_effort`` on tool requests.

Uses the real OpenAI SDK with only the HTTP layer replaced, replaying the error
Azure returns for a reasoning model asked to use tools on /chat/completions.
"""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from bot.providers._cloud import CloudProvider
from bot.providers.base import ProviderError

AZURE_REFUSAL = {
    "error": {
        "message": (
            "Function tools with reasoning_effort are not supported for gpt-x in "
            "/v1/chat/completions. To use function tools, use /v1/responses or set "
            "reasoning_effort to 'none'."
        ),
        "type": "invalid_request_error",
        "param": "reasoning_effort",
        "code": None,
    }
}
OTHER_400 = {"error": {"message": "Invalid messages.", "type": "invalid_request_error"}}

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
STREAM = (
    b'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)

TOOL = {
    "type": "function",
    "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
}
MESSAGES = [{"role": "user", "content": "hi"}]


def make_provider(
    *responses: tuple[int, Any], effort: str | None = None
) -> tuple[CloudProvider, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    remaining = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        status, body = remaining.pop(0)
        if isinstance(body, bytes):
            return httpx2.Response(
                status, headers={"content-type": "text/event-stream"}, content=body
            )
        return httpx2.Response(status, json=body)

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    provider = CloudProvider(
        client, name="test", default_model="test-model", tool_reasoning_effort=effort
    )
    return provider, requests


async def test_without_setting_nothing_is_sent() -> None:
    provider, requests = make_provider((200, COMPLETION), (200, COMPLETION))

    await provider.complete(MESSAGES, tools=[TOOL])
    await provider.complete(MESSAGES)

    assert all("reasoning_effort" not in r for r in requests)


async def test_configured_effort_goes_only_on_tool_requests() -> None:
    provider, requests = make_provider((200, COMPLETION), (200, COMPLETION), effort="low")

    await provider.complete(MESSAGES, tools=[TOOL])
    await provider.complete(MESSAGES)

    assert requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in requests[1]


async def test_refusal_retries_with_none_and_remembers() -> None:
    provider, requests = make_provider(
        (400, AZURE_REFUSAL), (200, COMPLETION), (200, COMPLETION), (200, COMPLETION)
    )

    response = await provider.complete(MESSAGES, tools=[TOOL])
    await provider.complete(MESSAGES, tools=[TOOL])
    await provider.complete(MESSAGES)

    assert response.content == "ok"
    assert "reasoning_effort" not in requests[0]
    assert requests[1]["reasoning_effort"] == "none"
    # Later tool requests send "none" up front; plain requests keep the default.
    assert requests[2]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in requests[3]
    assert len(requests) == 4


async def test_streaming_refusal_retries_with_none() -> None:
    provider, requests = make_provider((400, AZURE_REFUSAL), (200, STREAM))

    chunks = [c async for c in provider.stream(MESSAGES, tools=[TOOL])]

    assert "".join(c.delta for c in chunks) == "ok"
    assert requests[1]["reasoning_effort"] == "none"


async def test_refusal_with_none_already_sent_is_not_retried() -> None:
    provider, requests = make_provider((400, AZURE_REFUSAL), effort="none")

    with pytest.raises(ProviderError):
        await provider.complete(MESSAGES, tools=[TOOL])
    assert len(requests) == 1


async def test_unrelated_errors_are_not_retried() -> None:
    provider, requests = make_provider((400, OTHER_400))

    with pytest.raises(ProviderError):
        await provider.complete(MESSAGES, tools=[TOOL])
    assert len(requests) == 1
    assert provider.tool_reasoning_effort is None


async def test_refusal_without_tools_is_not_retried() -> None:
    provider, requests = make_provider((400, AZURE_REFUSAL))

    with pytest.raises(ProviderError):
        await provider.complete(MESSAGES)
    assert len(requests) == 1
