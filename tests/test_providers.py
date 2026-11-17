from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import pytest
from openai import OpenAIError

from bot.providers import build_cloud_provider, build_local_provider
from bot.providers._compat import OpenAICompatProvider
from bot.providers.azure import AzureOpenAIProvider
from bot.providers.base import ProviderError, ToolCall, Usage
from bot.providers.ollama import OllamaProvider
from bot.providers.openai import OpenAIProvider
from fakes import make_settings

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeCompletions:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_provider(result: Any) -> tuple[OpenAICompatProvider, FakeCompletions]:
    completions = FakeCompletions(result)
    client: Any = NS(chat=NS(completions=completions))
    provider = OpenAICompatProvider(client, name="test", default_model="default-model")
    return provider, completions


def usage(prompt: int, completion: int) -> NS:
    return NS(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


def event(
    content: str | None = None,
    tool_calls: list[NS] | None = None,
    finish_reason: str | None = None,
) -> NS:
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)], usage=None)


async def as_stream(events: list[NS]) -> AsyncIterator[NS]:
    for item in events:
        yield item


async def test_complete_maps_response() -> None:
    message = NS(
        content="hello",
        tool_calls=[NS(id="call_1", function=NS(name="web_search", arguments='{"q": "x"}'))],
    )
    response = NS(choices=[NS(message=message, finish_reason="tool_calls")], usage=usage(10, 5))
    provider, completions = make_provider(response)

    result = await provider.complete(MESSAGES, temperature=0, max_tokens=64)

    assert result.content == "hello"
    assert result.tool_calls == (ToolCall("call_1", "web_search", '{"q": "x"}'),)
    assert result.finish_reason == "tool_calls"
    assert result.usage == Usage(10, 5, 15)
    assert completions.calls == [
        {
            "model": "default-model",
            "messages": MESSAGES,
            "temperature": 0,
            "max_completion_tokens": 64,
        }
    ]


async def test_complete_passes_overrides_and_omits_unset_options() -> None:
    response = NS(
        choices=[NS(message=NS(content=None, tool_calls=None), finish_reason="stop")],
        usage=None,
    )
    provider, completions = make_provider(response)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    result = await provider.complete(
        MESSAGES, model="other", tools=tools, response_format={"type": "json_object"}
    )

    assert result.content == ""
    assert result.usage is None
    assert completions.calls == [
        {
            "model": "other",
            "messages": MESSAGES,
            "tools": tools,
            "response_format": {"type": "json_object"},
        }
    ]


async def test_complete_wraps_sdk_errors() -> None:
    provider, _ = make_provider(OpenAIError("boom"))
    with pytest.raises(ProviderError, match="test: boom"):
        await provider.complete(MESSAGES)


async def test_complete_rejects_empty_choices() -> None:
    provider, _ = make_provider(NS(choices=[], usage=None))
    with pytest.raises(ProviderError, match="no choices"):
        await provider.complete(MESSAGES)


async def test_stream_yields_deltas_then_final_chunk_with_usage() -> None:
    events = [
        event(content="Hel"),
        event(content="lo"),
        event(finish_reason="stop"),
        NS(choices=[], usage=usage(3, 2)),
    ]
    provider, completions = make_provider(as_stream(events))

    chunks = [chunk async for chunk in provider.stream(MESSAGES)]

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo", ""]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == Usage(3, 2, 5)
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["stream_options"] == {"include_usage": True}


async def test_stream_assembles_fragmented_tool_calls() -> None:
    def fragment(index: int, id: str | None, name: str | None, arguments: str) -> NS:
        return NS(index=index, id=id, function=NS(name=name, arguments=arguments))

    events = [
        event(tool_calls=[fragment(0, "call_a", "web_search", "")]),
        event(tool_calls=[fragment(0, None, None, '{"q": ')]),
        event(tool_calls=[fragment(1, "call_b", "fetch_url", '{"url": "u"}')]),
        event(tool_calls=[fragment(0, None, None, '"cats"}')]),
        event(finish_reason="tool_calls"),
    ]
    provider, _ = make_provider(as_stream(events))

    chunks = [chunk async for chunk in provider.stream(MESSAGES)]

    assert len(chunks) == 1
    assert chunks[0].finish_reason == "tool_calls"
    assert chunks[0].tool_calls == (
        ToolCall("call_a", "web_search", '{"q": "cats"}'),
        ToolCall("call_b", "fetch_url", '{"url": "u"}'),
    )


async def test_stream_wraps_errors_raised_mid_stream() -> None:
    async def failing() -> AsyncIterator[NS]:
        yield event(content="partial")
        raise OpenAIError("connection lost")

    provider, _ = make_provider(failing())
    received: list[str] = []
    with pytest.raises(ProviderError, match="connection lost"):
        async for chunk in provider.stream(MESSAGES):
            received.append(chunk.delta)
    assert received == ["partial"]


def make_embedder(result: Any, embed_model: str | None = "embed-model") -> tuple[Any, Any]:
    embeddings = FakeCompletions(result)
    client: Any = NS(embeddings=embeddings)
    provider = OpenAICompatProvider(
        client, name="test", default_model="chat-model", embed_model=embed_model
    )
    return provider, embeddings


async def test_embed_returns_vectors_in_input_order() -> None:
    response = NS(data=[NS(index=1, embedding=[0.0, 1.0]), NS(index=0, embedding=[1.0, 0.0])])
    provider, embeddings = make_embedder(response)

    vectors = await provider.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert embeddings.calls == [
        {"model": "embed-model", "input": ["first", "second"], "encoding_format": "float"}
    ]


async def test_embed_edge_cases() -> None:
    provider, embeddings = make_embedder(NS(data=[NS(index=0, embedding=[1.0])]))
    assert await provider.embed([]) == []
    assert embeddings.calls == []

    with pytest.raises(ProviderError, match="expected 2 embeddings"):
        await provider.embed(["a", "b"])

    unconfigured, _ = make_embedder(NS(data=[]), embed_model=None)
    with pytest.raises(ProviderError, match="no embedding model"):
        await unconfigured.embed(["a"])

    failing, _ = make_embedder(OpenAIError("boom"))
    with pytest.raises(ProviderError, match="boom"):
        await failing.embed(["a"])


class ScriptedCompletions:
    """Returns (or raises) the scripted outcomes one call at a time."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ollama(*outcomes: Any) -> tuple[OllamaProvider, ScriptedCompletions]:
    provider = OllamaProvider(base_url="http://localhost:11434/v1", default_model="local-model")
    completions = ScriptedCompletions(*outcomes)
    provider._client = NS(chat=NS(completions=completions))  # type: ignore[assignment]
    return provider, completions


OK = NS(choices=[NS(message=NS(content="fine", tool_calls=None), finish_reason="stop")], usage=None)


async def test_ollama_uses_max_tokens_and_disables_thinking() -> None:
    provider = OllamaProvider(base_url="http://localhost:11434/v1", default_model="local-model")
    request = provider._request(
        MESSAGES, model=None, tools=None, temperature=None, max_tokens=32, response_format=None
    )
    assert request == {
        "model": "local-model",
        "messages": MESSAGES,
        "max_tokens": 32,
        "reasoning_effort": "none",
        "extra_body": {"think": False},
    }
    await provider.aclose()

    thinking = OllamaProvider(
        base_url="http://localhost:11434/v1", default_model="local-model", think=True
    )
    request = thinking._request(
        MESSAGES, model=None, tools=None, temperature=None, max_tokens=None, response_format=None
    )
    assert request == {"model": "local-model", "messages": MESSAGES}
    await thinking.aclose()


async def test_ollama_drops_thinking_controls_when_the_model_rejects_them() -> None:
    provider, completions = ollama(OpenAIError('400: model "x" does not support thinking'), OK, OK)

    result = await provider.complete(MESSAGES)

    assert result.content == "fine"
    assert "reasoning_effort" in completions.calls[0]
    assert "reasoning_effort" not in completions.calls[1]
    # Later calls skip the controls entirely.
    await provider.complete(MESSAGES)
    assert "reasoning_effort" not in completions.calls[2]


async def test_ollama_reraises_unrelated_errors_without_retrying() -> None:
    provider, completions = ollama(OpenAIError("connection refused"))
    with pytest.raises(ProviderError, match="connection refused"):
        await provider.complete(MESSAGES)
    assert len(completions.calls) == 1


async def test_ollama_stream_retries_only_when_nothing_was_produced() -> None:
    async def failing() -> AsyncIterator[NS]:
        yield event(content="partial")
        raise OpenAIError("thinking broke mid-way")

    provider, completions = ollama(
        OpenAIError("thinking is not supported"),
        as_stream([event(content="ok"), event(finish_reason="stop")]),
        failing(),
    )

    chunks = [chunk async for chunk in provider.stream(MESSAGES)]
    assert [chunk.delta for chunk in chunks] == ["ok", ""]
    assert len(completions.calls) == 2

    # An error after output has started is never retried.
    with pytest.raises(ProviderError):
        async for _ in provider.stream(MESSAGES):
            pass
    assert len(completions.calls) == 3


async def test_empty_content_is_logged_with_diagnostics(caplog: pytest.LogCaptureFixture) -> None:
    message = NS(content="", tool_calls=None, reasoning="I was thinking...")
    response = NS(choices=[NS(message=message, finish_reason="length")], usage=usage(10, 256))
    provider, _ = make_provider(response)

    with caplog.at_level("WARNING", logger="bot.providers._compat"):
        result = await provider.complete(MESSAGES)

    assert result.content == ""
    [record] = [r for r in caplog.records if "no content" in r.getMessage()]
    assert record.had_reasoning is True  # type: ignore[attr-defined]
    assert record.finish_reason == "length"  # type: ignore[attr-defined]


async def test_factories_follow_settings() -> None:
    openai_settings = make_settings()
    azure_settings = make_settings(
        llm_provider_cloud="azure",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="key",
        azure_openai_api_version="2025-01-01",
        cloud_chat_model="my-deployment",
    )

    local = build_local_provider(openai_settings)
    cloud_openai = build_cloud_provider(openai_settings)
    cloud_azure = build_cloud_provider(azure_settings)

    assert isinstance(local, OllamaProvider)
    assert local.default_model == "qwen3-vl:4b-instruct"
    assert local.embed_model == "bge-m3"
    assert isinstance(cloud_openai, OpenAIProvider)
    assert cloud_openai.default_model == "chat-model"
    assert isinstance(cloud_azure, AzureOpenAIProvider)
    assert cloud_azure.default_model == "my-deployment"
    for provider in (local, cloud_openai, cloud_azure):
        await provider.aclose()
