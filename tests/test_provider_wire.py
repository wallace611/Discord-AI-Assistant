"""Provider tests through the real OpenAI SDK, with only the network replaced.

The other provider tests fake the SDK client; these check what actually goes on
the wire for tool calling, and that a real server-sent-event stream is parsed
into complete tool calls.
"""

import json
from typing import Any

import httpx2
from openai import AsyncOpenAI

from bot.agent import AgentLoop, Done, TextDelta, ToolFinished, ToolRegistry, ToolStarted
from bot.agent.tools.base import Tier, Tool, ToolContext, ToolResult, object_schema
from bot.providers._compat import OpenAICompatProvider
from bot.providers.base import Usage

CTX = ToolContext(guild_id=200, channel_id=100, user_id=7, message_id=5000)


def sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


USAGE_ONLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "test-model",
    "choices": [],
    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
}

# The model asks for one tool call, with the arguments split across events.
TOOL_CALL_STREAM = sse(
    chunk({"role": "assistant", "content": None}),
    chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": ""},
                }
            ]
        }
    ),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q": "py'}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'thon"}'}}]}),
    chunk({}, finish="tool_calls"),
    USAGE_ONLY,
)

ANSWER_STREAM = sse(
    chunk({"role": "assistant", "content": "It is "}),
    chunk({"content": "a language."}),
    chunk({}, finish="stop"),
    USAGE_ONLY,
)


def make_provider(*bodies: bytes) -> tuple[OpenAICompatProvider, list[dict[str, Any]]]:
    """A provider on the real SDK whose HTTP layer replays ``bodies`` in order."""
    requests: list[dict[str, Any]] = []
    remaining = list(bodies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=remaining.pop(0)
        )

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAICompatProvider(client, name="test", default_model="test-model"), requests


async def test_tool_call_round_trip_through_the_real_sdk() -> None:
    calls: list[dict[str, Any]] = []

    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        calls.append(args)
        return ToolResult("Python is a programming language.")

    lookup = Tool(
        name="lookup",
        description="Look something up",
        tier=Tier.READ_ONLY,
        parameters=object_schema({"q": {"type": "string"}}),
        run=run,
    )
    provider, requests = make_provider(TOOL_CALL_STREAM, ANSWER_STREAM)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "what is python?"}]

    events = [e async for e in AgentLoop(provider, ToolRegistry([lookup])).run(messages, CTX)]

    # The fragmented arguments were reassembled before the tool ran.
    assert calls == [{"q": "python"}]
    assert events == [
        ToolStarted("lookup", {"q": "python"}),
        ToolFinished("lookup", True, "Python is a programming language."),
        TextDelta("It is "),
        TextDelta("a language."),
        Done(1, Usage(22, 8, 30)),
    ]

    first, second = requests
    assert first["model"] == "test-model"
    assert first["stream"] is True
    assert first["stream_options"] == {"include_usage": True}
    assert first["tools"] == [lookup.spec()]
    assert first["messages"] == [{"role": "user", "content": "what is python?"}]

    # The follow-up request carries the assistant's tool call and the tool's output.
    assistant, tool = second["messages"][1:]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "python"}'},
        }
    ]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_abc"
    assert "Python is a programming language." in tool["content"]


async def test_plain_stream_through_the_real_sdk() -> None:
    provider, requests = make_provider(ANSWER_STREAM)

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert "".join(c.delta for c in chunks) == "It is a language."
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == Usage(11, 4, 15)
    assert "tools" not in requests[0]
