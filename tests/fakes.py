"""Test doubles shared across test modules."""

import json
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from bot.config import Settings
from bot.memory import StoredMessage
from bot.providers.base import ChatChunk, ChatResponse, Message, ToolCall, Usage

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def tool_call_stream(
    *calls: tuple[str, dict[str, Any]], text: str = "", usage: Usage | None = None
) -> list[ChatChunk]:
    """A scripted stream in which the model requests the given tool calls."""
    chunks = [ChatChunk(delta=text)] if text else []
    chunks.append(
        ChatChunk(
            tool_calls=tuple(
                ToolCall(id=f"call_{index}", name=name, arguments=json.dumps(args))
                for index, (name, args) in enumerate(calls)
            ),
            finish_reason="tool_calls",
            usage=usage,
        )
    )
    return chunks


def text_stream(*parts: str, usage: Usage | None = None) -> list[ChatChunk]:
    """A scripted stream that just answers with text."""
    return [
        *(ChatChunk(delta=part) for part in parts),
        ChatChunk(finish_reason="stop", usage=usage),
    ]


def make_settings(**overrides: Any) -> Settings:
    """Minimal valid settings that never read the real ``.env``."""
    values: dict[str, Any] = {
        "discord_token": "token",
        "openai_api_key": "key",
        "cloud_chat_model": "chat-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def stored(
    message_id: int,
    content: str = "hello",
    *,
    channel_id: int = 100,
    guild_id: int | None = 200,
    author_id: int = 7,
    author_name: str = "alice",
    is_bot: bool = False,
    minutes: float = 0,
) -> StoredMessage:
    """A stored message sent ``minutes`` after ``T0``."""
    return StoredMessage(
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        author_id=author_id,
        author_name=author_name,
        is_bot=is_bot,
        content=content,
        created_at=T0 + timedelta(minutes=minutes),
    )


def keyword_embedding(*keywords: str) -> Callable[[str], list[float]]:
    """Embedding with one dimension per keyword: 1.0 when the text contains it."""

    def embed(text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if keyword in lowered else 0.0 for keyword in keywords]

    return embed


class FakeProvider:
    """In-memory chat and embedding provider that records calls.

    ``replies`` are returned by ``complete`` in order; once they run out (or when
    none are given) ``reply`` is returned.
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        reply: str = "",
        replies: Sequence[str] = (),
        chunks: Sequence[str] = (),
        usage: Usage | None = None,
        error: Exception | None = None,
        stream_error: Exception | None = None,
        embed_fn: Callable[[str], list[float]] = lambda text: [1.0, 0.0],
        embed_error: Exception | None = None,
        streams: Sequence[Sequence[ChatChunk]] = (),
    ) -> None:
        self.name = name
        self.default_model = f"{name}-model"
        self.reply = reply
        self.replies = list(replies)
        self.chunks = chunks
        self.usage = usage
        self.error = error
        self.stream_error = stream_error
        self.embed_fn = embed_fn
        self.embed_error = embed_error
        # Scripted per-call streams (each a full chunk sequence); used before ``chunks``.
        self.streams = [list(stream) for stream in streams]
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.complete_calls.append({"messages": list(messages), **kwargs})
        if self.error is not None:
            raise self.error
        content = self.replies.pop(0) if self.replies else self.reply
        return ChatResponse(content=content, finish_reason="stop", usage=self.usage)

    async def stream(self, messages: Sequence[Message], **kwargs: Any) -> AsyncIterator[ChatChunk]:
        self.stream_calls.append({"messages": list(messages), **kwargs})
        if self.streams:
            for scripted in self.streams.pop(0):
                yield scripted
            return
        for chunk in self.chunks:
            yield ChatChunk(delta=chunk)
        if self.stream_error is not None:
            raise self.stream_error
        yield ChatChunk(finish_reason="stop", usage=self.usage)

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        return [self.embed_fn(text) for text in texts]

    async def aclose(self) -> None:
        return None


class FakeMessage:
    """Stands in for a sent Discord message; records every edit."""

    def __init__(self, content: str, id: int) -> None:
        self.id = id
        self.content = content
        self.sent_as = content  # the content of the original send
        self.edits: list[str] = []

    async def edit(self, *, content: str, **_: Any) -> None:
        self.content = content
        self.edits.append(content)


class FakeChannel:
    """Collects messages sent through a ``SendFn``."""

    def __init__(self, first_id: int = 9000) -> None:
        self.messages: list[FakeMessage] = []
        self._next_id = first_id

    async def send(self, content: str) -> FakeMessage:
        message = FakeMessage(content, self._next_id)
        self._next_id += 1
        self.messages.append(message)
        return message

    @property
    def texts(self) -> list[str]:
        return [message.content for message in self.messages]
