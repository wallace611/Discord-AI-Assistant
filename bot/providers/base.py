"""Provider-agnostic chat interface.

Every LLM call in the bot goes through ``ChatProvider``. Messages, tool specs and
response formats use the OpenAI chat-completions wire format, which all three
backends (Ollama ``/v1``, OpenAI, Azure OpenAI) accept. Business logic must not
import the ``openai`` SDK directly.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

type Message = dict[str, Any]
type ToolSpec = dict[str, Any]


class ProviderError(Exception):
    """A backend call failed (network, auth, rate limit, bad request, ...)."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string, exactly as produced by the model


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Result of a non-streaming completion."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One streaming event.

    Text arrives incrementally in ``delta``. The last chunk of a stream has an
    empty ``delta`` and carries the fully assembled ``tool_calls``, the
    ``finish_reason`` and, when the backend reports it, ``usage``.
    """

    delta: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: Usage | None = None


class ChatProvider(Protocol):
    """Chat backend. ``model=None`` means the provider's ``default_model``."""

    name: str
    default_model: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]: ...

    async def aclose(self) -> None: ...


class EmbeddingProvider(Protocol):
    """Text embedding backend. Returns one vector per input, in input order."""

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]: ...
