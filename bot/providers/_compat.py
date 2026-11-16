"""Shared implementation for backends that speak the OpenAI chat-completions API."""

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from openai import AsyncOpenAI, OpenAIError

from bot.providers.base import (
    ChatChunk,
    ChatResponse,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

logger = logging.getLogger(__name__)


def _to_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        prompt_tokens=raw.prompt_tokens or 0,
        completion_tokens=raw.completion_tokens or 0,
        total_tokens=raw.total_tokens or 0,
    )


class OpenAICompatProvider:
    """``ChatProvider`` backed by an ``AsyncOpenAI``-style client."""

    # OpenAI and Azure reject ``max_tokens`` on newer models; Ollama expects it.
    max_tokens_param: ClassVar[str] = "max_completion_tokens"

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        name: str,
        default_model: str,
        embed_model: str | None = None,
    ) -> None:
        self._client = client
        self.name = name
        self.default_model = default_model
        self.embed_model = embed_model

    def _request(
        self,
        messages: Sequence[Message],
        *,
        model: str | None,
        tools: Sequence[ToolSpec] | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": list(messages),
        }
        if tools:
            request["tools"] = list(tools)
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request[self.max_tokens_param] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format
        return request

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        request = self._request(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        try:
            response = await self._client.chat.completions.create(**request)
        except OpenAIError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        if not response.choices:
            raise ProviderError(f"{self.name}: response contained no choices")

        choice = response.choices[0]
        tool_calls = tuple(
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in choice.message.tool_calls or ()
        )
        content = choice.message.content or ""
        if not content and not tool_calls:
            # Typical cause: a thinking model spent the whole token budget on
            # reasoning, which some servers return in a separate field.
            reasoning = getattr(choice.message, "reasoning", None) or getattr(
                choice.message, "reasoning_content", None
            )
            logger.warning(
                "model returned no content",
                extra={
                    "provider": self.name,
                    "model": request["model"],
                    "finish_reason": choice.finish_reason,
                    "had_reasoning": bool(reasoning),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                },
            )
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=_to_usage(response.usage),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        request = self._request(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        # Tool calls arrive as fragments keyed by index; assemble them here so
        # callers only ever see complete calls.
        pending: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage: Usage | None = None
        try:
            events = await self._client.chat.completions.create(
                **request, stream=True, stream_options={"include_usage": True}
            )
            async for event in events:
                usage = _to_usage(getattr(event, "usage", None)) or usage
                if not event.choices:
                    continue
                choice = event.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                for piece in delta.tool_calls or ():
                    slot = pending.setdefault(piece.index, {"id": "", "name": "", "arguments": ""})
                    if piece.id:
                        slot["id"] = piece.id
                    if piece.function is not None:
                        slot["name"] += piece.function.name or ""
                        slot["arguments"] += piece.function.arguments or ""
                if delta.content:
                    yield ChatChunk(delta=delta.content)
        except OpenAIError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc

        yield ChatChunk(
            tool_calls=tuple(ToolCall(**pending[index]) for index in sorted(pending)),
            finish_reason=finish_reason,
            usage=usage,
        )

    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        chosen = model or self.embed_model
        if chosen is None:
            raise ProviderError(f"{self.name}: no embedding model configured")
        if not texts:
            return []
        try:
            # Ask for plain floats; not every compatible server supports base64.
            response = await self._client.embeddings.create(
                model=chosen, input=list(texts), encoding_format="float"
            )
        except OpenAIError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
        vectors = [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]
        if len(vectors) != len(texts):
            raise ProviderError(
                f"{self.name}: expected {len(texts)} embeddings, got {len(vectors)}"
            )
        return vectors

    async def aclose(self) -> None:
        await self._client.close()
