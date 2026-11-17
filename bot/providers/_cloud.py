"""Shared behaviour of the cloud backends (OpenAI and Azure OpenAI)."""

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import AsyncOpenAI

from bot.providers._compat import OpenAICompatProvider
from bot.providers.base import ChatChunk, ChatResponse, Message, ProviderError, ToolSpec

logger = logging.getLogger(__name__)

# What reasoning models accept on /chat/completions when tools are present.
_TOOLS_SAFE_EFFORT = "none"


class CloudProvider(OpenAICompatProvider):
    """Chat-completions backend that copes with reasoning models and tools.

    Some reasoning models (for example on Azure) reject function tools on
    ``/chat/completions`` unless ``reasoning_effort`` is ``"none"``. Requests
    with tools carry ``tool_reasoning_effort`` when it is set. When it is not
    set and the server refuses for that reason, the request is retried once with
    ``"none"``, and later tool requests send ``"none"`` from the start.
    Requests without tools never carry the field, so they keep the model's
    default reasoning.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        name: str,
        default_model: str,
        tool_reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(client, name=name, default_model=default_model)
        self.tool_reasoning_effort = tool_reasoning_effort

    def _request(self, messages: Sequence[Message], **kwargs: Any) -> dict[str, Any]:
        request = super()._request(messages, **kwargs)
        if request.get("tools") and self.tool_reasoning_effort is not None:
            request["reasoning_effort"] = self.tool_reasoning_effort
        return request

    def _effort_rejected(self, exc: ProviderError, tools: Sequence[ToolSpec] | None) -> bool:
        """True when the server refused tools because of reasoning; switches to "none"."""
        if not tools or self.tool_reasoning_effort == _TOOLS_SAFE_EFFORT:
            return False
        if "reasoning_effort" not in str(exc):
            return False
        logger.warning(
            "model rejected tools with reasoning, retrying with reasoning_effort=none",
            extra={
                "provider": self.name,
                "model": self.default_model,
                "configured": self.tool_reasoning_effort,
            },
        )
        self.tool_reasoning_effort = _TOOLS_SAFE_EFFORT
        return True

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        try:
            return await super().complete(messages, **kwargs)
        except ProviderError as exc:
            if not self._effort_rejected(exc, kwargs.get("tools")):
                raise
            return await super().complete(messages, **kwargs)

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
        kwargs: dict[str, Any] = {
            "model": model,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        produced = False
        try:
            async for chunk in super().stream(messages, **kwargs):
                produced = True
                yield chunk
        except ProviderError as exc:
            # Only safe to retry when the request was rejected up front.
            if produced or not self._effort_rejected(exc, tools):
                raise
            async for chunk in super().stream(messages, **kwargs):
                yield chunk
