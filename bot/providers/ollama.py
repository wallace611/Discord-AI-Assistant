"""Local models served by Ollama through its OpenAI-compatible ``/v1`` endpoint."""

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from openai import AsyncOpenAI

from bot.providers._compat import OpenAICompatProvider
from bot.providers.base import ChatChunk, ChatResponse, Message, ProviderError, ToolSpec

logger = logging.getLogger(__name__)


class OllamaProvider(OpenAICompatProvider):
    """Ollama backend. Thinking is switched off: the local model does small,
    latency-sensitive jobs (routing, summaries, fact extraction) where reasoning
    tokens only add seconds and can leave the answer empty."""

    max_tokens_param: ClassVar[str] = "max_tokens"

    def __init__(
        self,
        *,
        base_url: str,
        default_model: str,
        embed_model: str | None = None,
        timeout: float = 120.0,
        think: bool = False,
    ) -> None:
        # Ollama ignores the API key, but the SDK refuses to start without one.
        client = AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=timeout)
        super().__init__(
            client, name="ollama", default_model=default_model, embed_model=embed_model
        )
        # When False, every request asks Ollama not to think. Cleared automatically
        # if the server rejects the fields (model without thinking support).
        self._control_thinking = not think

    def _request(self, messages: Sequence[Message], **kwargs: Any) -> dict[str, Any]:
        request = super()._request(messages, **kwargs)
        if self._control_thinking:
            # ``reasoning_effort`` is the documented control on /v1; ``think`` is
            # the native field, passed too in case the server honours only that.
            request["reasoning_effort"] = "none"
            request["extra_body"] = {"think": False}
        return request

    def _thinking_rejected(self, exc: ProviderError) -> bool:
        """True when the server refused the thinking controls; disables them."""
        if self._control_thinking and "think" in str(exc).lower():
            logger.warning(
                "model rejected thinking controls, retrying without them",
                extra={"model": self.default_model, "error": str(exc)},
            )
            self._control_thinking = False
            return True
        return False

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        try:
            return await super().complete(messages, **kwargs)
        except ProviderError as exc:
            if not self._thinking_rejected(exc):
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
            if produced or not self._thinking_rejected(exc):
                raise
            async for chunk in super().stream(messages, **kwargs):
                yield chunk
