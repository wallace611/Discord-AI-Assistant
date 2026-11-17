"""OpenAI cloud provider."""

from openai import AsyncOpenAI

from bot.providers._cloud import CloudProvider


class OpenAIProvider(CloudProvider):
    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        tool_reasoning_effort: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        super().__init__(
            client,
            name="openai",
            default_model=default_model,
            tool_reasoning_effort=tool_reasoning_effort,
        )
