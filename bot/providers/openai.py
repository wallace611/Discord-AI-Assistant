"""OpenAI cloud provider."""

from openai import AsyncOpenAI

from bot.providers._compat import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    def __init__(self, *, api_key: str, default_model: str, timeout: float = 120.0) -> None:
        client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        super().__init__(client, name="openai", default_model=default_model)
