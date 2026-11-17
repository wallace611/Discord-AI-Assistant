"""Azure OpenAI cloud provider. Model names are Azure deployment names."""

from openai import AsyncAzureOpenAI

from bot.providers._cloud import CloudProvider


class AzureOpenAIProvider(CloudProvider):
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        default_model: str,
        tool_reasoning_effort: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )
        super().__init__(
            client,
            name="azure",
            default_model=default_model,
            tool_reasoning_effort=tool_reasoning_effort,
        )
