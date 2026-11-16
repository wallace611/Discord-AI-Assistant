"""Azure OpenAI cloud provider. Model names are Azure deployment names."""

from openai import AsyncAzureOpenAI

from bot.providers._compat import OpenAICompatProvider


class AzureOpenAIProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        default_model: str,
        timeout: float = 120.0,
    ) -> None:
        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )
        super().__init__(client, name="azure", default_model=default_model)
