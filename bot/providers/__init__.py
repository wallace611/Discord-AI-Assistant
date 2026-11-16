"""LLM provider abstraction. Switching backends only requires changing ``.env``."""

from bot.config import Settings
from bot.providers.azure import AzureOpenAIProvider
from bot.providers.base import (
    ChatChunk,
    ChatProvider,
    ChatResponse,
    EmbeddingProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)
from bot.providers.ollama import OllamaProvider
from bot.providers.openai import OpenAIProvider

__all__ = [
    "ChatChunk",
    "ChatProvider",
    "ChatResponse",
    "EmbeddingProvider",
    "Message",
    "ProviderError",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_cloud_provider",
    "build_local_provider",
]


def build_local_provider(settings: Settings) -> OllamaProvider:
    """Provider for the always-loaded local models (chat and embeddings)."""
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.local_router_model,
        embed_model=settings.local_embed_model,
    )


def build_cloud_provider(settings: Settings) -> ChatProvider:
    """Provider for hard reasoning, selected by ``LLM_PROVIDER_CLOUD``."""
    # ``Settings`` validation guarantees the credentials for the chosen backend exist.
    if settings.llm_provider_cloud == "azure":
        assert settings.azure_openai_endpoint is not None
        assert settings.azure_openai_api_key is not None
        assert settings.azure_openai_api_version is not None
        return AzureOpenAIProvider(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            default_model=settings.cloud_chat_model,
        )
    assert settings.openai_api_key is not None
    return OpenAIProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        default_model=settings.cloud_chat_model,
    )
