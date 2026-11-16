"""Application settings, loaded from the environment and ``.env``."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case env variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # ``FOO=`` in .env means "unset", so the field default applies.
        env_ignore_empty=True,
        extra="ignore",
    )

    discord_token: SecretStr

    llm_provider_cloud: Literal["openai", "azure"] = "openai"
    openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str | None = None

    # Cloud model names are never hard-coded; they always come from the environment.
    cloud_chat_model: str
    cloud_vision_model: str | None = None
    cloud_stt_model: str | None = None
    cloud_tts_model: str | None = None
    cloud_image_model: str | None = None

    ollama_base_url: str = "http://localhost:11434/v1"
    local_router_model: str = "qwen3-vl:4b"
    local_embed_model: str = "bge-m3"

    database_url: SecretStr | None = None
    searxng_url: str = "http://localhost:8080"

    # None means unlimited.
    daily_cloud_token_budget_per_user: int | None = Field(default=None, ge=0)

    voice_worker_ipc_url: str | None = None

    @model_validator(mode="after")
    def _require_cloud_credentials(self) -> Self:
        if self.llm_provider_cloud == "openai":
            required = {"OPENAI_API_KEY": self.openai_api_key}
        else:
            required = {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
                "AZURE_OPENAI_API_VERSION": self.azure_openai_api_version,
            }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"LLM_PROVIDER_CLOUD={self.llm_provider_cloud} requires: {', '.join(missing)}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
