from pathlib import Path

import pytest
from pydantic import ValidationError

from bot.config import Settings
from fakes import make_settings


def test_defaults() -> None:
    settings = make_settings()
    assert settings.llm_provider_cloud == "openai"
    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.local_router_model == "qwen3-vl:4b-instruct"
    assert settings.daily_cloud_token_budget_per_user is None
    assert settings.incus_sandbox_profile is None
    assert settings.run_code_timeout_seconds == 30.0
    assert settings.mcp_config_path == "mcp_servers.json"


def test_secrets_are_not_exposed_in_repr() -> None:
    settings = make_settings(discord_token="super-secret")
    assert "super-secret" not in repr(settings)
    assert settings.discord_token.get_secret_value() == "super-secret"


def test_openai_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_settings(openai_api_key=None)


def test_azure_requires_all_azure_settings() -> None:
    with pytest.raises(ValidationError, match="AZURE_OPENAI_API_VERSION"):
        make_settings(
            llm_provider_cloud="azure",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="key",
        )


def test_azure_does_not_need_openai_key() -> None:
    settings = make_settings(
        llm_provider_cloud="azure",
        openai_api_key=None,
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="key",
        azure_openai_api_version="2025-01-01",
    )
    assert settings.llm_provider_cloud == "azure"


def test_cloud_chat_model_is_required() -> None:
    with pytest.raises(ValidationError, match="cloud_chat_model"):
        Settings(_env_file=None, discord_token="token", openai_api_key="key")


def test_empty_values_in_env_file_fall_back_to_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DISCORD_TOKEN=token\n"
        "OPENAI_API_KEY=key\n"
        "CLOUD_CHAT_MODEL=chat-model\n"
        "CLOUD_VISION_MODEL=\n"
        "DAILY_CLOUD_TOKEN_BUDGET_PER_USER=\n"
        "OLLAMA_BASE_URL=\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.cloud_vision_model is None
    assert settings.daily_cloud_token_budget_per_user is None
    assert settings.ollama_base_url == "http://localhost:11434/v1"


def test_env_example_lists_every_setting() -> None:
    example = (Path(__file__).parent.parent / ".env.example").read_text(encoding="utf-8")
    listed = {
        line.split("=", 1)[0]
        for line in example.splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert {name.upper() for name in Settings.model_fields} <= listed
