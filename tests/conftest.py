import pytest

from bot.config import Settings

_ENV_VARS = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment out of every test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
