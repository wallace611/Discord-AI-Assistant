import pytest

from bot.__main__ import StartupError, open_store
from bot.memory import InMemoryStore
from fakes import make_settings


async def test_without_database_url_memory_is_kept_in_process() -> None:
    store = await open_store(make_settings())
    assert isinstance(store, InMemoryStore)


async def test_unreachable_database_gives_an_actionable_error() -> None:
    # Port 1 on loopback: nothing listens there, so the connection is refused.
    settings = make_settings(database_url="postgresql://bot:pw@127.0.0.1:1/bot")

    with pytest.raises(StartupError) as raised:
        await open_store(settings)

    message = str(raised.value)
    assert "docker compose up -d" in message
    assert "pw" not in message
