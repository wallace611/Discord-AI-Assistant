from pathlib import Path

import pytest

from bot.__main__ import StartupError, open_store, run
from bot.gateway import BotClient
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


async def test_run_wires_everything_and_shuts_down_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real startup path, stopped just short of logging in to Discord."""
    started: list[BotClient] = []

    async def fake_start(self: BotClient, token: str) -> None:
        started.append(self)
        assert token == "token"

    monkeypatch.setattr(BotClient, "start", fake_start)
    settings = make_settings(mcp_config_path=str(tmp_path / "none.json"))

    await run(settings)

    [client] = started
    assert client.tree.get_command("memory") is not None
    agent = client._handler._agent
    assert agent is not None
    tools = {tool.name for tool in agent._registry}
    assert {"web_search", "fetch_url", "search_server_history", "remember"} <= tools
    assert "run_code" not in tools  # no sandbox profile configured
    assert client.is_closed()
