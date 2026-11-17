import json
from pathlib import Path

from bot.agent.setup import add_sandbox, build_registry, start_mcp
from bot.agent.tools.run_code import CommandResult, remove_stale_sandboxes, verify_sandbox
from bot.memory import InMemoryStore, MemoryService
from fakes import FakeProvider, make_settings

OFFLINE_PROFILE = """\
config: {}
devices:
  root:
    path: /
    pool: default
    type: disk
name: sandbox
"""
ONLINE_PROFILE = (
    OFFLINE_PROFILE
    + """\
  eth0:
    name: eth0
    network: incusbr0
    type: nic
"""
)


class FakeIncus:
    def __init__(self, profile: str | None = OFFLINE_PROFILE, stale: str = "") -> None:
        self.profile = profile
        self.stale = stale
        self.calls: list[list[str]] = []

    async def __call__(self, argv: list[str], stdin: bytes | None, timeout: float) -> CommandResult:
        self.calls.append(argv)
        if argv[:3] == ["incus", "profile", "show"]:
            if self.profile is None:
                return CommandResult(1, "", "Error: Profile not found")
            return CommandResult(0, self.profile, "")
        if argv[:2] == ["incus", "list"]:
            return CommandResult(0, self.stale, "")
        return CommandResult(0, "", "")


def make_memory() -> MemoryService:
    provider = FakeProvider()
    return MemoryService(InMemoryStore(), chat=provider, embedder=provider)


def test_builtin_registry_has_the_documented_tools() -> None:
    registry = build_registry(make_settings(), make_memory())
    assert {tool.name for tool in registry} == {
        "web_search",
        "fetch_url",
        "search_server_history",
        "recall_memory",
        "remember",
        "create_thread",
        "pin_message",
        "create_poll",
    }


async def test_sandbox_is_skipped_without_a_profile() -> None:
    registry = build_registry(make_settings(), make_memory())
    incus = FakeIncus()
    assert await add_sandbox(make_settings(), registry, incus) is False
    assert incus.calls == []
    assert registry.get("run_code") is None


async def test_sandbox_is_added_when_the_profile_is_offline() -> None:
    settings = make_settings(incus_sandbox_profile="sandbox")
    registry = build_registry(settings, make_memory())
    incus = FakeIncus(stale="bot-sandbox-old1\nbot-sandbox-old2\n")

    assert await add_sandbox(settings, registry, incus) is True

    assert registry.get("run_code") is not None
    deletes = [argv for argv in incus.calls if argv[:2] == ["incus", "delete"]]
    assert [argv[-1] for argv in deletes] == ["bot-sandbox-old1", "bot-sandbox-old2"]


async def test_sandbox_is_refused_when_the_profile_has_a_nic() -> None:
    settings = make_settings(incus_sandbox_profile="sandbox")
    registry = build_registry(settings, make_memory())

    assert await add_sandbox(settings, registry, FakeIncus(profile=ONLINE_PROFILE)) is False
    assert registry.get("run_code") is None


async def test_verify_sandbox_reports_problems() -> None:
    from bot.agent.tools.run_code import SandboxConfig

    config = SandboxConfig(image="images:debian/12", profile="sandbox")
    assert await verify_sandbox(config, FakeIncus()) is None
    assert "network device" in (
        await verify_sandbox(config, FakeIncus(profile=ONLINE_PROFILE)) or ""
    )
    assert "not found" in (await verify_sandbox(config, FakeIncus(profile=None)) or "")

    async def missing(argv: list[str], stdin: bytes | None, timeout: float) -> CommandResult:
        raise FileNotFoundError("incus")

    assert await verify_sandbox(config, missing) == "incus is not installed"
    assert await remove_stale_sandboxes(FakeIncus()) == 0


async def test_mcp_is_skipped_without_a_config_file(tmp_path: Path) -> None:
    settings = make_settings(mcp_config_path=str(tmp_path / "missing.json"))
    assert await start_mcp(settings, build_registry(settings, make_memory())) is None
    assert (
        await start_mcp(
            make_settings(mcp_config_path=None), build_registry(settings, make_memory())
        )
        is None
    )


async def test_mcp_servers_that_fail_to_connect_are_skipped(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {"servers": {"bad": {"transport": "stdio", "command": "definitely-missing-binary"}}}
        ),
        encoding="utf-8",
    )
    settings = make_settings(mcp_config_path=str(config))
    registry = build_registry(settings, make_memory())

    manager = await start_mcp(settings, registry)

    assert manager is not None
    assert manager.tools == []
    await manager.aclose()
