import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from bot.agent.tools.base import Tier, Tool, ToolContext, ToolError
from bot.agent.tools.mcp import (
    MCPConfig,
    MCPManager,
    ServerConfig,
    load_config,
    result_text,
    tool_from_mcp,
)

CTX = ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4)


def text_item(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def call_result(
    *content: SimpleNamespace, is_error: bool = False, structured: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(content=list(content), is_error=is_error, structured_content=structured)


class FakeSession:
    """Stands in for ``mcp.ClientSession``; records every tool call."""

    def __init__(
        self, result: SimpleNamespace | None = None, error: Exception | None = None
    ) -> None:
        self.result = result if result is not None else call_result(text_item("ok"))
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


def mcp_tool(
    name: str = "lookup",
    description: str | None = "Look something up.",
    input_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, input_schema=input_schema)


# --- configuration ---------------------------------------------------------


def write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_config_reads_servers(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "servers": {
                "docs": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["mcp-server-docs"],
                    "env": {"KEY": "值"},
                    "tier": "read_only",
                },
                "crm": {
                    "transport": "http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer x"},
                },
            }
        },
    )

    config = load_config(path)

    docs = config.servers["docs"]
    assert (docs.transport, docs.command, docs.args) == ("stdio", "uvx", ["mcp-server-docs"])
    assert docs.env == {"KEY": "值"}
    assert docs.permission() is Tier.READ_ONLY
    crm = config.servers["crm"]
    assert (crm.transport, crm.url) == ("http", "https://mcp.example.com/mcp")
    assert crm.headers == {"Authorization": "Bearer x"}


def test_load_config_without_servers_is_empty(tmp_path: Path) -> None:
    assert load_config(write_config(tmp_path, {})).servers == {}


@pytest.mark.parametrize(
    "data",
    [
        {"servers": {"docs": {"command": "uvx", "tirr": "read_only"}}},
        {"servers": {}, "sevrers": {}},
    ],
)
def test_load_config_rejects_unknown_keys(tmp_path: Path, data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        load_config(write_config(tmp_path, data))


def test_server_defaults_to_stdio_and_user_confirmation() -> None:
    server = ServerConfig(command="uvx")

    assert server.transport == "stdio"
    assert server.permission() is Tier.SIDE_EFFECT


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("read_only", Tier.READ_ONLY),
        ("sandbox", Tier.SANDBOX),
        ("side_effect", Tier.SIDE_EFFECT),
    ],
)
def test_permission_maps_each_tier(tier: str, expected: Tier) -> None:
    assert ServerConfig(tier=tier).permission() is expected


@pytest.mark.parametrize("tier", ["admin", "READ_ONLY", "", "1"])
def test_permission_rejects_unknown_tier(tier: str) -> None:
    with pytest.raises(ValueError, match="tier must be one of"):
        ServerConfig(tier=tier).permission()


# --- result_text -----------------------------------------------------------


def test_result_text_joins_text_items() -> None:
    assert result_text(call_result(text_item("first"), text_item("second"))) == "first\nsecond"


def test_result_text_marks_non_text_content() -> None:
    image = SimpleNamespace(type="image", data="aGk=", mime_type="image/png")

    assert result_text(call_result(text_item("see:"), image)) == "see:\n[image content omitted]"


def test_result_text_prefixes_errors() -> None:
    assert result_text(call_result(text_item("no such page"), is_error=True)) == (
        "error: no such page"
    )


def test_result_text_falls_back_to_structured_content() -> None:
    text = result_text(call_result(structured={"城市": "台北", "temp": 21}))

    assert json.loads(text) == {"城市": "台北", "temp": 21}
    assert "台北" in text  # not \u-escaped


def test_result_text_prefers_text_over_structured_content() -> None:
    assert result_text(call_result(text_item("21 degrees"), structured={"temp": 21})) == (
        "21 degrees"
    )


@pytest.mark.parametrize(
    "result",
    [call_result(), SimpleNamespace(content=None), SimpleNamespace()],
)
def test_result_text_handles_empty_results(result: SimpleNamespace) -> None:
    assert result_text(result) == "(no output)"


def test_result_text_marks_empty_errors() -> None:
    assert result_text(call_result(is_error=True)) == "error: (no output)"


# --- tool_from_mcp ---------------------------------------------------------


def test_tool_is_namespaced_by_server() -> None:
    tool = tool_from_mcp("docs", FakeSession(), mcp_tool("lookup"), Tier.READ_ONLY)

    assert isinstance(tool, Tool)
    assert tool.name == "docs__lookup"
    assert tool.description == "Look something up."
    assert tool.tier is Tier.READ_ONLY


def test_tool_keeps_the_configured_tier() -> None:
    tool = tool_from_mcp("crm", FakeSession(), mcp_tool(), Tier.SIDE_EFFECT)

    assert tool.tier is Tier.SIDE_EFFECT


def test_tool_carries_the_input_schema_into_its_spec() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    tool = tool_from_mcp("docs", FakeSession(), mcp_tool(input_schema=schema), Tier.READ_ONLY)

    assert tool.parameters == schema
    assert tool.spec() == {
        "type": "function",
        "function": {
            "name": "docs__lookup",
            "description": "Look something up.",
            "parameters": schema,
        },
    }


def test_tool_without_schema_accepts_an_empty_object() -> None:
    tool = tool_from_mcp("docs", FakeSession(), mcp_tool(input_schema=None), Tier.READ_ONLY)

    assert tool.parameters == {"type": "object", "properties": {}}


def test_tool_without_description_falls_back_to_its_name() -> None:
    tool = tool_from_mcp("docs", FakeSession(), mcp_tool(description=None), Tier.READ_ONLY)

    assert tool.description == "lookup"


def test_long_descriptions_are_capped() -> None:
    tool = tool_from_mcp("docs", FakeSession(), mcp_tool(description="d" * 5000), Tier.READ_ONLY)

    assert len(tool.description) == 1000


async def test_run_calls_the_server_with_the_original_tool_name() -> None:
    session = FakeSession(call_result(text_item("42")))
    tool = tool_from_mcp("docs", session, mcp_tool("lookup"), Tier.READ_ONLY)

    result = await tool.run({"query": "answer"}, CTX)

    assert session.calls == [("lookup", {"query": "answer"})]
    assert result.content == "42"
    assert result.sources == ()


async def test_run_turns_server_failures_into_tool_errors() -> None:
    session = FakeSession(error=ConnectionError("pipe closed"))
    tool = tool_from_mcp("docs", session, mcp_tool(), Tier.READ_ONLY)

    with pytest.raises(ToolError, match="MCP server 'docs' failed: pipe closed") as excinfo:
        await tool.run({}, CTX)

    assert isinstance(excinfo.value.__cause__, ConnectionError)


async def test_run_reports_tool_level_errors_as_content() -> None:
    session = FakeSession(call_result(text_item("rate limited"), is_error=True))
    tool = tool_from_mcp("docs", session, mcp_tool(), Tier.READ_ONLY)

    result = await tool.run({}, CTX)

    assert result.content == "error: rate limited"


# --- MCPManager ------------------------------------------------------------


async def test_manager_with_no_servers_has_no_tools() -> None:
    manager = MCPManager(MCPConfig())

    assert await manager.start() == []
    await manager.aclose()


@pytest.mark.parametrize(
    "server",
    [
        ServerConfig(transport="carrier-pigeon"),
        ServerConfig(transport="stdio"),  # no command
        ServerConfig(transport="http"),  # no url
    ],
)
async def test_manager_skips_misconfigured_servers(server: ServerConfig) -> None:
    manager = MCPManager(MCPConfig(servers={"broken": server}))

    assert await manager.start() == []
    await manager.aclose()


async def test_manager_keeps_healthy_servers_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MCPConfig(servers={"down": ServerConfig(command="x"), "up": ServerConfig(command="y")})
    manager = MCPManager(config)

    async def connect(name: str, server: ServerConfig) -> list[Tool]:
        if name == "down":
            raise ConnectionError("refused")
        return [tool_from_mcp(name, FakeSession(), mcp_tool(), server.permission())]

    monkeypatch.setattr(manager, "_connect", connect)

    tools = await manager.start()

    assert [tool.name for tool in tools] == ["up__lookup"]
    assert manager.tools == tools
    await manager.aclose()
