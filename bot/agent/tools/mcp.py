"""Tools provided by MCP servers.

Servers are listed in a JSON file (``MCP_CONFIG_PATH``)::

    {
      "servers": {
        "docs": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-docs"],
                 "env": {"KEY": "value"}, "tier": "read_only"},
        "crm":  {"transport": "http", "url": "https://mcp.example.com/mcp",
                 "headers": {"Authorization": "Bearer ..."}, "tier": "side_effect"}
      }
    }

``tier`` sets the permission tier for every tool of that server and defaults to
``side_effect`` (user confirmation), the safe choice for unknown tools.
"""

import json
import logging
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bot.agent.tools.base import Tier, Tool, ToolContext, ToolError, ToolResult

logger = logging.getLogger(__name__)

_TIERS = {"read_only": Tier.READ_ONLY, "sandbox": Tier.SANDBOX, "side_effect": Tier.SIDE_EFFECT}


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    tier: str = "side_effect"

    def permission(self) -> Tier:
        if self.tier not in _TIERS:
            raise ValueError(f"tier must be one of {', '.join(_TIERS)}")
        return _TIERS[self.tier]


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: dict[str, ServerConfig] = Field(default_factory=dict)


def load_config(path: Path) -> MCPConfig:
    return MCPConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def result_text(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into text for the model."""
    parts: list[str] = []
    for item in getattr(result, "content", None) or ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(f"[{getattr(item, 'type', 'non-text')} content omitted]")
    structured = getattr(result, "structured_content", None)
    if not parts and structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False))
    text = "\n".join(parts) or "(no output)"
    if getattr(result, "is_error", False):
        return f"error: {text}"
    return text


_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def tool_name(server: str, tool: str) -> str:
    """``server__tool`` restricted to what function calling accepts (64 safe chars)."""
    return _NAME_CHARS.sub("_", f"{server}__{tool}")[:64]


def tool_from_mcp(server: str, session: Any, tool: Any, tier: Tier) -> Tool:
    """Wrap one MCP tool description as a local ``Tool``."""
    name = tool_name(server, tool.name)

    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await session.call_tool(tool.name, args)
        except Exception as exc:
            raise ToolError(f"MCP server '{server}' failed: {exc}") from exc
        return ToolResult(result_text(result))

    schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
    return Tool(
        name=name,
        description=(tool.description or tool.name)[:1000],
        tier=tier,
        parameters=schema,
        run=run,
    )


class MCPManager:
    """Keeps sessions to the configured servers open for the bot's lifetime."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._stack = AsyncExitStack()
        self.tools: list[Tool] = []

    async def start(self) -> list[Tool]:
        """Connect to every server; a server that fails is skipped with a log."""
        for name, server in self._config.servers.items():
            try:
                self.tools.extend(await self._connect(name, server))
            except Exception:
                logger.exception("MCP server unavailable, skipping", extra={"server": name})
        return self.tools

    async def _connect(self, name: str, server: ServerConfig) -> list[Tool]:
        from mcp import ClientSession

        if server.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            if not server.command:
                raise ValueError("stdio server needs 'command'")
            params = StdioServerParameters(
                command=server.command, args=server.args, env=server.env or None
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif server.transport == "http":
            from mcp.client.streamable_http import streamable_http_client

            if not server.url:
                raise ValueError("http server needs 'url'")
            http_client = None
            if server.headers:
                import httpx2  # the MCP SDK's own HTTP client

                http_client = httpx2.AsyncClient(headers=server.headers, timeout=30.0)
            streams = await self._stack.enter_async_context(
                streamable_http_client(server.url, http_client=http_client)
            )
            read, write = streams[0], streams[1]
        else:
            raise ValueError(f"unknown transport '{server.transport}'")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tier = server.permission()
        tools = [tool_from_mcp(name, session, tool, tier) for tool in listed.tools]
        logger.info(
            "MCP server connected", extra={"server": name, "tools": [t.name for t in tools]}
        )
        return tools

    async def aclose(self) -> None:
        await self._stack.aclose()
