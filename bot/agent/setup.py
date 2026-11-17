"""Assembles the tool registry from settings."""

import logging
from pathlib import Path

from bot.agent.registry import ToolRegistry
from bot.agent.tools.discord_tools import create_poll_tool, create_thread_tool, pin_message_tool
from bot.agent.tools.fetch_url import fetch_url_tool
from bot.agent.tools.history import recall_memory_tool, remember_tool, search_server_history_tool
from bot.agent.tools.mcp import MCPManager, load_config
from bot.agent.tools.run_code import (
    CommandRunner,
    SandboxConfig,
    remove_stale_sandboxes,
    run_code_tool,
    run_command,
    verify_sandbox,
)
from bot.agent.tools.web_search import web_search_tool
from bot.config import Settings
from bot.memory import MemoryService

logger = logging.getLogger(__name__)


def build_registry(settings: Settings, memory: MemoryService) -> ToolRegistry:
    """The built-in tools. The sandbox and MCP tools are added by the async setup steps."""
    return ToolRegistry(
        [
            web_search_tool(settings.searxng_url),
            fetch_url_tool(),
            search_server_history_tool(memory.retriever),
            recall_memory_tool(memory.facts),
            remember_tool(memory.facts),
            create_thread_tool(),
            pin_message_tool(),
            create_poll_tool(),
        ]
    )


async def add_sandbox(
    settings: Settings, registry: ToolRegistry, runner: CommandRunner = run_command
) -> bool:
    """Offer ``run_code`` only when the sandbox profile is configured and offline."""
    if not settings.incus_sandbox_profile:
        return False
    config = SandboxConfig(
        image=settings.incus_image,
        profile=settings.incus_sandbox_profile,
        timeout=settings.run_code_timeout_seconds,
    )
    problem = await verify_sandbox(config, runner)
    if problem is not None:
        logger.error("run_code disabled", extra={"reason": problem})
        return False
    try:
        await remove_stale_sandboxes(runner)
    except Exception:
        logger.exception("could not list stale sandboxes")
    registry.add(run_code_tool(config, runner))
    return True


async def start_mcp(settings: Settings, registry: ToolRegistry) -> MCPManager | None:
    """Connect the MCP servers from ``MCP_CONFIG_PATH`` and register their tools."""
    path = Path(settings.mcp_config_path) if settings.mcp_config_path else None
    if path is None or not path.exists():
        return None
    manager = MCPManager(load_config(path))
    for tool in await manager.start():
        if registry.get(tool.name) is not None:
            logger.warning("MCP tool name clashes, skipped", extra={"tool": tool.name})
            continue
        registry.add(tool)
    return manager
