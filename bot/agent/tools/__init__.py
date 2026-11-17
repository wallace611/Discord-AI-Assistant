"""Tool interface and built-in tools. ``bot.agent.setup`` assembles the registry."""

from bot.agent.tools.base import (
    Confirmer,
    DiscordActions,
    Tier,
    Tool,
    ToolCallRequest,
    ToolContext,
    ToolError,
    ToolResult,
)

__all__ = [
    "Confirmer",
    "DiscordActions",
    "Tier",
    "Tool",
    "ToolCallRequest",
    "ToolContext",
    "ToolError",
    "ToolResult",
]
