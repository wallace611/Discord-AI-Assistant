"""Side-effect tools on the Discord conversation. All need user confirmation."""

from typing import Any

from bot.agent.tools.base import (
    DiscordActions,
    Tier,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    object_schema,
    optional_int,
    require_str,
)

MAX_POLL_OPTIONS = 10


def _actions(ctx: ToolContext) -> DiscordActions:
    if ctx.discord is None:
        raise ToolError("this action is not available here")
    return ctx.discord


def create_thread_tool() -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = require_str(args, "name", max_len=100)
        url = await _actions(ctx).create_thread(name)
        return ToolResult(f"Thread created: {url}", sources=(url,))

    return Tool(
        name="create_thread",
        description="Open a thread on the user's message, for long or multi-step discussions.",
        tier=Tier.SIDE_EFFECT,
        parameters=object_schema({"name": {"type": "string", "description": "Thread title"}}),
        run=run,
    )


def pin_message_tool() -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await _actions(ctx).pin_message()
        return ToolResult("Pinned the user's message.")

    return Tool(
        name="pin_message",
        description="Pin the message the user sent you.",
        tier=Tier.SIDE_EFFECT,
        run=run,
    )


def create_poll_tool() -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = require_str(args, "question", max_len=300)
        raw_options = args.get("options")
        if not isinstance(raw_options, list):
            raise ToolError("'options' must be a list of strings")
        options = [str(option).strip()[:55] for option in raw_options if str(option).strip()]
        if not 2 <= len(options) <= MAX_POLL_OPTIONS:
            raise ToolError(f"a poll needs 2 to {MAX_POLL_OPTIONS} options")
        hours = optional_int(args, "hours", 24, lo=1, hi=24 * 7)
        url = await _actions(ctx).create_poll(question, options, hours)
        return ToolResult(f"Poll posted: {url}", sources=(url,))

    return Tool(
        name="create_poll",
        description="Post a poll in this channel.",
        tier=Tier.SIDE_EFFECT,
        parameters=object_schema(
            {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}, "description": "2-10"},
                "hours": {"type": "integer", "description": "How long it stays open, 1-168"},
            },
            required=["question", "options"],
        ),
        run=run,
    )
