"""Tools over the bot's own memory: server history and per-user facts."""

from typing import Any

from bot.agent.tools.base import (
    Tier,
    Tool,
    ToolContext,
    ToolResult,
    object_schema,
    optional_int,
    require_str,
)
from bot.memory.rag import Retriever, jump_url
from bot.memory.user_facts import MAX_FACT_CHARS, UserFacts


def search_server_history_tool(retriever: Retriever) -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = require_str(args, "query", max_len=500)
        limit = optional_int(args, "max_results", 4, lo=1, hi=8)
        hits = await retriever.search(
            query, guild_id=ctx.guild_id, channel_id=ctx.channel_id, top_k=limit
        )
        if not hits:
            return ToolResult("No matching conversations found.")
        blocks = [
            f"[{index}] {hit.started_at:%Y-%m-%d %H:%M} UTC, {jump_url(hit)}\n{hit.content}"
            for index, hit in enumerate(hits[:limit], start=1)
        ]
        return ToolResult("\n\n".join(blocks), sources=tuple(jump_url(hit) for hit in hits[:limit]))

    return Tool(
        name="search_server_history",
        description=(
            "Search earlier conversations in this server (or this DM) by meaning. "
            "Returns excerpts with links to the original messages."
        ),
        tier=Tier.READ_ONLY,
        parameters=object_schema(
            {
                "query": {"type": "string", "description": "What to look for"},
                "max_results": {"type": "integer", "description": "1-8, default 4"},
            },
            required=["query"],
        ),
        run=run,
    )


def recall_memory_tool(facts: UserFacts) -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        known = await facts.view(ctx.user_id, ctx.guild_id)
        if not known:
            return ToolResult("Nothing is saved about this user here.")
        return ToolResult("\n".join(f"- {fact.fact}" for fact in known))

    return Tool(
        name="recall_memory",
        description="List the facts saved about the user you are talking to.",
        tier=Tier.READ_ONLY,
        run=run,
    )


def remember_tool(facts: UserFacts) -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        fact = require_str(args, "fact", max_len=MAX_FACT_CHARS)
        added = await facts.remember(ctx.user_id, ctx.guild_id, fact, ctx.message_id)
        return ToolResult("Saved." if added else "That was already saved.")

    return Tool(
        name="remember",
        description=(
            "Save a lasting fact about the user you are talking to, in their words, "
            "for future conversations. Only for facts they ask you to remember."
        ),
        tier=Tier.SIDE_EFFECT,
        parameters=object_schema(
            {"fact": {"type": "string", "description": "One short sentence about the user"}}
        ),
        run=run,
    )
