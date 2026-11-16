"""Slash commands. Replies are ephemeral: only the invoking user sees them."""

from collections.abc import Sequence

import discord
from discord import app_commands

from bot.gateway.streaming import DISCORD_MESSAGE_LIMIT, NO_MENTIONS
from bot.memory.store import UserFact
from bot.memory.user_facts import UserFacts

NO_FACTS = "我在這裡沒有記住任何關於你的事。"
FORGET_USAGE = (
    "請指定要刪除的編號（`fact_id`），或把 `everything` 設為 True 來全部刪除。"
    "編號可以用 `/memory view` 查看。"
)
FACT_NOT_FOUND = "找不到編號 {fact_id} 的記憶。用 `/memory view` 確認編號。"
FACT_FORGOTTEN = "已刪除編號 {fact_id} 的記憶。"
ALL_FORGOTTEN = "已刪除我在這裡記住的全部 {count} 筆記憶。"
VIEW_HEADER = "我在這裡記住了關於你的 {count} 件事："
VIEW_FOOTER = "用 `/memory forget` 刪除單筆或全部。"
TRUNCATED = "…（太長，未全部顯示）"


def render_facts(facts: Sequence[UserFact]) -> str:
    """Format facts for ``/memory view``, within the Discord message limit."""
    if not facts:
        return NO_FACTS
    header = VIEW_HEADER.format(count=len(facts))
    budget = DISCORD_MESSAGE_LIMIT - len(header) - len(VIEW_FOOTER) - len(TRUNCATED) - 4
    lines: list[str] = []
    for fact in facts:
        line = f"`{fact.fact_id}` {fact.fact}"
        if budget - len(line) - 1 < 0:
            lines.append(TRUNCATED)
            break
        budget -= len(line) + 1
        lines.append(line)
    return "\n".join([header, *lines, "", VIEW_FOOTER])


async def _respond(interaction: discord.Interaction, content: str) -> None:
    await interaction.response.send_message(content, ephemeral=True, allowed_mentions=NO_MENTIONS)


def register_memory_commands(tree: app_commands.CommandTree, facts: UserFacts) -> None:
    """Add the ``/memory`` command group to ``tree``."""
    group = app_commands.Group(name="memory", description="查看或刪除 bot 記住的關於你的事")

    @group.command(name="view", description="查看 bot 在這個伺服器記住的關於你的事")
    async def view(interaction: discord.Interaction) -> None:
        found = await facts.view(interaction.user.id, interaction.guild_id)
        await _respond(interaction, render_facts(found))

    @group.command(name="forget", description="刪除 bot 記住的關於你的事")
    @app_commands.describe(
        fact_id="要刪除的記憶編號（見 /memory view）",
        everything="刪除在這個伺服器的全部記憶",
    )
    async def forget(
        interaction: discord.Interaction,
        fact_id: int | None = None,
        everything: bool = False,
    ) -> None:
        if everything:
            count = await facts.forget_all(interaction.user.id, interaction.guild_id)
            await _respond(interaction, ALL_FORGOTTEN.format(count=count))
        elif fact_id is None:
            await _respond(interaction, FORGET_USAGE)
        elif await facts.forget(interaction.user.id, fact_id):
            await _respond(interaction, FACT_FORGOTTEN.format(fact_id=fact_id))
        else:
            await _respond(interaction, FACT_NOT_FOUND.format(fact_id=fact_id))

    tree.add_command(group)
