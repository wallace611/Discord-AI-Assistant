"""Discord side effects for tools, and the confirmation prompt that gates them."""

import datetime
import logging

import discord

from bot.agent.tools.base import ToolCallRequest, ToolError
from bot.gateway.streaming import NO_MENTIONS

logger = logging.getLogger(__name__)

CONFIRM_TIMEOUT = 60.0
CONFIRM_TEXT = "🤖 我想執行 **{tool}**：{summary}\n要繼續嗎？"
CONFIRMED_TEXT = "✅ 已確認：{tool}"
DECLINED_TEXT = "❌ 已取消：{tool}"
EXPIRED_TEXT = "⌛ 未在時限內確認，已取消：{tool}"


def describe_arguments(arguments: dict[str, object], limit: int = 300) -> str:
    """Compact, human-readable rendering of the tool arguments."""
    parts = [f"{key}={value!r}" for key, value in arguments.items()]
    text = ", ".join(parts) or "(no arguments)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConfirmView(discord.ui.View):
    """Two buttons that only the requesting user may press."""

    def __init__(self, user_id: int) -> None:
        super().__init__(timeout=CONFIRM_TIMEOUT)
        self._user_id = user_id
        self.approved: bool | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self._user_id:
            return True
        await interaction.response.send_message("這個確認只有發問的人可以按。", ephemeral=True)
        return False

    @discord.ui.button(label="執行", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.approved = True
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.approved = False
        await interaction.response.defer()
        self.stop()


class MessageActions:
    """``DiscordActions`` and ``Confirmer`` bound to one incoming message."""

    def __init__(self, message: discord.Message) -> None:
        self._message = message

    async def confirm(self, request: ToolCallRequest) -> bool:
        view = ConfirmView(self._message.author.id)
        prompt = await self._message.channel.send(
            CONFIRM_TEXT.format(tool=request.tool, summary=describe_arguments(request.arguments)),
            view=view,
            allowed_mentions=NO_MENTIONS,
        )
        await view.wait()
        outcome = {True: CONFIRMED_TEXT, False: DECLINED_TEXT, None: EXPIRED_TEXT}[view.approved]
        try:
            await prompt.edit(content=outcome.format(tool=request.tool), view=None)
        except discord.HTTPException:
            logger.debug("could not update confirmation prompt", exc_info=True)
        return view.approved is True

    async def create_thread(self, name: str) -> str:
        if self._message.guild is None:
            raise ToolError("threads cannot be created in direct messages")
        try:
            thread = await self._message.create_thread(name=name[:100])
        except discord.HTTPException as exc:
            raise ToolError(f"Discord refused to create the thread: {exc.text}") from exc
        return thread.jump_url

    async def pin_message(self) -> None:
        try:
            await self._message.pin()
        except discord.HTTPException as exc:
            raise ToolError(f"Discord refused to pin the message: {exc.text}") from exc

    async def create_poll(self, question: str, options: list[str], hours: int) -> str:
        poll = discord.Poll(question=question, duration=datetime.timedelta(hours=hours))
        for option in options:
            poll.add_answer(text=option)
        try:
            sent = await self._message.channel.send(poll=poll, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as exc:
            raise ToolError(f"Discord refused to create the poll: {exc.text}") from exc
        return sent.jump_url
