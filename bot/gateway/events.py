"""Conversion from raw Discord messages to the bot's internal types."""

import logging
import re
from dataclasses import dataclass

import discord

from bot.gateway.streaming import NO_MENTIONS, Editable, SendFn
from bot.memory.store import StoredMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A user message addressed to the bot, detached from discord.py objects."""

    message_id: int
    text: str
    user_id: int
    author_name: str
    channel_id: int
    guild_id: int | None
    image_count: int
    bot_id: int
    bot_name: str
    send: SendFn  # posts a reply chunk and returns the created message


def is_addressed(message: discord.Message, bot_id: int) -> bool:
    """True for DMs, direct @mentions of the bot, and replies to the bot."""
    if message.guild is None:
        return True
    if any(user.id == bot_id for user in message.mentions):
        return True
    replied_to = message.reference.resolved if message.reference else None
    return isinstance(replied_to, discord.Message) and replied_to.author.id == bot_id


def strip_bot_mention(content: str, bot_id: int) -> str:
    return re.sub(rf"<@!?{bot_id}>", "", content).strip()


def count_images(message: discord.Message) -> int:
    return sum(
        1
        for attachment in message.attachments
        if (attachment.content_type or "").startswith("image/")
    )


def make_sender(message: discord.Message) -> SendFn:
    """First chunk replies to ``message``; later chunks go to the same channel."""
    is_first = True

    async def send(content: str) -> Editable:
        nonlocal is_first
        if is_first:
            is_first = False
            try:
                return await message.reply(
                    content, mention_author=False, allowed_mentions=NO_MENTIONS
                )
            except discord.HTTPException:
                # Usually the original message was deleted; use a plain send instead.
                logger.debug("reply failed, sending to channel instead", exc_info=True)
        return await message.channel.send(content, allowed_mentions=NO_MENTIONS)

    return send


def to_event(
    message: discord.Message, bot_id: int, bot_name: str = "assistant"
) -> IncomingMessage | None:
    """Build an event, or None when the message is not for the bot."""
    if message.author.bot or not is_addressed(message, bot_id):
        return None
    text = strip_bot_mention(message.content, bot_id)
    image_count = count_images(message)
    if not text and image_count == 0:
        return None
    return IncomingMessage(
        message_id=message.id,
        text=text,
        user_id=message.author.id,
        author_name=message.author.display_name,
        channel_id=message.channel.id,
        guild_id=message.guild.id if message.guild else None,
        image_count=image_count,
        bot_id=bot_id,
        bot_name=bot_name,
        send=make_sender(message),
    )


def to_stored(message: discord.Message) -> StoredMessage | None:
    """Build the record kept in memory, or None when there is nothing to keep.

    Other bots are skipped. This bot's own replies are recorded by the handler
    once they are complete, not from the gateway event that carries a partial one.
    """
    if message.author.bot:
        return None
    # clean_content turns <@id> mentions into readable @names.
    content = message.clean_content.strip()
    if not content:
        return None
    return StoredMessage(
        message_id=message.id,
        guild_id=message.guild.id if message.guild else None,
        channel_id=message.channel.id,
        author_id=message.author.id,
        author_name=message.author.display_name,
        is_bot=False,
        content=content,
        created_at=message.created_at,
    )
