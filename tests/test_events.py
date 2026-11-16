from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
from discord import app_commands

from bot.gateway.client import BotClient
from bot.gateway.events import strip_bot_mention, to_event, to_stored
from bot.gateway.streaming import NO_MENTIONS
from bot.memory import InMemoryStore, MemoryService
from fakes import T0, FakeProvider, stored

BOT_ID = 42


def make_message(
    content: str = "hello",
    *,
    in_guild: bool = True,
    mentions_bot: bool = False,
    author_is_bot: bool = False,
    reply_to_author_id: int | None = None,
    attachments: list[Any] | None = None,
) -> Any:
    reference = None
    if reply_to_author_id is not None:
        resolved = MagicMock(spec=discord.Message)
        resolved.author = NS(id=reply_to_author_id)
        reference = NS(resolved=resolved)
    return NS(
        id=1,
        content=content,
        clean_content=content.replace(f"<@{BOT_ID}>", "@bot"),
        created_at=T0,
        author=NS(id=7, bot=author_is_bot, display_name="alice"),
        guild=NS(id=200) if in_guild else None,
        channel=NS(id=100, send=AsyncMock(return_value="sent")),
        mentions=[NS(id=BOT_ID)] if mentions_bot else [NS(id=999)],
        reference=reference,
        attachments=attachments or [],
        reply=AsyncMock(return_value="replied"),
    )


def test_strip_bot_mention_handles_both_mention_forms() -> None:
    assert strip_bot_mention(f"<@{BOT_ID}> hi", BOT_ID) == "hi"
    assert strip_bot_mention(f"hi <@!{BOT_ID}>", BOT_ID) == "hi"
    assert strip_bot_mention("<@999> hi", BOT_ID) == "<@999> hi"


def test_guild_message_without_mention_is_ignored() -> None:
    assert to_event(make_message(), BOT_ID) is None


def test_mention_creates_event_with_cleaned_text() -> None:
    event = to_event(make_message(f"<@{BOT_ID}> what is up", mentions_bot=True), BOT_ID)
    assert event is not None
    assert event.text == "what is up"
    assert (event.user_id, event.channel_id, event.guild_id) == (7, 100, 200)
    assert event.image_count == 0


def test_dm_is_always_addressed() -> None:
    event = to_event(make_message("hi", in_guild=False), BOT_ID)
    assert event is not None
    assert event.guild_id is None


def test_reply_to_bot_is_addressed_but_reply_to_others_is_not() -> None:
    assert to_event(make_message(reply_to_author_id=BOT_ID), BOT_ID) is not None
    assert to_event(make_message(reply_to_author_id=555), BOT_ID) is None


def test_messages_from_bots_are_ignored() -> None:
    assert to_event(make_message(in_guild=False, author_is_bot=True), BOT_ID) is None


def test_bare_mention_is_ignored_unless_it_has_an_image() -> None:
    bare = make_message(f"<@{BOT_ID}>", mentions_bot=True)
    assert to_event(bare, BOT_ID) is None

    with_image = make_message(
        f"<@{BOT_ID}>",
        mentions_bot=True,
        attachments=[
            NS(content_type="image/png"),
            NS(content_type="application/pdf"),
            NS(content_type=None),
        ],
    )
    event = to_event(with_image, BOT_ID)
    assert event is not None
    assert event.image_count == 1


async def test_sender_replies_first_then_sends_to_channel_without_mentions() -> None:
    message = make_message(in_guild=False)
    event = to_event(message, BOT_ID)
    assert event is not None

    assert await event.send("one") == "replied"
    assert await event.send("two") == "sent"

    message.reply.assert_awaited_once_with(
        "one", mention_author=False, allowed_mentions=NO_MENTIONS
    )
    message.channel.send.assert_awaited_once_with("two", allowed_mentions=NO_MENTIONS)


async def test_sender_falls_back_to_channel_when_reply_fails() -> None:
    message = make_message(in_guild=False)
    message.reply.side_effect = discord.HTTPException(MagicMock(status=400), "unknown message")
    event = to_event(message, BOT_ID)
    assert event is not None

    assert await event.send("one") == "sent"


def test_event_carries_identity_of_author_and_bot() -> None:
    event = to_event(make_message("hi", in_guild=False), BOT_ID, "helper")
    assert event is not None
    assert (event.message_id, event.author_name) == (1, "alice")
    assert (event.bot_id, event.bot_name) == (BOT_ID, "helper")


def test_to_stored_keeps_every_human_message_with_readable_mentions() -> None:
    saved = to_stored(make_message(f"<@{BOT_ID}> what is up", mentions_bot=True))
    assert saved == stored(1, "@bot what is up")

    # Messages that are not addressed to the bot are stored too.
    assert to_stored(make_message("just chatting")) is not None


def test_to_stored_skips_bots_and_empty_messages() -> None:
    assert to_stored(make_message(author_is_bot=True)) is None
    assert to_stored(make_message("   ")) is None


def test_dm_is_stored_without_a_guild() -> None:
    saved = to_stored(make_message("secret", in_guild=False))
    assert saved is not None
    assert saved.guild_id is None


def test_client_disables_mentions_and_registers_memory_commands() -> None:
    provider = FakeProvider()
    memory = MemoryService(InMemoryStore(), chat=provider, embedder=provider)
    client = BotClient(MagicMock(), memory)

    assert client.intents.message_content is True
    assert client.allowed_mentions is not None
    assert client.allowed_mentions.to_dict() == NO_MENTIONS.to_dict()
    group = client.tree.get_command("memory")
    assert isinstance(group, app_commands.Group)
    assert {command.name for command in group.commands} == {"view", "forget"}
