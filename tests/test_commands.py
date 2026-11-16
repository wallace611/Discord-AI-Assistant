from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from discord import app_commands

from bot.gateway.commands import (
    ALL_FORGOTTEN,
    FACT_FORGOTTEN,
    FACT_NOT_FOUND,
    FORGET_USAGE,
    NO_FACTS,
    TRUNCATED,
    register_memory_commands,
    render_facts,
)
from bot.gateway.streaming import DISCORD_MESSAGE_LIMIT, NO_MENTIONS
from bot.memory import InMemoryStore, UserFact
from bot.memory.user_facts import UserFacts
from fakes import T0, FakeProvider


def make_interaction(user_id: int = 7, guild_id: int | None = 200) -> Any:
    return NS(user=NS(id=user_id), guild_id=guild_id, response=NS(send_message=AsyncMock()))


def sent(interaction: Any) -> str:
    """Text of the single response, after checking it is private and mention-free."""
    interaction.response.send_message.assert_awaited_once()
    call = interaction.response.send_message.await_args
    assert call.kwargs == {"ephemeral": True, "allowed_mentions": NO_MENTIONS}
    return call.args[0]


async def make_commands() -> tuple[InMemoryStore, Any, Any]:
    store = InMemoryStore()
    tree = MagicMock()
    register_memory_commands(tree, UserFacts(store, FakeProvider()))
    group = tree.add_command.call_args.args[0]
    assert isinstance(group, app_commands.Group)
    view = group.get_command("view")
    forget = group.get_command("forget")
    assert view is not None and forget is not None
    return store, view.callback, forget.callback


def test_render_facts_lists_ids_and_fits_in_one_message() -> None:
    assert render_facts([]) == NO_FACTS

    text = render_facts([UserFact(3, 7, 200, "Likes Rust.", T0), UserFact(9, 7, 200, "Tea.", T0)])
    assert "`3` Likes Rust." in text
    assert "`9` Tea." in text

    many = [UserFact(i, 7, 200, "x" * 150, T0) for i in range(50)]
    long_text = render_facts(many)
    assert len(long_text) <= DISCORD_MESSAGE_LIMIT
    assert TRUNCATED in long_text


async def test_view_shows_only_the_callers_facts_in_this_guild() -> None:
    store, view, _ = await make_commands()
    await store.add_fact(7, 200, "Alice writes Rust.", None)
    await store.add_fact(7, 201, "Alice in another guild.", None)
    await store.add_fact(8, 200, "Bob writes Go.", None)
    interaction = make_interaction()

    await view(interaction)

    text = sent(interaction)
    assert "Alice writes Rust." in text
    assert "another guild" not in text
    assert "Bob" not in text


async def test_view_with_nothing_stored() -> None:
    _, view, _ = await make_commands()
    interaction = make_interaction()
    await view(interaction)
    assert sent(interaction) == NO_FACTS


async def test_forget_one_fact() -> None:
    store, _, forget = await make_commands()
    await store.add_fact(7, 200, "Alice writes Rust.", None)
    [fact] = await store.list_facts(7, 200)
    interaction = make_interaction()

    await forget(interaction, fact_id=fact.fact_id)

    assert sent(interaction) == FACT_FORGOTTEN.format(fact_id=fact.fact_id)
    assert await store.list_facts(7, 200) == []


async def test_forget_cannot_touch_another_users_fact() -> None:
    store, _, forget = await make_commands()
    await store.add_fact(8, 200, "Bob writes Go.", None)
    [fact] = await store.list_facts(8, 200)
    interaction = make_interaction(user_id=7)

    await forget(interaction, fact_id=fact.fact_id)

    assert sent(interaction) == FACT_NOT_FOUND.format(fact_id=fact.fact_id)
    assert len(await store.list_facts(8, 200)) == 1


async def test_forget_everything_is_limited_to_this_guild() -> None:
    store, _, forget = await make_commands()
    await store.add_fact(7, 200, "One.", None)
    await store.add_fact(7, 200, "Two.", None)
    await store.add_fact(7, 201, "Elsewhere.", None)
    interaction = make_interaction()

    await forget(interaction, everything=True)

    assert sent(interaction) == ALL_FORGOTTEN.format(count=2)
    assert await store.list_facts(7, 200) == []
    assert len(await store.list_facts(7, 201)) == 1


async def test_forget_without_arguments_explains_usage_and_deletes_nothing() -> None:
    store, _, forget = await make_commands()
    await store.add_fact(7, 200, "One.", None)
    interaction = make_interaction()

    await forget(interaction)

    assert sent(interaction) == FORGET_USAGE
    assert len(await store.list_facts(7, 200)) == 1
