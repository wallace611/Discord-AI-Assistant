import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.agent.tools.base import ToolCallRequest, ToolError
from bot.gateway.actions import (
    CONFIRM_TIMEOUT,
    CONFIRMED_TEXT,
    DECLINED_TEXT,
    EXPIRED_TEXT,
    ConfirmView,
    MessageActions,
    describe_arguments,
)
from bot.gateway.streaming import NO_MENTIONS

USER_ID = 7
REQUEST = ToolCallRequest("create_thread", "Open a thread.", {"name": "Planning"})


def http_error(text: str = "Missing Permissions", status: int = 403) -> discord.HTTPException:
    response = MagicMock(status=status, reason="Forbidden")
    return discord.HTTPException(response, text)


def make_interaction(user_id: int) -> MagicMock:
    interaction = MagicMock()
    interaction.user.id = user_id
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    return interaction


def make_message(*, in_guild: bool = True) -> MagicMock:
    message = MagicMock()
    message.author.id = USER_ID
    if not in_guild:
        message.guild = None
    message.create_thread = AsyncMock()
    message.pin = AsyncMock()
    message.channel.send = AsyncMock()
    return message


# --- describe_arguments ----------------------------------------------------


def test_describe_arguments_lists_each_argument() -> None:
    assert describe_arguments({"name": "Planning", "hours": 3}) == "name='Planning', hours=3"


def test_describe_arguments_without_arguments() -> None:
    assert describe_arguments({}) == "(no arguments)"


def test_describe_arguments_truncates_long_text() -> None:
    text = describe_arguments({"code": "x" * 1000}, limit=50)

    assert len(text) == 50
    assert text.endswith("…")
    assert text.startswith("code='xxx")


def test_describe_arguments_keeps_text_at_the_limit() -> None:
    exact = describe_arguments({"a": 1})

    assert describe_arguments({"a": 1}, limit=len(exact)) == exact


# --- ConfirmView -----------------------------------------------------------


async def test_view_has_two_buttons_and_a_timeout() -> None:
    view = ConfirmView(USER_ID)

    assert view.timeout == CONFIRM_TIMEOUT
    assert view.approved is None
    assert [type(item) for item in view.children] == [discord.ui.Button, discord.ui.Button]


async def test_requesting_user_passes_the_interaction_check() -> None:
    view = ConfirmView(USER_ID)
    interaction = make_interaction(USER_ID)

    assert await view.interaction_check(interaction) is True
    interaction.response.send_message.assert_not_awaited()


async def test_other_users_are_refused_privately() -> None:
    view = ConfirmView(USER_ID)
    interaction = make_interaction(USER_ID + 1)

    assert await view.interaction_check(interaction) is False
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
    assert view.approved is None
    assert not view.is_finished()


async def test_confirm_button_approves_and_stops_the_view() -> None:
    view = ConfirmView(USER_ID)
    interaction = make_interaction(USER_ID)

    await view.confirm.callback(interaction)

    assert view.approved is True
    assert view.is_finished()
    interaction.response.defer.assert_awaited_once()


async def test_cancel_button_declines_and_stops_the_view() -> None:
    view = ConfirmView(USER_ID)
    interaction = make_interaction(USER_ID)

    await view.cancel.callback(interaction)

    assert view.approved is False
    assert view.is_finished()
    interaction.response.defer.assert_awaited_once()


# --- MessageActions.confirm ------------------------------------------------


def answer_prompt(message: MagicMock, approved: bool | None) -> MagicMock:
    """Make the confirmation prompt resolve as soon as it is sent."""
    prompt = MagicMock()
    prompt.edit = AsyncMock()

    async def send(content: str, *, view: ConfirmView, **kwargs: Any) -> MagicMock:
        view.approved = approved
        view.stop()
        return prompt

    message.channel.send = AsyncMock(side_effect=send)
    return prompt


@pytest.mark.parametrize(
    ("approved", "expected", "outcome"),
    [(True, True, CONFIRMED_TEXT), (False, False, DECLINED_TEXT), (None, False, EXPIRED_TEXT)],
)
async def test_confirm_reports_the_users_choice(
    approved: bool | None, expected: bool, outcome: str
) -> None:
    message = make_message()
    prompt = answer_prompt(message, approved)

    assert await MessageActions(message).confirm(REQUEST) is expected

    prompt.edit.assert_awaited_once_with(content=outcome.format(tool="create_thread"), view=None)


async def test_confirm_prompt_names_the_tool_and_arguments_without_mentions() -> None:
    message = make_message()
    answer_prompt(message, True)

    await MessageActions(message).confirm(REQUEST)

    call = message.channel.send.await_args
    assert "create_thread" in call.args[0]
    assert "name='Planning'" in call.args[0]
    assert call.kwargs["allowed_mentions"] is NO_MENTIONS
    assert isinstance(call.kwargs["view"], ConfirmView)


async def test_confirm_view_is_bound_to_the_message_author() -> None:
    message = make_message()
    answer_prompt(message, True)

    await MessageActions(message).confirm(REQUEST)

    view: ConfirmView = message.channel.send.await_args.kwargs["view"]
    assert await view.interaction_check(make_interaction(USER_ID)) is True
    assert await view.interaction_check(make_interaction(USER_ID + 1)) is False


async def test_confirm_survives_a_failed_prompt_update() -> None:
    message = make_message()
    prompt = answer_prompt(message, True)
    prompt.edit.side_effect = http_error("Unknown Message", status=404)

    assert await MessageActions(message).confirm(REQUEST) is True


# --- MessageActions side effects -------------------------------------------


async def test_create_thread_returns_the_jump_url() -> None:
    message = make_message()
    message.create_thread.return_value.jump_url = "https://discord.com/channels/1/2/30"

    url = await MessageActions(message).create_thread("Planning")

    assert url == "https://discord.com/channels/1/2/30"
    message.create_thread.assert_awaited_once_with(name="Planning")


async def test_create_thread_caps_the_name() -> None:
    message = make_message()

    await MessageActions(message).create_thread("n" * 250)

    message.create_thread.assert_awaited_once_with(name="n" * 100)


async def test_create_thread_is_refused_in_direct_messages() -> None:
    message = make_message(in_guild=False)

    with pytest.raises(ToolError, match="direct messages"):
        await MessageActions(message).create_thread("Planning")

    message.create_thread.assert_not_awaited()


async def test_create_thread_turns_discord_errors_into_tool_errors() -> None:
    message = make_message()
    message.create_thread.side_effect = http_error("Missing Permissions")

    with pytest.raises(ToolError, match="Missing Permissions") as excinfo:
        await MessageActions(message).create_thread("Planning")

    assert isinstance(excinfo.value.__cause__, discord.HTTPException)


async def test_pin_message_pins_the_users_message() -> None:
    message = make_message()

    await MessageActions(message).pin_message()

    message.pin.assert_awaited_once_with()


async def test_pin_message_turns_discord_errors_into_tool_errors() -> None:
    message = make_message()
    message.pin.side_effect = http_error("Maximum number of pins reached (50)")

    with pytest.raises(ToolError, match="Maximum number of pins"):
        await MessageActions(message).pin_message()


async def test_create_poll_sends_a_poll_without_mentions() -> None:
    message = make_message()
    message.channel.send.return_value.jump_url = "https://discord.com/channels/1/2/40"

    url = await MessageActions(message).create_poll("Lunch?", ["ramen", "curry"], 6)

    assert url == "https://discord.com/channels/1/2/40"
    call = message.channel.send.await_args
    assert call.kwargs["allowed_mentions"] is NO_MENTIONS
    poll: discord.Poll = call.kwargs["poll"]
    assert poll.question == "Lunch?"
    assert [answer.text for answer in poll.answers] == ["ramen", "curry"]
    assert poll.duration == datetime.timedelta(hours=6)


async def test_create_poll_turns_discord_errors_into_tool_errors() -> None:
    message = make_message()
    message.channel.send.side_effect = http_error("Cannot send polls in this channel")

    with pytest.raises(ToolError, match="Cannot send polls"):
        await MessageActions(message).create_poll("Lunch?", ["ramen", "curry"], 6)
