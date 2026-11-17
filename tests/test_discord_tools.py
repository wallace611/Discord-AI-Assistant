from typing import Any

import pytest

from bot.agent.tools.base import Tier, Tool, ToolContext, ToolError
from bot.agent.tools.discord_tools import (
    MAX_POLL_OPTIONS,
    create_poll_tool,
    create_thread_tool,
    pin_message_tool,
)

THREAD_URL = "https://discord.com/channels/1/2/30"
POLL_URL = "https://discord.com/channels/1/2/40"


class FakeDiscord:
    """Records the side effects a tool asked for."""

    def __init__(self) -> None:
        self.threads: list[str] = []
        self.pins = 0
        self.polls: list[tuple[str, list[str], int]] = []

    async def create_thread(self, name: str) -> str:
        self.threads.append(name)
        return THREAD_URL

    async def pin_message(self) -> None:
        self.pins += 1

    async def create_poll(self, question: str, options: list[str], hours: int) -> str:
        self.polls.append((question, options, hours))
        return POLL_URL


def ctx(discord: FakeDiscord | None) -> ToolContext:
    return ToolContext(guild_id=1, channel_id=2, user_id=3, message_id=4, discord=discord)


def poll_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"question": "Lunch?", "options": ["ramen", "curry"]}
    args.update(overrides)
    return args


ALL_TOOLS = [create_thread_tool, pin_message_tool, create_poll_tool]
VALID_ARGS: dict[str, dict[str, Any]] = {
    "create_thread": {"name": "Planning"},
    "pin_message": {},
    "create_poll": poll_args(),
}


@pytest.mark.parametrize("factory", ALL_TOOLS)
def test_every_tool_needs_confirmation(factory: Any) -> None:
    tool: Tool = factory()

    assert tool.tier is Tier.SIDE_EFFECT


def test_tool_names_are_stable() -> None:
    assert [factory().name for factory in ALL_TOOLS] == list(VALID_ARGS)


@pytest.mark.parametrize("factory", ALL_TOOLS)
async def test_tools_fail_cleanly_outside_discord(factory: Any) -> None:
    tool: Tool = factory()

    with pytest.raises(ToolError, match="not available"):
        await tool.run(VALID_ARGS[tool.name], ctx(None))


# --- create_thread ---------------------------------------------------------


async def test_create_thread_returns_the_link_as_a_source() -> None:
    discord = FakeDiscord()

    result = await create_thread_tool().run({"name": "  Planning  "}, ctx(discord))

    assert discord.threads == ["Planning"]
    assert THREAD_URL in result.content
    assert result.sources == (THREAD_URL,)


async def test_create_thread_caps_the_title_at_discords_limit() -> None:
    discord = FakeDiscord()

    await create_thread_tool().run({"name": "n" * 500}, ctx(discord))

    assert discord.threads == ["n" * 100]


@pytest.mark.parametrize("args", [{}, {"name": ""}, {"name": "   "}, {"name": 7}])
async def test_create_thread_needs_a_name(args: dict[str, Any]) -> None:
    discord = FakeDiscord()

    with pytest.raises(ToolError, match="name"):
        await create_thread_tool().run(args, ctx(discord))

    assert discord.threads == []


def test_create_thread_schema_requires_name() -> None:
    parameters = create_thread_tool().spec()["function"]["parameters"]

    assert parameters["required"] == ["name"]
    assert parameters["additionalProperties"] is False


# --- pin_message -----------------------------------------------------------


async def test_pin_message_pins_once() -> None:
    discord = FakeDiscord()

    result = await pin_message_tool().run({}, ctx(discord))

    assert discord.pins == 1
    assert "Pinned" in result.content
    assert result.sources == ()


def test_pin_message_takes_no_arguments() -> None:
    parameters = pin_message_tool().spec()["function"]["parameters"]

    assert parameters["properties"] == {}


# --- create_poll -----------------------------------------------------------


async def test_create_poll_posts_and_returns_the_link() -> None:
    discord = FakeDiscord()

    result = await create_poll_tool().run(poll_args(hours=6), ctx(discord))

    assert discord.polls == [("Lunch?", ["ramen", "curry"], 6)]
    assert POLL_URL in result.content
    assert result.sources == (POLL_URL,)


async def test_create_poll_defaults_to_one_day() -> None:
    discord = FakeDiscord()

    await create_poll_tool().run(poll_args(), ctx(discord))

    assert discord.polls[0][2] == 24


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(-5, 1), (0, 1), (1, 1), (72, 72), (168, 168), (169, 168), (10_000, 168)],
)
async def test_create_poll_clamps_hours_to_one_week(hours: int, expected: int) -> None:
    discord = FakeDiscord()

    await create_poll_tool().run(poll_args(hours=hours), ctx(discord))

    assert discord.polls[0][2] == expected


@pytest.mark.parametrize("hours", ["24", 2.5, True, None])
async def test_create_poll_rejects_non_integer_hours(hours: Any) -> None:
    discord = FakeDiscord()

    with pytest.raises(ToolError, match="hours"):
        await create_poll_tool().run(poll_args(hours=hours), ctx(discord))

    assert discord.polls == []


@pytest.mark.parametrize("count", [2, MAX_POLL_OPTIONS])
async def test_create_poll_accepts_two_to_ten_options(count: int) -> None:
    discord = FakeDiscord()
    options = [f"option {i}" for i in range(count)]

    await create_poll_tool().run(poll_args(options=options), ctx(discord))

    assert discord.polls[0][1] == options


@pytest.mark.parametrize("count", [0, 1, MAX_POLL_OPTIONS + 1])
async def test_create_poll_rejects_too_few_or_too_many_options(count: int) -> None:
    discord = FakeDiscord()
    options = [f"option {i}" for i in range(count)]

    with pytest.raises(ToolError, match="2 to 10 options"):
        await create_poll_tool().run(poll_args(options=options), ctx(discord))

    assert discord.polls == []


async def test_create_poll_ignores_blank_options_when_counting() -> None:
    discord = FakeDiscord()

    with pytest.raises(ToolError, match="2 to 10 options"):
        await create_poll_tool().run(poll_args(options=["ramen", "  ", ""]), ctx(discord))

    assert discord.polls == []


async def test_create_poll_trims_and_caps_option_text() -> None:
    discord = FakeDiscord()

    await create_poll_tool().run(poll_args(options=["  ramen ", "c" * 200]), ctx(discord))

    assert discord.polls[0][1] == ["ramen", "c" * 55]


@pytest.mark.parametrize("options", [None, "ramen,curry", {"a": 1}, 3])
async def test_create_poll_needs_a_list_of_options(options: Any) -> None:
    discord = FakeDiscord()

    with pytest.raises(ToolError, match="options"):
        await create_poll_tool().run(poll_args(options=options), ctx(discord))

    assert discord.polls == []


@pytest.mark.parametrize("question", [None, "", "  "])
async def test_create_poll_needs_a_question(question: Any) -> None:
    discord = FakeDiscord()

    with pytest.raises(ToolError, match="question"):
        await create_poll_tool().run(poll_args(question=question), ctx(discord))

    assert discord.polls == []


async def test_create_poll_caps_the_question_at_discords_limit() -> None:
    discord = FakeDiscord()

    await create_poll_tool().run(poll_args(question="q" * 1000), ctx(discord))

    assert discord.polls[0][0] == "q" * 300


def test_create_poll_schema_makes_hours_optional() -> None:
    parameters = create_poll_tool().spec()["function"]["parameters"]

    assert parameters["required"] == ["question", "options"]
    assert set(parameters["properties"]) == {"question", "options", "hours"}
