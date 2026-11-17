from typing import Any

from bot.agent import AGENT_PROMPT, AgentLoop, ToolRegistry
from bot.agent.tools.base import Tier, Tool, ToolContext, ToolResult
from bot.gateway.events import IncomingMessage
from bot.gateway.handler import (
    BUDGET_NOTICE,
    ERROR_NOTICE,
    MEDIA_NOT_READY,
    REFERENCE_NOTE,
    SYSTEM_PROMPT,
    TOOL_STATUS,
    MessageHandler,
    build_messages,
)
from bot.memory import ChunkHit, InMemoryStore, MemoryContext, MemoryService, UserFact
from bot.providers.base import ProviderError, Usage
from bot.router import DailyTokenBudget, Router
from fakes import T0, FakeChannel, FakeProvider, stored, text_stream, tool_call_stream


def route_json(tier: str) -> str:
    return f'{{"intent": "chat", "tier": "{tier}", "needs_vision": false}}'


def make_event(channel: FakeChannel, text: str = "hello", user_id: int = 7) -> IncomingMessage:
    return IncomingMessage(
        message_id=5000,
        text=text,
        user_id=user_id,
        author_name="alice",
        channel_id=100,
        guild_id=200,
        image_count=0,
        bot_id=42,
        bot_name="bot",
        send=channel.send,
    )


class Harness:
    """A handler wired to fake providers and an in-memory store."""

    def __init__(
        self,
        local: FakeProvider,
        cloud: FakeProvider | None = None,
        budget: DailyTokenBudget | None = None,
        memory_provider: FakeProvider | None = None,
    ) -> None:
        self.store = InMemoryStore()
        self.local = local
        self.cloud = cloud or FakeProvider(name="cloud")
        self.budget = budget or DailyTokenBudget(None)
        self.memory_provider = memory_provider or FakeProvider(name="memory")
        self.memory = MemoryService(
            self.store, chat=self.memory_provider, embedder=self.memory_provider
        )
        self.channel = FakeChannel()
        self.handler = MessageHandler(
            router=Router(local),
            local=local,
            cloud=self.cloud,
            budget=self.budget,
            memory=self.memory,
            edit_interval=0.0,
        )

    async def run(self, text: str = "hello") -> None:
        await self.handler.handle(make_event(self.channel, text))


async def test_local_tier_streams_from_local_provider() -> None:
    local = FakeProvider(name="local", reply=route_json("local"), chunks=["Hi", " there"])
    harness = Harness(local, FakeProvider(name="cloud", chunks=["should not be used"]))

    await harness.run()

    assert harness.channel.texts == ["Hi there"]
    assert harness.cloud.stream_calls == []
    assert local.stream_calls[0]["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "alice: hello"},
    ]


async def test_routing_and_memory_lookup_run_concurrently() -> None:
    import asyncio

    started: list[str] = []
    both_started = asyncio.Event()

    class SlowProvider(FakeProvider):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            started.append("router")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return await super().complete(messages, **kwargs)

        async def embed(self, texts: Any, **kwargs: Any) -> Any:
            started.append("embed")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return await super().embed(texts, **kwargs)

    local = SlowProvider(name="local", reply=route_json("local"), chunks=["ok"])
    harness = Harness(local, memory_provider=SlowProvider(name="memory"))

    await harness.run()

    # If they ran one after the other the timeout would have fired instead.
    assert sorted(started) == ["embed", "router"]
    assert harness.channel.texts == ["ok"]


async def test_cloud_tier_streams_from_cloud_and_records_usage() -> None:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"))
    cloud = FakeProvider(name="cloud", chunks=["Answer"], usage=Usage(30, 12, 42))
    harness = Harness(local, cloud, DailyTokenBudget(1000))

    await harness.run()

    assert harness.channel.texts == ["Answer"]
    assert local.stream_calls == []
    assert harness.budget.used(7) == 42


async def test_local_usage_does_not_count_against_cloud_budget() -> None:
    local = FakeProvider(
        name="local", reply=route_json("local"), chunks=["Hi"], usage=Usage(5, 5, 10)
    )
    harness = Harness(local, budget=DailyTokenBudget(1000))

    await harness.run()

    assert harness.budget.used(7) == 0


async def test_exhausted_budget_downgrades_to_local_with_notice() -> None:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"), chunks=["Local answer"])
    budget = DailyTokenBudget(100)
    budget.record(7, 100)
    harness = Harness(local, FakeProvider(name="cloud", chunks=["Cloud answer"]), budget)

    await harness.run()

    assert harness.channel.texts == [BUDGET_NOTICE + "Local answer"]
    assert harness.cloud.stream_calls == []


async def test_media_tier_replies_with_notice_without_calling_a_model() -> None:
    local = FakeProvider(name="local", reply=route_json("media"))
    harness = Harness(local)

    await harness.run("draw a cat")

    assert harness.channel.texts == [MEDIA_NOT_READY]
    assert local.stream_calls == []
    assert harness.cloud.stream_calls == []


async def test_stream_failure_keeps_partial_text_and_appends_error() -> None:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"))
    cloud = FakeProvider(
        name="cloud", chunks=["Partial"], stream_error=ProviderError("connection lost")
    )
    harness = Harness(local, cloud, DailyTokenBudget(1000))

    await harness.run()

    assert harness.channel.texts == ["Partial" + ERROR_NOTICE]
    assert harness.budget.used(7) == 0


async def test_reply_is_stored_as_a_bot_message_without_notices() -> None:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"), chunks=["Local answer"])
    budget = DailyTokenBudget(0)
    harness = Harness(local, budget=budget)

    await harness.run()

    [saved] = await harness.store.recent_messages(100, limit=10)
    assert saved.is_bot is True
    assert saved.content == "Local answer"
    assert saved.message_id == harness.channel.messages[0].id
    assert (saved.author_id, saved.author_name) == (42, "bot")
    assert (saved.guild_id, saved.channel_id) == (200, 100)


async def test_history_before_the_current_message_is_sent_to_the_model() -> None:
    local = FakeProvider(name="local", reply=route_json("local"), chunks=["ok"])
    harness = Harness(local)
    await harness.store.add_message(stored(4000, "what is pgvector?", author_name="bob"))
    await harness.store.add_message(
        stored(4001, "A Postgres extension.", is_bot=True, author_name="bot")
    )
    # The current message may already have been ingested; it must not appear twice.
    await harness.store.add_message(stored(5000, "@bot hello"))

    await harness.run()

    assert local.stream_calls[0]["messages"][1:] == [
        {"role": "user", "content": "bob: what is pgvector?"},
        {"role": "assistant", "content": "A Postgres extension."},
        {"role": "user", "content": "alice: hello"},
    ]


async def test_memory_failure_does_not_block_the_reply() -> None:
    local = FakeProvider(name="local", reply=route_json("local"), chunks=["still here"])
    harness = Harness(local)

    async def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database is down")

    harness.store.get_summary = broken  # type: ignore[method-assign]
    harness.store.add_message = broken  # type: ignore[method-assign]

    await harness.run()

    assert harness.channel.texts == ["still here"]


async def test_embedding_outage_only_disables_retrieval() -> None:
    local = FakeProvider(name="local", reply=route_json("local"), chunks=["ok"])
    harness = Harness(local, memory_provider=FakeProvider(embed_error=ProviderError("down")))
    await harness.store.add_message(stored(4000, "earlier message", author_name="bob"))

    await harness.run()

    assert harness.channel.texts == ["ok"]
    assert {"role": "user", "content": "bob: earlier message"} in local.stream_calls[0]["messages"]


def agent_harness(cloud: FakeProvider, tools: list[Tool]) -> Harness:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"))
    harness = Harness(local, cloud, DailyTokenBudget(10_000))
    harness.handler._agent = AgentLoop(cloud, ToolRegistry(tools))
    return harness


def make_tool(name: str, tier: Tier, calls: list) -> Tool:
    async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        calls.append((args, ctx))
        return ToolResult("result")

    return Tool(name=name, description="d", tier=tier, run=run)


async def test_agent_path_streams_tool_status_then_answer_and_records_usage() -> None:
    calls: list = []
    cloud = FakeProvider(
        name="cloud",
        streams=[
            tool_call_stream(("lookup", {"q": "x"}), usage=Usage(10, 5, 15)),
            text_stream("Found it.", usage=Usage(20, 5, 25)),
        ],
    )
    harness = agent_harness(cloud, [make_tool("lookup", Tier.READ_ONLY, calls)])

    await harness.run("look it up")

    [(args, ctx)] = calls
    assert args == {"q": "x"}
    assert (ctx.guild_id, ctx.channel_id, ctx.user_id, ctx.message_id) == (200, 100, 7, 5000)
    assert ctx.discord is None
    message = harness.channel.messages[0]
    assert message.sent_as == TOOL_STATUS.format(tool="lookup")
    assert message.edits == ["Found it."]  # the status is gone before the answer streams
    assert harness.budget.used(7) == 40
    assert AGENT_PROMPT in cloud.stream_calls[0]["messages"][0]["content"]
    [saved] = await harness.store.recent_messages(100, limit=10)
    assert saved.content == "Found it."


async def test_agent_side_effects_are_refused_without_discord_actions() -> None:
    calls: list = []
    cloud = FakeProvider(
        name="cloud", streams=[tool_call_stream(("pin", {})), text_stream("Could not.")]
    )
    harness = agent_harness(cloud, [make_tool("pin", Tier.SIDE_EFFECT, calls)])

    await harness.run("pin this")

    assert calls == []
    assert harness.channel.texts == ["Could not."]


async def test_without_an_agent_cloud_tier_is_a_plain_completion() -> None:
    local = FakeProvider(name="local", reply=route_json("cloud_agent"))
    cloud = FakeProvider(name="cloud", chunks=["plain"])
    harness = Harness(local, cloud)

    await harness.run()

    assert harness.channel.texts == ["plain"]
    assert cloud.stream_calls[0].get("tools") is None
    assert AGENT_PROMPT not in cloud.stream_calls[0]["messages"][0]["content"]


def test_build_messages_adds_memory_sections_as_reference_data() -> None:
    hit = ChunkHit(
        chunk_id=1,
        guild_id=200,
        channel_id=300,
        first_message_id=1111,
        last_message_id=1112,
        started_at=T0,
        content="bob: we deploy on Fridays",
        score=0.9,
    )
    context = MemoryContext(
        summary="They discussed the release plan.",
        hits=(hit,),
        facts=(UserFact(1, 7, 200, "Alice writes Rust.", T0),),
    )

    system = build_messages(make_event(FakeChannel(), "when do we deploy?"), context)[0]["content"]

    assert system.startswith(SYSTEM_PROMPT)
    assert "## What you know about alice\n- Alice writes Rust." in system
    assert "They discussed the release plan." in system
    assert "https://discord.com/channels/200/300/1111" in system
    assert "bob: we deploy on Fridays" in system
    assert system.endswith(REFERENCE_NOTE)
