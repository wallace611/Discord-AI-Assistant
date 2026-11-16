from bot.memory import InMemoryStore
from bot.memory.short_term import ShortTermMemory
from bot.providers.base import ProviderError
from fakes import FakeProvider, stored


async def fill(store: InMemoryStore, count: int, start: int = 1) -> None:
    for message_id in range(start, start + count):
        await store.add_message(stored(message_id, f"message {message_id}"))


def make(provider: FakeProvider, store: InMemoryStore) -> ShortTermMemory:
    return ShortTermMemory(store, provider, model="m", keep_recent=3, summarize_batch=2)


async def test_no_summary_until_enough_messages_pile_up() -> None:
    store, provider = InMemoryStore(), FakeProvider(reply="summary")
    memory = make(provider, store)
    await fill(store, 5)  # keep_recent + summarize_batch

    assert await memory.maybe_summarize(100, 200) is False
    assert provider.complete_calls == []

    summary, recent = await memory.load(100)
    assert summary is None
    assert [m.message_id for m in recent] == [1, 2, 3, 4, 5]


async def test_older_messages_are_folded_and_recent_ones_kept_verbatim() -> None:
    store, provider = InMemoryStore(), FakeProvider(reply="<think>hmm</think>They said hi.")
    memory = make(provider, store)
    await fill(store, 6)

    assert await memory.maybe_summarize(100, 200) is True

    prompt = provider.complete_calls[0]["messages"][1]["content"]
    assert "(none)" in prompt
    assert "alice: message 3" in prompt
    assert "alice: message 4" not in prompt
    assert provider.complete_calls[0]["model"] == "m"

    summary, recent = await memory.load(100)
    assert summary == "They said hi."
    assert [m.message_id for m in recent] == [4, 5, 6]


async def test_next_update_builds_on_the_previous_summary() -> None:
    store, provider = InMemoryStore(), FakeProvider(replies=["first summary", "second summary"])
    memory = make(provider, store)
    await fill(store, 6)
    await memory.maybe_summarize(100, 200)

    await fill(store, 2, start=7)
    assert await memory.maybe_summarize(100, 200) is False  # only 5 messages after id 3

    await fill(store, 1, start=9)
    assert await memory.maybe_summarize(100, 200) is True

    prompt = provider.complete_calls[1]["messages"][1]["content"]
    assert "first summary" in prompt
    assert "alice: message 4" in prompt and "alice: message 6" in prompt
    assert "alice: message 3" not in prompt and "alice: message 7" not in prompt
    summary, recent = await memory.load(100)
    assert summary == "second summary"
    assert [m.message_id for m in recent] == [7, 8, 9]


async def test_load_can_exclude_the_message_being_answered() -> None:
    store = InMemoryStore()
    memory = make(FakeProvider(), store)
    await fill(store, 4)
    _, recent = await memory.load(100, before_id=4)
    assert [m.message_id for m in recent] == [1, 2, 3]


async def test_failed_or_empty_summary_leaves_state_untouched() -> None:
    store = InMemoryStore()
    await fill(store, 6)

    failing = make(FakeProvider(error=ProviderError("down")), store)
    assert await failing.maybe_summarize(100, 200) is False

    empty = make(FakeProvider(reply="<think>nothing</think>"), store)
    assert await empty.maybe_summarize(100, 200) is False

    assert await store.get_summary(100) is None


async def test_channels_are_independent() -> None:
    store, provider = InMemoryStore(), FakeProvider(reply="summary of 100")
    memory = make(provider, store)
    await fill(store, 6)
    await store.add_message(stored(50, "elsewhere", channel_id=101))
    await memory.maybe_summarize(100, 200)

    summary, recent = await memory.load(101)
    assert summary is None
    assert [m.content for m in recent] == ["elsewhere"]
