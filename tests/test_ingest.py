from datetime import timedelta

from bot.memory import InMemoryStore
from bot.memory.ingest import Chunker, ChunkingConfig, split_into_groups, transcript
from fakes import T0, FakeProvider, stored

CONFIG = ChunkingConfig(gap=timedelta(minutes=5), max_messages=3, max_chars=1000)


def ids(groups: list[tuple[list, bool]]) -> list[tuple[list[int], bool]]:
    return [([m.message_id for m in group], closed) for group, closed in groups]


def test_no_messages_no_groups() -> None:
    assert split_into_groups([], CONFIG, now=T0) == []


def test_a_long_pause_starts_a_new_group() -> None:
    messages = [stored(1, minutes=0), stored(2, minutes=1), stored(3, minutes=10)]
    groups = split_into_groups(messages, CONFIG, now=T0 + timedelta(minutes=11))
    assert ids(groups) == [([1, 2], True), ([3], False)]


def test_last_group_closes_once_the_channel_goes_quiet() -> None:
    messages = [stored(1, minutes=0), stored(2, minutes=1)]
    assert ids(split_into_groups(messages, CONFIG, now=T0 + timedelta(minutes=3))) == [
        ([1, 2], False)
    ]
    assert ids(split_into_groups(messages, CONFIG, now=T0 + timedelta(minutes=7))) == [
        ([1, 2], True)
    ]


def test_group_is_capped_by_message_count() -> None:
    messages = [stored(i, minutes=i * 0.1) for i in range(1, 8)]
    groups = split_into_groups(messages, CONFIG, now=T0 + timedelta(minutes=1))
    assert ids(groups) == [([1, 2, 3], True), ([4, 5, 6], True), ([7], False)]


def test_group_is_capped_by_size() -> None:
    config = ChunkingConfig(gap=timedelta(minutes=5), max_messages=50, max_chars=40)
    messages = [stored(i, "x" * 12) for i in range(1, 5)]  # "alice: xxxxxxxxxxxx" = 19 chars
    groups = split_into_groups(messages, config, now=T0)
    assert ids(groups) == [([1, 2], True), ([3, 4], True)]


def test_transcript_names_each_author() -> None:
    messages = [stored(1, "hi"), stored(2, "hello", author_name="bob")]
    assert transcript(messages) == "alice: hi\nbob: hello"


async def test_chunker_stores_closed_groups_and_leaves_the_open_one() -> None:
    store = InMemoryStore()
    provider = FakeProvider(embed_fn=lambda text: [float(len(text)), 1.0])
    now = T0 + timedelta(minutes=11)
    chunker = Chunker(store, provider, config=CONFIG, clock=lambda: now)
    for message in (
        stored(1, "first", minutes=0),
        stored(2, "second", minutes=1, author_name="bob"),
        stored(3, "later", minutes=10),
    ):
        await store.add_message(message)

    assert await chunker.chunk_channel(100) == 1

    assert provider.embed_calls == [["alice: first\nbob: second"]]
    assert [m.message_id for m in await store.unchunked_messages(100, limit=10)] == [3]
    [hit] = await store.search_chunks([1.0, 0.0], guild_id=200, channel_id=100, limit=5)
    assert hit.content == "alice: first\nbob: second"
    assert (hit.first_message_id, hit.last_message_id) == (1, 2)
    assert hit.started_at == T0

    # Nothing new is closed, so no further embedding call is made.
    assert await chunker.chunk_channel(100) == 0
    assert len(provider.embed_calls) == 1


async def test_flush_closes_stale_groups_in_every_channel() -> None:
    store = InMemoryStore()
    provider = FakeProvider()
    now = T0
    chunker = Chunker(store, provider, config=CONFIG, clock=lambda: now)
    await store.add_message(stored(1, "in a", channel_id=100))
    await store.add_message(stored(2, "in b", channel_id=101))

    assert await chunker.flush() == 0

    now = T0 + timedelta(minutes=6)
    assert await chunker.flush() == 2
    assert await store.channels_with_unchunked() == []
