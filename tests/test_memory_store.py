from bot.memory import InMemoryStore, NewChunk
from bot.memory.postgres import _affected, _vector_literal
from bot.memory.store import in_scope
from fakes import T0, stored


def chunk(
    content: str,
    embedding: list[float],
    *,
    guild_id: int | None = 200,
    channel_id: int = 100,
    message_ids: tuple[int, ...] = (1, 2),
) -> NewChunk:
    return NewChunk(
        guild_id=guild_id,
        channel_id=channel_id,
        message_ids=message_ids,
        started_at=T0,
        ended_at=T0,
        content=content,
        embedding=embedding,
    )


def test_scope_rule() -> None:
    # Guild callers see the whole guild, and nothing else.
    assert in_scope(200, 100, 200, 999) is True
    assert in_scope(200, 100, 201, 100) is False
    assert in_scope(200, 100, None, 100) is False
    # DM callers see only their own DM channel.
    assert in_scope(None, 100, None, 100) is True
    assert in_scope(None, 100, None, 101) is False
    assert in_scope(None, 100, 200, 100) is False


async def test_recent_messages_window_and_order() -> None:
    store = InMemoryStore()
    for message_id in (5, 3, 1, 4, 2):
        await store.add_message(stored(message_id, f"m{message_id}"))
    await store.add_message(stored(6, "other channel", channel_id=999))

    recent = await store.recent_messages(100, limit=3)
    assert [m.message_id for m in recent] == [3, 4, 5]

    bounded = await store.recent_messages(100, after_id=1, before_id=5, limit=10)
    assert [m.message_id for m in bounded] == [2, 3, 4]

    assert await store.count_messages(100, after_id=2) == 3


async def test_duplicate_message_ids_are_ignored() -> None:
    store = InMemoryStore()
    await store.add_message(stored(1, "original"))
    await store.add_message(stored(1, "duplicate"))
    [only] = await store.recent_messages(100, limit=10)
    assert only.content == "original"


async def test_add_chunk_marks_messages_as_chunked() -> None:
    store = InMemoryStore()
    for message_id in (1, 2, 3):
        await store.add_message(stored(message_id))
    assert await store.channels_with_unchunked() == [100]

    await store.add_chunk(chunk("a", [1.0, 0.0], message_ids=(1, 2)))

    assert [m.message_id for m in await store.unchunked_messages(100, limit=10)] == [3]
    await store.add_chunk(chunk("b", [1.0, 0.0], message_ids=(3,)))
    assert await store.channels_with_unchunked() == []


async def test_search_ranks_by_similarity_and_never_crosses_guilds() -> None:
    store = InMemoryStore()
    await store.add_chunk(chunk("close", [1.0, 0.1], channel_id=101))
    await store.add_chunk(chunk("far", [0.0, 1.0], channel_id=102))
    await store.add_chunk(chunk("other guild", [1.0, 0.0], guild_id=201))
    await store.add_chunk(chunk("a dm", [1.0, 0.0], guild_id=None, channel_id=100))

    hits = await store.search_chunks([1.0, 0.0], guild_id=200, channel_id=100, limit=10)

    assert [hit.content for hit in hits] == ["close", "far"]
    assert hits[0].score > 0.9 > hits[1].score
    assert (hits[0].first_message_id, hits[0].last_message_id) == (1, 2)


async def test_dm_search_only_sees_its_own_channel() -> None:
    store = InMemoryStore()
    await store.add_chunk(chunk("my dm", [1.0, 0.0], guild_id=None, channel_id=100))
    await store.add_chunk(chunk("someone else's dm", [1.0, 0.0], guild_id=None, channel_id=101))
    await store.add_chunk(chunk("a guild", [1.0, 0.0], guild_id=200, channel_id=100))

    hits = await store.search_chunks([1.0, 0.0], guild_id=None, channel_id=100, limit=10)

    assert [hit.content for hit in hits] == ["my dm"]


async def test_facts_are_scoped_deduplicated_and_owner_protected() -> None:
    store = InMemoryStore()
    assert await store.add_fact(7, 200, "Likes Rust.", None) is True
    assert await store.add_fact(7, 200, "likes rust.", None) is False
    assert await store.add_fact(7, 201, "Likes Rust.", None) is True
    assert await store.add_fact(7, None, "Lives in a DM.", None) is True
    assert await store.add_fact(8, 200, "Likes Go.", None) is True

    mine = await store.list_facts(7, 200)
    assert [fact.fact for fact in mine] == ["Likes Rust."]

    # User 8 cannot delete user 7's fact.
    assert await store.delete_fact(8, mine[0].fact_id) is False
    assert await store.delete_fact(7, mine[0].fact_id) is True
    assert await store.list_facts(7, 200) == []

    assert await store.delete_all_facts(7, None) == 1
    assert len(await store.list_facts(7, 201)) == 1
    assert len(await store.list_facts(8, 200)) == 1


def test_postgres_helpers() -> None:
    assert _vector_literal([1, 0.5, -1e-05]) == "[1.0,0.5,-1e-05]"
    assert _affected("DELETE 3") == 3
    assert _affected("INSERT 0 1") == 1
