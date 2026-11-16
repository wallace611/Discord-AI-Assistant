from collections.abc import Sequence

from bot.memory import InMemoryStore, NewChunk
from bot.memory.rag import Retriever, jump_url
from fakes import T0, FakeProvider, keyword_embedding

EMBED = keyword_embedding("cat", "dog", "python")


async def add(
    store: InMemoryStore,
    content: str,
    *,
    guild_id: int | None = 200,
    channel_id: int = 100,
    message_ids: tuple[int, ...] = (1, 2),
) -> None:
    await store.add_chunk(
        NewChunk(
            guild_id=guild_id,
            channel_id=channel_id,
            message_ids=message_ids,
            started_at=T0,
            ended_at=T0,
            content=content,
            embedding=EMBED(content),
        )
    )


def make(store: InMemoryStore, **kwargs: object) -> Retriever:
    return Retriever(store, FakeProvider(embed_fn=EMBED), **kwargs)  # type: ignore[arg-type]


async def test_returns_matching_chunks_and_drops_weak_ones() -> None:
    store = InMemoryStore()
    await add(store, "my cat sleeps all day")
    await add(store, "the dog barked")
    await add(store, "python and cat pictures")

    hits = await make(store).search("tell me about the cat", guild_id=200, channel_id=100)

    assert [hit.content for hit in hits] == ["my cat sleeps all day", "python and cat pictures"]


async def test_never_returns_chunks_from_another_guild_or_dm() -> None:
    store = InMemoryStore()
    await add(store, "cat in guild 200")
    await add(store, "cat in guild 201", guild_id=201)
    await add(store, "cat in a dm", guild_id=None, channel_id=555)

    in_guild = await make(store).search("cat", guild_id=200, channel_id=100)
    in_dm = await make(store).search("cat", guild_id=None, channel_id=555)
    in_other_dm = await make(store).search("cat", guild_id=None, channel_id=556)

    assert [hit.content for hit in in_guild] == ["cat in guild 200"]
    assert [hit.content for hit in in_dm] == ["cat in a dm"]
    assert in_other_dm == []


async def test_top_k_limits_results() -> None:
    store = InMemoryStore()
    for index in range(6):
        await add(store, f"cat number {index}")
    hits = await make(store, top_k=2).search("cat", guild_id=200, channel_id=100)
    assert len(hits) == 2


async def test_blank_query_skips_embedding() -> None:
    provider = FakeProvider(embed_fn=EMBED)
    hits = await Retriever(InMemoryStore(), provider).search("  ", guild_id=200, channel_id=100)
    assert hits == []
    assert provider.embed_calls == []


async def test_chunks_already_in_the_recent_window_are_excluded() -> None:
    store = InMemoryStore()
    await add(store, "old cat talk", message_ids=(10, 11))
    await add(store, "recent cat talk", message_ids=(40, 41))
    await add(store, "recent cat talk elsewhere", channel_id=101, message_ids=(40, 41))

    hits = await make(store).search("cat", guild_id=200, channel_id=100, newer_than_excluded=30)

    assert sorted(hit.content for hit in hits) == ["old cat talk", "recent cat talk elsewhere"]


async def test_reranker_reorders_candidates() -> None:
    class LongestFirst:
        async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
            return [float(len(document)) for document in documents]

    store = InMemoryStore()
    await add(store, "cat")
    await add(store, "cat with a much longer description")

    hits = await make(store, reranker=LongestFirst()).search("cat", guild_id=200, channel_id=100)

    assert [hit.content for hit in hits] == ["cat with a much longer description", "cat"]


async def test_jump_url_for_guild_and_dm() -> None:
    store = InMemoryStore()
    await add(store, "cat", channel_id=300, message_ids=(1111, 1112))
    await add(store, "cat", guild_id=None, channel_id=400, message_ids=(2222,))

    [guild_hit] = await make(store).search("cat", guild_id=200, channel_id=100)
    [dm_hit] = await make(store).search("cat", guild_id=None, channel_id=400)

    assert jump_url(guild_hit) == "https://discord.com/channels/200/300/1111"
    assert jump_url(dm_hit) == "https://discord.com/channels/@me/400/2222"
