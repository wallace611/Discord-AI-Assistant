from datetime import timedelta

import pytest

from bot.memory import InMemoryStore, MemoryService
from bot.providers.base import ProviderError
from bot.text import extract_json_object, strip_reasoning
from fakes import T0, FakeProvider, keyword_embedding, stored


def make_service(provider: FakeProvider) -> tuple[MemoryService, InMemoryStore]:
    store = InMemoryStore()
    return MemoryService(store, chat=provider, embedder=provider, chat_model="m"), store


async def test_ingest_stores_chunks_and_summarises() -> None:
    provider = FakeProvider(reply="rolling summary")
    service, store = make_service(provider)
    now = T0 + timedelta(hours=1)
    service.chunker._clock = lambda: now

    # 25 messages: more than the 12 + 12 kept verbatim, so one summary pass runs.
    for index in range(1, 26):
        await service.ingest(stored(index, f"message {index}", minutes=index * 0.1))

    assert await store.count_messages(100) == 25
    summary = await store.get_summary(100)
    assert summary is not None
    assert summary.summary == "rolling summary"
    assert summary.covered_message_id == 13
    # The first 20 messages filled a chunk; the other 5 are closed too, since the
    # clock says the channel has been quiet for an hour.
    hits = await store.search_chunks([1.0, 0.0], guild_id=200, channel_id=100, limit=10)
    assert len(hits) >= 1
    assert await service.flush() == 0


async def test_flush_closes_quiet_channels() -> None:
    provider = FakeProvider()
    service, store = make_service(provider)
    now = T0
    service.chunker._clock = lambda: now
    await service.ingest(stored(1, "only message"))
    assert await store.channels_with_unchunked() == [100]

    now = T0 + timedelta(minutes=10)
    assert await service.flush() == 1
    assert await store.channels_with_unchunked() == []


async def test_context_combines_summary_recent_hits_and_facts() -> None:
    provider = FakeProvider(embed_fn=keyword_embedding("cat", "dog"))
    service, store = make_service(provider)
    now = T0 + timedelta(hours=1)
    service.chunker._clock = lambda: now

    # An old conversation in another channel of the same guild, already chunked.
    await service.ingest(stored(1, "my cat is called Mochi", channel_id=101, author_name="bob"))
    # The same topic in a different guild must stay invisible.
    await service.ingest(stored(2, "secret cat facts", channel_id=900, guild_id=999))
    await store.set_summary(100, 200, "Earlier they talked about pets.", 10)
    await store.add_message(stored(11, "any pets?", minutes=58))
    await store.add_message(stored(12, "@bot what is the cat called?", minutes=59))
    await store.add_fact(7, 200, "Alice likes cats.", None)

    context = await service.context(
        guild_id=200,
        channel_id=100,
        user_id=7,
        query="what is the cat called?",
        before_message_id=12,
    )

    assert context.summary == "Earlier they talked about pets."
    assert [m.message_id for m in context.recent] == [11]
    assert [hit.content for hit in context.hits] == ["bob: my cat is called Mochi"]
    assert [fact.fact for fact in context.facts] == ["Alice likes cats."]


async def test_context_survives_an_embedding_outage() -> None:
    provider = FakeProvider(embed_error=ProviderError("ollama down"))
    service, store = make_service(provider)
    await store.add_message(stored(1, "hello"))

    context = await service.context(guild_id=200, channel_id=100, user_id=7, query="hi")

    assert [m.message_id for m in context.recent] == [1]
    assert context.hits == ()


def test_strip_reasoning() -> None:
    assert strip_reasoning("<think>\nplanning\n</think>\n answer ") == "answer"
    assert strip_reasoning("no reasoning") == "no reasoning"


def test_extract_json_object() -> None:
    assert extract_json_object('<think>{"x": 1}</think>```json\n{"a": [1, 2]}\n```') == {
        "a": [1, 2]
    }
    for bad in ("", "plain text", '{"a": ', "}{"):
        with pytest.raises(ValueError):
            extract_json_object(bad)
