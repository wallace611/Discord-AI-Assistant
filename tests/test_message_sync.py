"""Deleting and editing messages on Discord must reach the bot's stored copies."""

import asyncio
import logging
import re
from datetime import timedelta
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bot.gateway.client import BotClient
from bot.memory import InMemoryStore, MemoryService, NewChunk
from bot.memory import service as service_module
from bot.memory.postgres import PostgresStore
from bot.providers.base import ChatResponse
from fakes import T0, FakeProvider, keyword_embedding, stored
from test_postgres_store import RecordingPool

CHANNEL = 100
GUILD = 200


def chunk(
    message_ids: tuple[int, ...],
    content: str = "alice: hi",
    *,
    channel_id: int = CHANNEL,
    guild_id: int | None = GUILD,
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> NewChunk:
    return NewChunk(
        guild_id=guild_id,
        channel_id=channel_id,
        message_ids=message_ids,
        started_at=T0,
        ended_at=T0,
        content=content,
        embedding=list(embedding),
    )


async def chunk_contents(store: InMemoryStore, channel_id: int = CHANNEL) -> list[str]:
    hits = await store.search_chunks([1.0, 0.0], guild_id=GUILD, channel_id=channel_id, limit=50)
    return [hit.content for hit in hits]


async def unchunked_ids(store: InMemoryStore, channel_id: int = CHANNEL) -> list[int]:
    return [m.message_id for m in await store.unchunked_messages(channel_id, limit=100)]


async def stored_ids(store: InMemoryStore, channel_id: int = CHANNEL) -> list[int]:
    return [m.message_id for m in await store.recent_messages(channel_id, limit=100)]


async def populated_store() -> InMemoryStore:
    """Messages 1-3 in one chunk, 4-5 in another, 6 unchunked; a summary and facts."""
    store = InMemoryStore()
    for message_id in range(1, 7):
        await store.add_message(stored(message_id, f"message {message_id}"))
    await store.add_chunk(chunk((1, 2, 3), "chunk A"))
    await store.add_chunk(chunk((4, 5), "chunk B"))
    await store.set_summary(CHANNEL, GUILD, "summary", 3)
    await store.add_fact(7, GUILD, "Fact from 2.", 2)
    await store.add_fact(7, GUILD, "Fact from 5.", 5)
    await store.add_fact(7, GUILD, "Fact from nowhere.", None)
    return store


# --- deleting: the in-process store -----------------------------------------


async def test_deleting_a_message_removes_only_that_message() -> None:
    store = await populated_store()

    assert await store.delete_messages(CHANNEL, [6]) == 1

    assert await stored_ids(store) == [1, 2, 3, 4, 5]
    assert sorted(await chunk_contents(store)) == ["chunk A", "chunk B"]
    assert [fact.fact for fact in await store.list_facts(7, GUILD)] == [
        "Fact from 2.",
        "Fact from 5.",
        "Fact from nowhere.",
    ]


async def test_deleting_a_chunked_message_drops_its_chunk_and_frees_the_others() -> None:
    store = await populated_store()

    assert await store.delete_messages(CHANNEL, [2]) == 1

    assert await chunk_contents(store) == ["chunk B"]
    # 1 and 3 can be chunked again; 4 and 5 stay in their own chunk.
    assert await unchunked_ids(store) == [1, 3, 6]
    assert await stored_ids(store) == [1, 3, 4, 5, 6]


async def test_summary_is_dropped_only_when_it_covers_a_deleted_message() -> None:
    store = await populated_store()  # the summary covers messages up to 3

    await store.delete_messages(CHANNEL, [5])
    assert (await store.get_summary(CHANNEL)) is not None

    await store.delete_messages(CHANNEL, [3])
    assert (await store.get_summary(CHANNEL)) is None


async def test_facts_extracted_from_a_deleted_message_are_deleted() -> None:
    store = await populated_store()

    await store.delete_messages(CHANNEL, [2])

    assert [fact.fact for fact in await store.list_facts(7, GUILD)] == [
        "Fact from 5.",
        "Fact from nowhere.",
    ]


async def test_bulk_delete_handles_duplicates_and_unknown_ids() -> None:
    store = await populated_store()

    assert await store.delete_messages(CHANNEL, [1, 1, 5, 999]) == 2

    assert await stored_ids(store) == [2, 3, 4, 6]
    assert await chunk_contents(store) == []
    assert await unchunked_ids(store) == [2, 3, 4, 6]
    assert await store.delete_messages(CHANNEL, []) == 0


async def test_delete_is_scoped_to_the_given_channel_and_leaves_other_data_alone() -> None:
    store = await populated_store()
    # Same shape of data in another channel of the same guild, and in another guild.
    await store.add_message(stored(50, "elsewhere", channel_id=101))
    await store.add_chunk(chunk((50,), "other channel", channel_id=101))
    await store.set_summary(101, GUILD, "other summary", 50)
    await store.add_message(stored(60, "foreign", channel_id=900, guild_id=999))
    await store.add_chunk(chunk((60,), "other guild", channel_id=900, guild_id=999))
    await store.set_summary(900, 999, "foreign summary", 60)
    await store.add_fact(8, 999, "Foreign fact.", 60)

    # Message 2 is stored in channel 100: naming any other channel must not touch it.
    assert await store.delete_messages(101, [2]) == 0
    assert await store.delete_messages(900, [2]) == 0
    assert await stored_ids(store) == [1, 2, 3, 4, 5, 6]
    assert sorted(await chunk_contents(store)) == ["chunk A", "chunk B", "other channel"]
    assert (await store.get_summary(CHANNEL)) is not None
    assert len(await store.list_facts(7, GUILD)) == 3

    # A real delete in channel 100 changes nothing anywhere else.
    assert await store.delete_messages(CHANNEL, [2]) == 1
    assert await stored_ids(store, 101) == [50]
    assert await stored_ids(store, 900) == [60]
    # Chunk A went with message 2; every other chunk of the guild is still there.
    assert sorted(await chunk_contents(store)) == ["chunk B", "other channel"]
    assert (await store.get_summary(101)) is not None
    assert (await store.get_summary(900)) is not None
    assert [f.fact for f in await store.list_facts(8, 999)] == ["Foreign fact."]
    foreign = await store.search_chunks([1.0, 0.0], guild_id=999, channel_id=900, limit=5)
    assert [hit.content for hit in foreign] == ["other guild"]


async def test_summary_of_another_channel_survives_even_if_it_covers_the_id() -> None:
    store = InMemoryStore()
    await store.add_message(stored(5, "mine", channel_id=100))
    await store.set_summary(101, GUILD, "other", 999)  # covers id 5, but is not channel 100's

    await store.delete_messages(100, [5])

    assert (await store.get_summary(101)) is not None


async def test_delete_facts_from_removes_only_facts_from_those_messages() -> None:
    store = await populated_store()

    assert await store.delete_facts_from([2, 999]) == 1
    assert await store.delete_facts_from([]) == 0

    assert [f.fact for f in await store.list_facts(7, GUILD)] == [
        "Fact from 5.",
        "Fact from nowhere.",
    ]
    assert await stored_ids(store) == [1, 2, 3, 4, 5, 6]  # messages are untouched


async def test_a_user_can_still_delete_a_fact_whose_source_was_tracked() -> None:
    store = await populated_store()
    [first, *_] = await store.list_facts(7, GUILD)

    assert await store.delete_fact(7, first.fact_id) is True
    assert await store.delete_all_facts(7, GUILD) == 2
    # Nothing is left for the cleanup to find, and it does not fail on stale entries.
    assert await store.delete_facts_from([2, 5]) == 0


# --- editing: the in-process store ------------------------------------------


async def test_edit_replaces_text_and_drops_what_was_derived_from_it() -> None:
    store = await populated_store()

    assert await store.edit_message(CHANNEL, 2, "edited text") is True

    messages = {m.message_id: m.content for m in await store.recent_messages(CHANNEL, limit=100)}
    assert messages[2] == "edited text"
    assert messages[1] == "message 1"
    assert await chunk_contents(store) == ["chunk B"]
    assert await unchunked_ids(store) == [1, 2, 3, 6]
    assert (await store.get_summary(CHANNEL)) is None
    assert [f.fact for f in await store.list_facts(7, GUILD)] == [
        "Fact from 5.",
        "Fact from nowhere.",
    ]


async def test_editing_to_identical_text_changes_nothing() -> None:
    store = await populated_store()

    assert await store.edit_message(CHANNEL, 2, "message 2") is False

    assert sorted(await chunk_contents(store)) == ["chunk A", "chunk B"]
    assert (await store.get_summary(CHANNEL)) is not None
    assert len(await store.list_facts(7, GUILD)) == 3


async def test_edit_of_a_later_message_keeps_an_earlier_summary() -> None:
    store = await populated_store()  # the summary covers up to 3

    await store.edit_message(CHANNEL, 6, "late edit")

    assert (await store.get_summary(CHANNEL)) is not None
    assert sorted(await chunk_contents(store)) == ["chunk A", "chunk B"]


async def test_edit_ignores_unknown_messages_and_other_channels() -> None:
    store = await populated_store()

    assert await store.edit_message(CHANNEL, 999, "x") is False
    assert await store.edit_message(101, 2, "x") is False
    assert await store.edit_message(900, 2, "x") is False

    [second] = [m for m in await store.recent_messages(CHANNEL, limit=100) if m.message_id == 2]
    assert second.content == "message 2"
    assert sorted(await chunk_contents(store)) == ["chunk A", "chunk B"]
    assert (await store.get_summary(CHANNEL)) is not None


# --- the service ------------------------------------------------------------


def make_service(
    provider: FakeProvider | None = None,
) -> tuple[MemoryService, InMemoryStore, FakeProvider]:
    provider = provider or FakeProvider(embed_fn=keyword_embedding("rust", "go"))
    store = InMemoryStore()
    service = MemoryService(store, chat=provider, embedder=provider, chat_model="m")
    service.chunker._clock = lambda: T0 + timedelta(hours=1)  # every channel is quiet
    return service, store, provider


async def test_a_deleted_message_is_not_chunked_into_retrieval() -> None:
    service, store, _ = make_service()
    now = T0 + timedelta(minutes=2)  # the channel is still active, so chunks stay open
    service.chunker._clock = lambda: now
    await service.ingest(stored(1, "I love rust", minutes=0))
    await service.ingest(stored(2, "my password is hunter2, rust", minutes=1))
    await service.ingest(stored(3, "rust is great", minutes=2))
    now = T0 + timedelta(hours=1)
    assert await service.flush() == 1  # one chunk holding all three messages

    await service.forget_messages(CHANNEL, [2])
    assert await unchunked_ids(store) == [1, 3]  # the survivors are free to be chunked again
    assert await service.flush() == 1

    hits = await service.retriever.search("rust", guild_id=GUILD, channel_id=999)
    contents = " | ".join(hit.content for hit in hits)
    assert "hunter2" not in contents
    assert "I love rust" in contents and "rust is great" in contents


async def test_an_edited_message_is_chunked_again_with_its_new_text() -> None:
    service, store, _ = make_service()
    now = T0 + timedelta(minutes=1)
    service.chunker._clock = lambda: now
    await service.ingest(stored(1, "I love rust", minutes=0))
    await service.ingest(stored(2, "my address is 1 Main Street, rust", minutes=1))
    now = T0 + timedelta(hours=1)
    assert await service.flush() == 1

    assert await service.edit_message(CHANNEL, 2, "never mind, rust") is True
    assert await unchunked_ids(store) == [1, 2]
    assert await service.flush() == 1

    hits = await service.retriever.search("rust", guild_id=GUILD, channel_id=999)
    contents = " | ".join(hit.content for hit in hits)
    assert "Main Street" not in contents
    assert "never mind, rust" in contents


async def test_a_delete_that_overtakes_the_queued_write_wins() -> None:
    service, store, _ = make_service()

    await service.forget_messages(CHANNEL, [5])
    await service.record(stored(5, "too late"))
    await service.record(stored(6, "unrelated"))

    assert await stored_ids(store) == [6]


async def test_an_edit_that_overtakes_the_queued_write_wins() -> None:
    service, store, _ = make_service()

    assert await service.edit_message(CHANNEL, 5, "new text") is False  # not stored yet
    await service.record(stored(5, "old text"))

    [only] = await store.recent_messages(CHANNEL, limit=10)
    assert only.content == "new text"


async def test_forgetting_waits_for_chunking_or_summarising_in_progress() -> None:
    service, store, _ = make_service()
    await service.record(stored(1, "hello"))

    await service._maintenance.acquire()  # a maintenance pass is running
    task = asyncio.create_task(service.forget_messages(CHANNEL, [1]))
    await asyncio.sleep(0.01)
    assert not task.done()
    assert await stored_ids(store) == [1]

    service._maintenance.release()
    assert await task == 1
    assert await stored_ids(store) == []


async def test_records_arriving_during_a_forget_are_not_lost_or_resurrected() -> None:
    service, store, _ = make_service()
    await service.record(stored(1, "doomed"))
    await service._maintenance.acquire()
    forgetting = asyncio.create_task(service.forget_messages(CHANNEL, [1]))
    await asyncio.sleep(0.01)

    # The tombstone is set at once, so a duplicate write of the same message is dropped
    # and an unrelated message is stored normally.
    await service.record(stored(1, "doomed"))
    await service.record(stored(2, "fine"))
    service._maintenance.release()
    await forgetting

    assert await stored_ids(store) == [2]


FACTS_REPLY = '{"facts": ["Likes Rust."]}'
LONG_MESSAGE = "I really like writing Rust every day"


async def test_facts_are_extracted_normally_for_an_untouched_message() -> None:
    service, store, provider = make_service(FakeProvider(reply=FACTS_REPLY))
    message = stored(1, LONG_MESSAGE)
    await service.record(message)

    assert await service.extract_facts(message) == ["Likes Rust."]
    assert [f.fact for f in await store.list_facts(7, GUILD)] == ["Likes Rust."]


async def test_facts_are_not_extracted_from_a_message_already_deleted() -> None:
    service, store, provider = make_service(FakeProvider(reply=FACTS_REPLY))
    message = stored(1, LONG_MESSAGE)
    await service.record(message)
    await service.forget_messages(CHANNEL, [1])

    assert await service.extract_facts(message) == []
    assert provider.complete_calls == []


async def test_facts_are_not_extracted_from_text_that_was_already_edited() -> None:
    service, store, provider = make_service(FakeProvider(reply=FACTS_REPLY))
    message = stored(1, LONG_MESSAGE)
    await service.record(message)
    await service.edit_message(CHANNEL, 1, "never mind, forget that")

    assert await service.extract_facts(message) == []
    assert provider.complete_calls == []


@pytest.mark.parametrize("change", ["delete", "edit"])
async def test_facts_are_removed_when_the_message_changes_during_extraction(change: str) -> None:
    class ChangingProvider(FakeProvider):
        service: MemoryService

        async def complete(self, messages: Any, **kwargs: Any) -> ChatResponse:
            response = await super().complete(messages, **kwargs)
            # The user deletes or edits the message while the model is still reading it.
            if change == "delete":
                await self.service.forget_messages(CHANNEL, [1])
            else:
                await self.service.edit_message(CHANNEL, 1, "something else entirely")
            return response

    provider = ChangingProvider(reply=FACTS_REPLY)
    service, store, _ = make_service(provider)
    provider.service = service
    message = stored(1, LONG_MESSAGE)
    await service.record(message)

    assert await service.extract_facts(message) == []
    assert await store.list_facts(7, GUILD) == []


async def test_remembered_deletions_and_edits_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "TRACKED_LIMIT", 3)
    service, _, _ = make_service()

    await service.forget_messages(CHANNEL, [1, 2, 3, 4, 5])
    await service.edit_message(CHANNEL, 10, "a")
    await service.edit_message(CHANNEL, 11, "b")
    await service.edit_message(CHANNEL, 12, "c")
    await service.edit_message(CHANNEL, 13, "d")

    assert list(service._deleted) == [3, 4, 5]  # the oldest were forgotten
    assert list(service._edits) == [11, 12, 13]


# --- the gateway client -----------------------------------------------------


def make_client() -> tuple[BotClient, MemoryService, InMemoryStore]:
    service, store, _ = make_service(FakeProvider())
    return BotClient(AsyncMock(), service), service, store


def discord_message(
    message_id: int, content: str, *, is_bot: bool = False, channel_id: int = CHANNEL
) -> Any:
    return NS(
        id=message_id,
        clean_content=content,
        created_at=T0,
        author=NS(id=7, bot=is_bot, display_name="alice"),
        guild=NS(id=GUILD),
        channel=NS(id=channel_id),
    )


async def test_raw_delete_forgets_the_message() -> None:
    client, service, store = make_client()
    await service.record(stored(1, "keep"))
    await service.record(stored(2, "remove"))

    await client.on_raw_message_delete(NS(message_id=2, channel_id=CHANNEL, guild_id=GUILD))

    assert await stored_ids(store) == [1]


async def test_raw_delete_of_the_bots_own_reply_is_handled() -> None:
    client, service, store = make_client()
    await service.record(stored(1, "question"))
    await service.record(stored(2, "my answer", is_bot=True, author_id=42))
    await store.add_chunk(chunk((1, 2), "question and answer"))

    await client.on_raw_message_delete(NS(message_id=2, channel_id=CHANNEL, guild_id=GUILD))

    assert await stored_ids(store) == [1]
    assert await chunk_contents(store) == []
    assert await unchunked_ids(store) == [1]


async def test_raw_bulk_delete_forgets_every_message_in_that_channel_only() -> None:
    client, service, store = make_client()
    for message_id in (1, 2, 3):
        await service.record(stored(message_id))
    await service.record(stored(4, channel_id=101))

    await client.on_raw_bulk_message_delete(
        NS(message_ids={1, 3, 4}, channel_id=CHANNEL, guild_id=GUILD)
    )

    assert await stored_ids(store) == [2]
    assert await stored_ids(store, 101) == [4]  # id 4 was named, but it is not in this channel


async def test_raw_edit_updates_the_stored_text() -> None:
    client, service, store = make_client()
    await service.record(stored(1, "before"))
    await store.add_chunk(chunk((1,), "alice: before"))

    await client.on_raw_message_edit(
        NS(message_id=1, channel_id=CHANNEL, message=discord_message(1, "after"))
    )

    [only] = await store.recent_messages(CHANNEL, limit=10)
    assert only.content == "after"
    assert await chunk_contents(store) == []


async def test_raw_edit_of_a_bot_message_is_ignored() -> None:
    # The bot edits its own reply while streaming; the stored copy is the final text.
    client, service, store = make_client()
    await service.record(stored(1, "the full answer", is_bot=True, author_id=42))

    await client.on_raw_message_edit(
        NS(message_id=1, channel_id=CHANNEL, message=discord_message(1, "partial", is_bot=True))
    )

    [only] = await store.recent_messages(CHANNEL, limit=10)
    assert only.content == "the full answer"


async def test_raw_edit_to_empty_text_is_ignored() -> None:
    client, service, store = make_client()
    await service.record(stored(1, "text"))

    await client.on_raw_message_edit(
        NS(message_id=1, channel_id=CHANNEL, message=discord_message(1, "  "))
    )

    [only] = await store.recent_messages(CHANNEL, limit=10)
    assert only.content == "text"


async def test_memory_failures_are_logged_and_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, service, _ = make_client()
    service.forget_messages = AsyncMock(side_effect=RuntimeError("database down"))  # type: ignore[method-assign]
    service.edit_message = AsyncMock(side_effect=RuntimeError("database down"))  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR, logger="bot.gateway.client"):
        await client.on_raw_message_delete(NS(message_id=1, channel_id=CHANNEL, guild_id=GUILD))
        await client.on_raw_bulk_message_delete(
            NS(message_ids={1, 2}, channel_id=CHANNEL, guild_id=GUILD)
        )
        await client.on_raw_message_edit(
            NS(message_id=1, channel_id=CHANNEL, message=discord_message(1, "x"))
        )

    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("could not forget deleted messages") == 2
    assert messages.count("could not apply a message edit") == 1


async def test_remembering_a_message_that_was_deleted_in_the_meantime_stores_nothing() -> None:
    client, service, store = make_client()
    message = stored(1, LONG_MESSAGE)

    await client.on_raw_message_delete(NS(message_id=1, channel_id=CHANNEL, guild_id=GUILD))
    await client._remember(  # the queued job runs after the delete event
        NS(message=message, addressed=True)  # type: ignore[arg-type]
    )

    assert await stored_ids(store) == []
    assert await store.list_facts(7, GUILD) == []


# --- Postgres: statement shapes ----------------------------------------------


class ScriptedPool(RecordingPool):
    """Recording pool that answers the SELECT/UPDATE with prepared rows."""

    def __init__(self, rows: list[dict[str, Any]], row: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.rows = rows
        self.row = row
        self.transactions = 0
        self.open_transactions = 0
        self.in_transaction: list[bool] = []

    def _record(self, sql: str, args: tuple[Any, ...]) -> None:
        super()._record(sql, args)
        self.in_transaction.append(self.open_transactions > 0)

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._record(sql, args)
        return self.rows

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._record(sql, args)
        return self.row

    def transaction(self) -> Any:
        pool = self

        class Transaction:
            async def __aenter__(self) -> None:
                pool.transactions += 1
                pool.open_transactions += 1

            async def __aexit__(self, *exc: object) -> None:
                pool.open_transactions -= 1

        return Transaction()


def assert_placeholders_match(pool: RecordingPool) -> None:
    for sql, args in pool.statements:
        used = {int(number) for number in re.findall(r"\$(\d+)", sql)}
        assert used == set(range(1, len(args) + 1)), sql


async def test_postgres_delete_cascades_in_one_transaction() -> None:
    rows = [
        {"message_id": 5, "chunk_id": 9},
        {"message_id": 6, "chunk_id": 9},
        {"message_id": 7, "chunk_id": None},
    ]
    pool = ScriptedPool(rows)

    removed = await PostgresStore(pool).delete_messages(CHANNEL, [5, 6, 7, 7, 8])  # type: ignore[arg-type]

    assert removed == 3
    assert_placeholders_match(pool)
    assert pool.transactions == 1 and all(pool.in_transaction)
    [select, chunks, summary, facts, delete] = pool.statements
    assert select[0].startswith("SELECT message_id, chunk_id FROM messages")
    assert "channel_id = $1" in select[0] and "FOR UPDATE" in select[0]
    assert select[1] == (CHANNEL, [5, 6, 7, 8])
    assert chunks[0].startswith("DELETE FROM chunks") and "channel_id = $1" in chunks[0]
    assert chunks[1] == (CHANNEL, [9])
    assert summary[0].startswith("DELETE FROM channel_summaries")
    assert "channel_id = $1" in summary[0] and "covered_message_id >= $2" in summary[0]
    assert summary[1] == (CHANNEL, 5)  # the oldest deleted message decides
    assert facts[0].startswith("DELETE FROM user_facts") and facts[1] == ([5, 6, 7],)
    assert delete[0].startswith("DELETE FROM messages") and "channel_id = $1" in delete[0]
    assert delete[1] == (CHANNEL, [5, 6, 7])


async def test_postgres_delete_of_unknown_messages_deletes_nothing() -> None:
    pool = ScriptedPool([])
    store = PostgresStore(pool)  # type: ignore[arg-type]

    assert await store.delete_messages(CHANNEL, [5]) == 0
    assert await store.delete_messages(CHANNEL, []) == 0

    assert [sql.split()[0] for sql, _ in pool.statements] == ["SELECT"]


async def test_postgres_delete_without_chunks_skips_the_chunk_statement() -> None:
    pool = ScriptedPool([{"message_id": 5, "chunk_id": None}])

    await PostgresStore(pool).delete_messages(CHANNEL, [5])  # type: ignore[arg-type]

    assert not any(sql.startswith("DELETE FROM chunks") for sql, _ in pool.statements)


async def test_postgres_edit_cascades_in_one_transaction() -> None:
    pool = ScriptedPool([], row={"chunk_id": 9})

    assert await PostgresStore(pool).edit_message(CHANNEL, 5, "new text") is True  # type: ignore[arg-type]

    assert_placeholders_match(pool)
    assert pool.transactions == 1 and all(pool.in_transaction)
    [update, chunks, summary, facts] = pool.statements
    assert update[0].startswith("UPDATE messages SET content = $3")
    assert "channel_id = $1 AND message_id = $2" in update[0]
    assert "content IS DISTINCT FROM $3" in update[0]
    assert update[1] == (CHANNEL, 5, "new text")
    assert chunks[1] == (CHANNEL, [9])
    assert summary[1] == (CHANNEL, 5)
    assert facts[1] == ([5],)


async def test_postgres_edit_that_changes_nothing_stops_after_the_update() -> None:
    pool = ScriptedPool([], row=None)

    assert await PostgresStore(pool).edit_message(CHANNEL, 5, "same") is False  # type: ignore[arg-type]

    assert len(pool.statements) == 1


async def test_postgres_delete_facts_from_reports_affected_rows() -> None:
    pool = RecordingPool()  # answers every execute with "DELETE 2"
    store = PostgresStore(pool)  # type: ignore[arg-type]

    assert await store.delete_facts_from([1, 2]) == 2
    assert await store.delete_facts_from([]) == 0

    [(sql, args)] = pool.statements
    assert "source_message_id = ANY($1::bigint[])" in sql and args == ([1, 2],)


def test_the_schema_frees_chunk_siblings_without_a_new_migration() -> None:
    from bot.memory.postgres import MIGRATIONS_DIR

    schema = "\n".join(path.read_text(encoding="utf-8") for path in MIGRATIONS_DIR.glob("*.sql"))
    assert "chunk_id    BIGINT REFERENCES chunks (chunk_id) ON DELETE SET NULL" in schema
