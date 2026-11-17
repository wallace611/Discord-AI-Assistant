"""Facade that ties storage, chunking, summaries, retrieval and user facts together."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from bot.memory.ingest import Chunker
from bot.memory.rag import Reranker, Retriever
from bot.memory.short_term import ShortTermMemory
from bot.memory.store import ChunkHit, MemoryStore, StoredMessage, UserFact
from bot.memory.user_facts import UserFacts
from bot.providers.base import ChatProvider, EmbeddingProvider, ProviderError

logger = logging.getLogger(__name__)

TRACKED_LIMIT = 10_000


def _track[V](seen: dict[int, V], message_ids: Sequence[int], value: V) -> None:
    """Remember ``message_ids`` in ``seen``, dropping the oldest beyond ``TRACKED_LIMIT``."""
    for message_id in message_ids:
        seen.pop(message_id, None)  # re-insert so the newest entry is last
        seen[message_id] = value
    while len(seen) > TRACKED_LIMIT:
        del seen[next(iter(seen))]


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Everything memory contributes to one reply."""

    summary: str | None = None
    recent: tuple[StoredMessage, ...] = ()
    hits: tuple[ChunkHit, ...] = ()
    facts: tuple[UserFact, ...] = ()


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        *,
        chat: ChatProvider,
        embedder: EmbeddingProvider,
        chat_model: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self.short_term = ShortTermMemory(store, chat, model=chat_model)
        self.chunker = Chunker(store, embedder)
        self.retriever = Retriever(store, embedder, reranker=reranker)
        self.facts = UserFacts(store, chat, model=chat_model)
        # Chunking and summarising share one local GPU; run them one at a time.
        self._maintenance = asyncio.Lock()
        # Short writes to messages. Lock order: ``_maintenance`` before ``_store_lock``.
        self._store_lock = asyncio.Lock()
        # Recent deletions and edits, so a write that is still queued cannot bring a
        # deleted message back or store its old text. Bounded: oldest are forgotten.
        self._deleted: dict[int, None] = {}
        self._edits: dict[int, str] = {}

    async def record(self, message: StoredMessage) -> None:
        """Store a message without doing any model work.

        A message that was deleted or edited before it got here (the gateway events
        can overtake the queued write) is skipped or stored with its new text.
        """
        async with self._store_lock:
            if message.message_id in self._deleted:
                return
            edited = self._edits.get(message.message_id)
            if edited is not None and edited != message.content:
                message = replace(message, content=edited)
            await self._store.add_message(message)

    async def forget_messages(self, channel_id: int, message_ids: Sequence[int]) -> int:
        """Drop messages deleted on Discord and everything derived from them.

        Returns how many were stored. Waits for any chunking or summarising in
        progress, so a chunk built from a deleted message cannot be saved afterwards.
        """
        _track(self._deleted, message_ids, None)
        async with self._maintenance, self._store_lock:
            return await self._store.delete_messages(channel_id, message_ids)

    async def edit_message(self, channel_id: int, message_id: int, content: str) -> bool:
        """Apply an edit made on Discord. Returns False when the stored text is unchanged."""
        _track(self._edits, [message_id], content)
        async with self._maintenance, self._store_lock:
            return await self._store.edit_message(channel_id, message_id, content)

    async def extract_facts(self, message: StoredMessage) -> list[str]:
        """Extract user facts from ``message``, unless it was deleted or edited meanwhile."""
        if self._is_stale(message):
            return []
        added = await self.facts.extract(message)
        if added and self._is_stale(message):
            # The message changed while the model was reading it; the facts are not
            # about what it says now.
            async with self._store_lock:
                await self._store.delete_facts_from([message.message_id])
            return []
        return added

    def _is_stale(self, message: StoredMessage) -> bool:
        if message.message_id in self._deleted:
            return True
        return self._edits.get(message.message_id, message.content) != message.content

    async def maintain(self, channel_id: int, guild_id: int | None) -> None:
        """Chunk and summarise a channel if due. Uses the local model."""
        async with self._maintenance:
            await self.chunker.chunk_channel(channel_id)
            await self.short_term.maybe_summarize(channel_id, guild_id)

    async def ingest(self, message: StoredMessage) -> None:
        """Store a message, then chunk and summarise its channel if due."""
        await self.record(message)
        await self.maintain(message.channel_id, message.guild_id)

    async def flush(self) -> int:
        """Close chunks in channels that have gone quiet. Returns chunks stored."""
        async with self._maintenance:
            return await self.chunker.flush()

    async def context(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        query: str,
        before_message_id: int | None = None,
    ) -> MemoryContext:
        """Gather what the model should know before answering ``query``."""
        summary, recent = await self.short_term.load(channel_id, before_id=before_message_id)
        facts = await self.facts.view(user_id, guild_id)
        try:
            hits = await self.retriever.search(
                query,
                guild_id=guild_id,
                channel_id=channel_id,
                newer_than_excluded=recent[0].message_id if recent else before_message_id,
            )
        except ProviderError:
            # Retrieval is a bonus; answer without it when embedding is unavailable.
            logger.warning("retrieval failed, continuing without it", exc_info=True)
            hits = []
        return MemoryContext(
            summary=summary, recent=tuple(recent), hits=tuple(hits), facts=tuple(facts)
        )

    async def close(self) -> None:
        await self._store.close()
