"""Facade that ties storage, chunking, summaries, retrieval and user facts together."""

import asyncio
import logging
from dataclasses import dataclass

from bot.memory.ingest import Chunker
from bot.memory.rag import Reranker, Retriever
from bot.memory.short_term import ShortTermMemory
from bot.memory.store import ChunkHit, MemoryStore, StoredMessage, UserFact
from bot.memory.user_facts import UserFacts
from bot.providers.base import ChatProvider, EmbeddingProvider, ProviderError

logger = logging.getLogger(__name__)


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

    async def record(self, message: StoredMessage) -> None:
        """Store a message without doing any model work."""
        await self._store.add_message(message)

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
