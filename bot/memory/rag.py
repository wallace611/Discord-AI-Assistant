"""Retrieval over embedded message chunks: vector search plus optional rerank."""

from collections.abc import Sequence
from typing import Protocol

from bot.memory.store import ChunkHit, MemoryStore
from bot.providers.base import EmbeddingProvider


class Reranker(Protocol):
    """Optional second-stage scorer. Returns one relevance score per document."""

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


def jump_url(hit: ChunkHit) -> str:
    """Discord link to the first message of the chunk."""
    guild = hit.guild_id if hit.guild_id is not None else "@me"
    return f"https://discord.com/channels/{guild}/{hit.channel_id}/{hit.first_message_id}"


class Retriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        *,
        top_k: int = 4,
        candidates: int = 12,
        min_score: float = 0.45,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._candidates = candidates
        self._min_score = min_score
        self._reranker = reranker

    async def search(
        self,
        query: str,
        *,
        guild_id: int | None,
        channel_id: int,
        newer_than_excluded: int | None = None,
        top_k: int | None = None,
    ) -> list[ChunkHit]:
        """Best-matching chunks visible from (guild_id, channel_id).

        The store enforces the scope, so other guilds and other DMs are never
        searched. Chunks of this channel ending at or after ``newer_than_excluded``
        are dropped because the caller already has those messages verbatim.
        """
        if not query.strip():
            return []
        [vector] = await self._embedder.embed([query])
        hits = await self._store.search_chunks(
            vector, guild_id=guild_id, channel_id=channel_id, limit=self._candidates
        )
        hits = [
            hit
            for hit in hits
            if hit.score >= self._min_score
            and not (
                newer_than_excluded is not None
                and hit.channel_id == channel_id
                and hit.last_message_id >= newer_than_excluded
            )
        ]
        if self._reranker is not None and hits:
            scores = await self._reranker.rerank(query, [hit.content for hit in hits])
            ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: pair[0], reverse=True)
            hits = [hit for _, hit in ranked]
        return hits[: min(top_k or self._top_k, self._candidates)]
