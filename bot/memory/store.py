"""Storage interface for memory, plus a non-persistent in-process implementation.

Scoping rule used everywhere: a guild message is visible within its guild; a DM
(``guild_id is None``) is visible only within its own channel. Data never crosses
guilds or DM channels.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredMessage:
    message_id: int  # Discord snowflake; increases with time
    guild_id: int | None
    channel_id: int  # the thread id when the message is in a thread
    author_id: int
    author_name: str
    is_bot: bool  # True only for this bot's own replies
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class NewChunk:
    guild_id: int | None
    channel_id: int
    message_ids: tuple[int, ...]
    started_at: datetime
    ended_at: datetime
    content: str
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: int
    guild_id: int | None
    channel_id: int
    first_message_id: int
    last_message_id: int
    started_at: datetime
    content: str
    score: float  # cosine similarity, higher is closer


@dataclass(frozen=True, slots=True)
class ChannelSummary:
    channel_id: int
    summary: str
    covered_message_id: int  # the summary covers messages up to and including this id


@dataclass(frozen=True, slots=True)
class UserFact:
    fact_id: int
    user_id: int
    guild_id: int | None
    fact: str
    created_at: datetime


class MemoryStore(Protocol):
    async def add_message(self, message: StoredMessage) -> None:
        """Store a message. A message id that already exists is ignored."""
        ...

    async def recent_messages(
        self, channel_id: int, *, after_id: int = 0, before_id: int | None = None, limit: int
    ) -> list[StoredMessage]:
        """The newest ``limit`` messages with ``after_id < id < before_id``, oldest first."""
        ...

    async def count_messages(self, channel_id: int, *, after_id: int = 0) -> int: ...

    async def get_summary(self, channel_id: int) -> ChannelSummary | None: ...

    async def set_summary(
        self, channel_id: int, guild_id: int | None, summary: str, covered_message_id: int
    ) -> None: ...

    async def unchunked_messages(self, channel_id: int, *, limit: int) -> list[StoredMessage]:
        """The oldest ``limit`` messages not yet part of a chunk, oldest first."""
        ...

    async def channels_with_unchunked(self) -> list[int]: ...

    async def add_chunk(self, chunk: NewChunk) -> None:
        """Store a chunk and mark its messages as chunked, atomically."""
        ...

    async def search_chunks(
        self, embedding: Sequence[float], *, guild_id: int | None, channel_id: int, limit: int
    ) -> list[ChunkHit]:
        """Nearest chunks inside the caller's scope, best first."""
        ...

    async def list_facts(self, user_id: int, guild_id: int | None) -> list[UserFact]: ...

    async def add_fact(
        self, user_id: int, guild_id: int | None, fact: str, source_message_id: int | None
    ) -> bool:
        """Store a fact. Returns False when the same fact already exists."""
        ...

    async def delete_fact(self, user_id: int, fact_id: int) -> bool:
        """Delete one fact, only if it belongs to ``user_id``."""
        ...

    async def delete_all_facts(self, user_id: int, guild_id: int | None) -> int: ...

    async def close(self) -> None: ...


def in_scope(
    guild_id: int | None, channel_id: int, row_guild_id: int | None, row_channel_id: int
) -> bool:
    """Whether a stored row is visible to a caller in (guild_id, channel_id)."""
    if guild_id is None:
        return row_guild_id is None and row_channel_id == channel_id
    return row_guild_id == guild_id


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm


class InMemoryStore:
    """``MemoryStore`` kept in process memory.

    Used by tests and as a fallback when no database is configured. Everything
    is lost on restart, and search is a linear scan.
    """

    def __init__(self) -> None:
        self._messages: dict[int, StoredMessage] = {}
        self._chunked: set[int] = set()
        self._chunks: list[tuple[int, NewChunk]] = []
        self._summaries: dict[int, ChannelSummary] = {}
        self._facts: dict[int, UserFact] = {}
        self._next_id = 1

    def _channel(self, channel_id: int) -> list[StoredMessage]:
        rows = [m for m in self._messages.values() if m.channel_id == channel_id]
        return sorted(rows, key=lambda m: m.message_id)

    async def add_message(self, message: StoredMessage) -> None:
        self._messages.setdefault(message.message_id, message)

    async def recent_messages(
        self, channel_id: int, *, after_id: int = 0, before_id: int | None = None, limit: int
    ) -> list[StoredMessage]:
        rows = [
            m
            for m in self._channel(channel_id)
            if m.message_id > after_id and (before_id is None or m.message_id < before_id)
        ]
        return rows[-limit:] if limit > 0 else []

    async def count_messages(self, channel_id: int, *, after_id: int = 0) -> int:
        return sum(1 for m in self._channel(channel_id) if m.message_id > after_id)

    async def get_summary(self, channel_id: int) -> ChannelSummary | None:
        return self._summaries.get(channel_id)

    async def set_summary(
        self, channel_id: int, guild_id: int | None, summary: str, covered_message_id: int
    ) -> None:
        self._summaries[channel_id] = ChannelSummary(channel_id, summary, covered_message_id)

    async def unchunked_messages(self, channel_id: int, *, limit: int) -> list[StoredMessage]:
        rows = [m for m in self._channel(channel_id) if m.message_id not in self._chunked]
        return rows[:limit]

    async def channels_with_unchunked(self) -> list[int]:
        pending = {
            m.channel_id for m in self._messages.values() if m.message_id not in self._chunked
        }
        return sorted(pending)

    async def add_chunk(self, chunk: NewChunk) -> None:
        self._chunks.append((self._next_id, chunk))
        self._next_id += 1
        self._chunked.update(chunk.message_ids)

    async def search_chunks(
        self, embedding: Sequence[float], *, guild_id: int | None, channel_id: int, limit: int
    ) -> list[ChunkHit]:
        hits = [
            ChunkHit(
                chunk_id=chunk_id,
                guild_id=chunk.guild_id,
                channel_id=chunk.channel_id,
                first_message_id=chunk.message_ids[0],
                last_message_id=chunk.message_ids[-1],
                started_at=chunk.started_at,
                content=chunk.content,
                score=_cosine(embedding, chunk.embedding),
            )
            for chunk_id, chunk in self._chunks
            if in_scope(guild_id, channel_id, chunk.guild_id, chunk.channel_id)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def list_facts(self, user_id: int, guild_id: int | None) -> list[UserFact]:
        rows = [f for f in self._facts.values() if f.user_id == user_id and f.guild_id == guild_id]
        return sorted(rows, key=lambda f: f.fact_id)

    async def add_fact(
        self, user_id: int, guild_id: int | None, fact: str, source_message_id: int | None
    ) -> bool:
        existing = await self.list_facts(user_id, guild_id)
        if any(f.fact.casefold() == fact.casefold() for f in existing):
            return False
        fact_id = self._next_id
        self._next_id += 1
        self._facts[fact_id] = UserFact(fact_id, user_id, guild_id, fact, datetime.now(UTC))
        return True

    async def delete_fact(self, user_id: int, fact_id: int) -> bool:
        fact = self._facts.get(fact_id)
        if fact is None or fact.user_id != user_id:
            return False
        del self._facts[fact_id]
        return True

    async def delete_all_facts(self, user_id: int, guild_id: int | None) -> int:
        doomed = [f.fact_id for f in await self.list_facts(user_id, guild_id)]
        for fact_id in doomed:
            del self._facts[fact_id]
        return len(doomed)

    async def close(self) -> None:
        return None
