"""Groups stored messages into chunks and embeds them for retrieval.

A chunk is a run of consecutive messages from one channel that are close in
time. A run ends at a long pause, or when it grows too large.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bot.memory.store import MemoryStore, NewChunk, StoredMessage
from bot.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    gap: timedelta = timedelta(minutes=5)  # a pause longer than this ends a chunk
    max_messages: int = 20
    max_chars: int = 2000


def format_line(message: StoredMessage) -> str:
    return f"{message.author_name}: {message.content}"


def transcript(messages: Sequence[StoredMessage]) -> str:
    return "\n".join(format_line(message) for message in messages)


def split_into_groups(
    messages: Sequence[StoredMessage], config: ChunkingConfig, *, now: datetime
) -> list[tuple[list[StoredMessage], bool]]:
    """Split time-ordered messages into ``(group, is_closed)`` pairs.

    Only the last group can still be open: it is closed once it is full or the
    channel has been quiet for longer than ``config.gap``.
    """
    groups: list[list[StoredMessage]] = []
    current: list[StoredMessage] = []
    chars = 0
    for message in messages:
        size = len(format_line(message)) + 1
        if current and (
            message.created_at - current[-1].created_at > config.gap
            or len(current) >= config.max_messages
            or chars + size > config.max_chars
        ):
            groups.append(current)
            current, chars = [], 0
        current.append(message)
        chars += size
    if not current:
        return []

    last_is_closed = (
        now - current[-1].created_at > config.gap
        or len(current) >= config.max_messages
        or chars >= config.max_chars
    )
    return [(group, True) for group in groups] + [(current, last_is_closed)]


class Chunker:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        *,
        config: ChunkingConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        batch_size: int = 200,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or ChunkingConfig()
        self._clock = clock
        self._batch_size = batch_size

    async def chunk_channel(self, channel_id: int) -> int:
        """Embed and store every closed group in the channel. Returns how many."""
        pending = await self._store.unchunked_messages(channel_id, limit=self._batch_size)
        closed = [
            group
            for group, is_closed in split_into_groups(pending, self._config, now=self._clock())
            if is_closed
        ]
        if not closed:
            return 0

        texts = [transcript(group) for group in closed]
        vectors = await self._embedder.embed(texts)
        for group, text, vector in zip(closed, texts, vectors, strict=True):
            await self._store.add_chunk(
                NewChunk(
                    guild_id=group[0].guild_id,
                    channel_id=channel_id,
                    message_ids=tuple(message.message_id for message in group),
                    started_at=group[0].created_at,
                    ended_at=group[-1].created_at,
                    content=text,
                    embedding=vector,
                )
            )
        logger.debug("stored chunks", extra={"channel_id": channel_id, "chunks": len(closed)})
        return len(closed)

    async def flush(self) -> int:
        """Chunk every channel that has pending messages (closes stale open groups)."""
        total = 0
        for channel_id in await self._store.channels_with_unchunked():
            total += await self.chunk_channel(channel_id)
        return total
