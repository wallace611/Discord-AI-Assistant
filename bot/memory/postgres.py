"""``MemoryStore`` backed by Postgres + pgvector."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import asyncpg

from bot.memory.store import ChannelSummary, ChunkHit, NewChunk, StoredMessage, UserFact

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_MESSAGE_COLUMNS = (
    "message_id, guild_id, channel_id, author_id, author_name, is_bot, content, created_at"
)

# Guild rows are visible inside their guild ($2); DM rows only inside their channel ($3).
_SCOPE_FILTER = """(
    ($2::bigint IS NOT NULL AND guild_id = $2)
    OR ($2::bigint IS NULL AND guild_id IS NULL AND channel_id = $3)
)"""


def _vector_literal(embedding: Sequence[float]) -> str:
    # Sent as text and cast with ``::text::vector`` so no custom codec is needed.
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


def _affected(status: str) -> int:
    """Row count from a command tag such as ``DELETE 3``."""
    return int(status.rsplit(" ", 1)[-1])


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str, *, migrations_dir: Path = MIGRATIONS_DIR) -> Self:
        """Open a connection pool and apply any pending migrations."""
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        store = cls(pool)
        await store.migrate(migrations_dir)
        return store

    async def migrate(self, migrations_dir: Path) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            rows = await conn.fetch("SELECT version FROM schema_migrations")
            applied = {row["version"] for row in rows}
            for path in sorted(migrations_dir.glob("*.sql")):
                if path.name in applied:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text(encoding="utf-8"))
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)", path.name
                    )
                logger.info("applied migration", extra={"migration": path.name})

    async def add_message(self, message: StoredMessage) -> None:
        await self._pool.execute(
            f"INSERT INTO messages ({_MESSAGE_COLUMNS}) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (message_id) DO NOTHING",
            message.message_id,
            message.guild_id,
            message.channel_id,
            message.author_id,
            message.author_name,
            message.is_bot,
            message.content,
            message.created_at,
        )

    async def recent_messages(
        self, channel_id: int, *, after_id: int = 0, before_id: int | None = None, limit: int
    ) -> list[StoredMessage]:
        rows = await self._pool.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM ("
            f"  SELECT {_MESSAGE_COLUMNS} FROM messages"
            "   WHERE channel_id = $1 AND message_id > $2"
            "     AND ($3::bigint IS NULL OR message_id < $3)"
            "   ORDER BY message_id DESC LIMIT $4"
            ") AS newest ORDER BY message_id",
            channel_id,
            after_id,
            before_id,
            limit,
        )
        return [StoredMessage(**dict(row)) for row in rows]

    async def count_messages(self, channel_id: int, *, after_id: int = 0) -> int:
        count: Any = await self._pool.fetchval(
            "SELECT count(*) FROM messages WHERE channel_id = $1 AND message_id > $2",
            channel_id,
            after_id,
        )
        return int(count)

    async def get_summary(self, channel_id: int) -> ChannelSummary | None:
        row = await self._pool.fetchrow(
            "SELECT channel_id, summary, covered_message_id FROM channel_summaries "
            "WHERE channel_id = $1",
            channel_id,
        )
        return ChannelSummary(**dict(row)) if row else None

    async def set_summary(
        self, channel_id: int, guild_id: int | None, summary: str, covered_message_id: int
    ) -> None:
        await self._pool.execute(
            "INSERT INTO channel_summaries (channel_id, guild_id, summary, covered_message_id) "
            "VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (channel_id) DO UPDATE SET summary = EXCLUDED.summary, "
            "covered_message_id = EXCLUDED.covered_message_id, updated_at = now()",
            channel_id,
            guild_id,
            summary,
            covered_message_id,
        )

    async def unchunked_messages(self, channel_id: int, *, limit: int) -> list[StoredMessage]:
        rows = await self._pool.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages "
            "WHERE channel_id = $1 AND chunk_id IS NULL ORDER BY message_id LIMIT $2",
            channel_id,
            limit,
        )
        return [StoredMessage(**dict(row)) for row in rows]

    async def channels_with_unchunked(self) -> list[int]:
        rows = await self._pool.fetch(
            "SELECT DISTINCT channel_id FROM messages WHERE chunk_id IS NULL ORDER BY channel_id"
        )
        return [row["channel_id"] for row in rows]

    async def add_chunk(self, chunk: NewChunk) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            chunk_id = await conn.fetchval(
                "INSERT INTO chunks (guild_id, channel_id, first_message_id, last_message_id, "
                "started_at, ended_at, content, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text::vector) RETURNING chunk_id",
                chunk.guild_id,
                chunk.channel_id,
                chunk.message_ids[0],
                chunk.message_ids[-1],
                chunk.started_at,
                chunk.ended_at,
                chunk.content,
                _vector_literal(chunk.embedding),
            )
            await conn.execute(
                "UPDATE messages SET chunk_id = $1 WHERE message_id = ANY($2::bigint[])",
                chunk_id,
                list(chunk.message_ids),
            )

    async def search_chunks(
        self, embedding: Sequence[float], *, guild_id: int | None, channel_id: int, limit: int
    ) -> list[ChunkHit]:
        rows = await self._pool.fetch(
            "SELECT chunk_id, guild_id, channel_id, first_message_id, last_message_id, "
            "started_at, content, 1 - (embedding <=> $1::text::vector) AS score "
            f"FROM chunks WHERE {_SCOPE_FILTER} "
            "ORDER BY embedding <=> $1::text::vector LIMIT $4",
            _vector_literal(embedding),
            guild_id,
            channel_id,
            limit,
        )
        return [ChunkHit(**dict(row)) for row in rows]

    @staticmethod
    async def _invalidate(
        conn: Any, channel_id: int, message_ids: Sequence[int], chunk_ids: Sequence[int]
    ) -> None:
        """Drop what was derived from these messages. Runs inside the caller's transaction."""
        if chunk_ids:
            # messages.chunk_id is ON DELETE SET NULL, so the chunk's other messages
            # become unchunked and are chunked again later.
            await conn.execute(
                "DELETE FROM chunks WHERE channel_id = $1 AND chunk_id = ANY($2::bigint[])",
                channel_id,
                list(chunk_ids),
            )
        # A summary cannot be un-summarised; rebuild it from the remaining messages.
        await conn.execute(
            "DELETE FROM channel_summaries WHERE channel_id = $1 AND covered_message_id >= $2",
            channel_id,
            min(message_ids),
        )
        await conn.execute(
            "DELETE FROM user_facts WHERE source_message_id = ANY($1::bigint[])",
            list(message_ids),
        )

    async def delete_messages(self, channel_id: int, message_ids: Sequence[int]) -> int:
        wanted = list(dict.fromkeys(message_ids))
        if not wanted:
            return 0
        async with self._pool.acquire() as conn, conn.transaction():
            rows = await conn.fetch(
                "SELECT message_id, chunk_id FROM messages "
                "WHERE channel_id = $1 AND message_id = ANY($2::bigint[]) FOR UPDATE",
                channel_id,
                wanted,
            )
            if not rows:
                return 0
            found = [row["message_id"] for row in rows]
            chunk_ids = sorted({row["chunk_id"] for row in rows if row["chunk_id"] is not None})
            await self._invalidate(conn, channel_id, found, chunk_ids)
            await conn.execute(
                "DELETE FROM messages WHERE channel_id = $1 AND message_id = ANY($2::bigint[])",
                channel_id,
                found,
            )
            return len(found)

    async def edit_message(self, channel_id: int, message_id: int, content: str) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "UPDATE messages SET content = $3 "
                "WHERE channel_id = $1 AND message_id = $2 AND content IS DISTINCT FROM $3 "
                "RETURNING chunk_id",
                channel_id,
                message_id,
                content,
            )
            if row is None:
                return False
            chunk_ids = [row["chunk_id"]] if row["chunk_id"] is not None else []
            await self._invalidate(conn, channel_id, [message_id], chunk_ids)
            return True

    async def delete_facts_from(self, message_ids: Sequence[int]) -> int:
        if not message_ids:
            return 0
        status = await self._pool.execute(
            "DELETE FROM user_facts WHERE source_message_id = ANY($1::bigint[])",
            list(message_ids),
        )
        return _affected(status)

    async def list_facts(self, user_id: int, guild_id: int | None) -> list[UserFact]:
        rows = await self._pool.fetch(
            "SELECT fact_id, user_id, guild_id, fact, created_at FROM user_facts "
            "WHERE user_id = $1 AND guild_id IS NOT DISTINCT FROM $2 ORDER BY fact_id",
            user_id,
            guild_id,
        )
        return [UserFact(**dict(row)) for row in rows]

    async def add_fact(
        self, user_id: int, guild_id: int | None, fact: str, source_message_id: int | None
    ) -> bool:
        fact_id = await self._pool.fetchval(
            "INSERT INTO user_facts (user_id, guild_id, fact, source_message_id) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING RETURNING fact_id",
            user_id,
            guild_id,
            fact,
            source_message_id,
        )
        return fact_id is not None

    async def delete_fact(self, user_id: int, fact_id: int) -> bool:
        status = await self._pool.execute(
            "DELETE FROM user_facts WHERE user_id = $1 AND fact_id = $2", user_id, fact_id
        )
        return _affected(status) > 0

    async def delete_all_facts(self, user_id: int, guild_id: int | None) -> int:
        status = await self._pool.execute(
            "DELETE FROM user_facts WHERE user_id = $1 AND guild_id IS NOT DISTINCT FROM $2",
            user_id,
            guild_id,
        )
        return _affected(status)

    async def close(self) -> None:
        await self._pool.close()
