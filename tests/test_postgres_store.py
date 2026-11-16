"""Static checks for ``PostgresStore``.

These run against a recording fake, not a real database: they catch placeholder
and argument mismatches and check that the scope filter is applied, but they do
not prove the SQL is valid. That needs an integration run against Postgres.
"""

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from bot.memory import InMemoryStore, MemoryStore, NewChunk
from bot.memory.postgres import MIGRATIONS_DIR, PostgresStore
from fakes import T0, stored


class RecordingPool:
    """Records every statement; doubles as its own connection."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def _record(self, sql: str, args: tuple[Any, ...]) -> None:
        self.statements.append((sql, args))

    async def execute(self, sql: str, *args: Any) -> str:
        self._record(sql, args)
        return "DELETE 2"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._record(sql, args)
        return []

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self._record(sql, args)
        return None

    async def fetchval(self, sql: str, *args: Any) -> int:
        self._record(sql, args)
        return 1

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self

    @asynccontextmanager
    async def transaction(self) -> Any:
        yield None


CHUNK = NewChunk(
    guild_id=200,
    channel_id=100,
    message_ids=(1, 2),
    started_at=T0,
    ended_at=T0,
    content="alice: hi",
    embedding=[0.1, 0.2],
)


async def exercise(store: MemoryStore) -> None:
    await store.add_message(stored(1))
    await store.recent_messages(100, after_id=0, before_id=5, limit=10)
    await store.count_messages(100, after_id=0)
    await store.get_summary(100)
    await store.set_summary(100, 200, "summary", 1)
    await store.unchunked_messages(100, limit=10)
    await store.channels_with_unchunked()
    await store.add_chunk(CHUNK)
    await store.search_chunks([0.1, 0.2], guild_id=200, channel_id=100, limit=5)
    await store.list_facts(7, 200)
    await store.add_fact(7, 200, "A fact.", 1)
    await store.delete_fact(7, 1)
    await store.delete_all_facts(7, None)


async def test_every_statement_has_matching_placeholders_and_arguments() -> None:
    pool = RecordingPool()
    await exercise(PostgresStore(pool))  # type: ignore[arg-type]

    assert len(pool.statements) == 14  # add_chunk issues two statements
    for sql, args in pool.statements:
        used = {int(number) for number in re.findall(r"\$(\d+)", sql)}
        assert used == set(range(1, len(args) + 1)), sql


async def test_the_same_calls_work_on_the_in_memory_store() -> None:
    await exercise(InMemoryStore())


async def test_search_is_scoped_and_sends_the_vector_as_text() -> None:
    pool = RecordingPool()
    store = PostgresStore(pool)  # type: ignore[arg-type]

    await store.search_chunks([0.5, 1.0], guild_id=None, channel_id=100, limit=3)

    sql, args = pool.statements[0]
    assert "guild_id = $2" in sql
    assert "guild_id IS NULL AND channel_id = $3" in sql
    assert args == ("[0.5,1.0]", None, 100, 3)


async def test_deletes_report_affected_rows_and_are_owner_scoped() -> None:
    pool = RecordingPool()
    store = PostgresStore(pool)  # type: ignore[arg-type]

    assert await store.delete_fact(7, 3) is True
    assert await store.delete_all_facts(7, 200) == 2

    for sql, args in pool.statements:
        assert "user_id = $1" in sql
        assert args[0] == 7


async def test_migrate_applies_only_new_files_in_order(tmp_path: Path) -> None:
    (tmp_path / "002_b.sql").write_text("SELECT 2;", encoding="utf-8")
    (tmp_path / "001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "003_c.sql").write_text("SELECT 3;", encoding="utf-8")

    class Pool(RecordingPool):
        async def fetch(self, sql: str, *args: Any) -> list[Any]:
            self._record(sql, args)
            return [{"version": "001_a.sql"}]

    pool = Pool()
    await PostgresStore(pool).migrate(tmp_path)  # type: ignore[arg-type]

    executed = [sql for sql, _ in pool.statements]
    assert "SELECT 1;" not in executed
    assert executed.index("SELECT 2;") < executed.index("SELECT 3;")
    recorded = [args[0] for sql, args in pool.statements if sql.startswith("INSERT INTO schema")]
    assert recorded == ["002_b.sql", "003_c.sql"]


def test_shipped_migration_defines_the_tables_the_store_uses() -> None:
    sql = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS_DIR.glob("*.sql"))
    )
    for table in ("chunks", "messages", "channel_summaries", "user_facts"):
        assert f"CREATE TABLE {table}" in sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql


@pytest.mark.parametrize("name", [name for name in dir(InMemoryStore) if not name.startswith("_")])
def test_postgres_store_implements_the_same_interface(name: str) -> None:
    assert callable(getattr(PostgresStore, name, None))
