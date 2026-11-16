"""Entry point: ``python -m bot``."""

import asyncio
import logging
import sys

from pydantic import ValidationError

from bot.config import Settings, get_settings
from bot.gateway import BotClient, MessageHandler
from bot.logging_setup import configure_logging
from bot.memory import InMemoryStore, MemoryService, MemoryStore
from bot.providers import build_cloud_provider, build_local_provider
from bot.router import DailyTokenBudget, Router

logger = logging.getLogger("bot")


class StartupError(Exception):
    """The bot cannot start; the message says what to fix."""


async def open_store(settings: Settings) -> MemoryStore:
    if settings.database_url is None:
        logger.warning("DATABASE_URL is not set: memory is kept in process and lost on restart")
        return InMemoryStore()
    # Imported lazily: the driver is only needed when a database is configured.
    import asyncpg

    from bot.memory.postgres import PostgresStore

    try:
        return await PostgresStore.connect(settings.database_url.get_secret_value())
    except (OSError, asyncpg.PostgresError) as exc:
        raise StartupError(
            f"cannot use the database at DATABASE_URL ({type(exc).__name__}: {exc}). "
            "Start it with `docker compose up -d`, or leave DATABASE_URL empty to run "
            "with non-persistent in-process memory."
        ) from exc


async def run(settings: Settings) -> None:
    local = build_local_provider(settings)
    cloud = build_cloud_provider(settings)
    memory: MemoryService | None = None
    try:
        memory = MemoryService(
            await open_store(settings),
            chat=local,
            embedder=local,
            chat_model=settings.local_router_model,
        )
        handler = MessageHandler(
            router=Router(local, model=settings.local_router_model),
            local=local,
            cloud=cloud,
            budget=DailyTokenBudget(settings.daily_cloud_token_budget_per_user),
            memory=memory,
        )
        client = BotClient(handler, memory)
        async with client:
            await client.start(settings.discord_token.get_secret_value())
    finally:
        if memory is not None:
            await memory.close()
        await local.aclose()
        await cloud.aclose()


def main() -> int:
    configure_logging()
    try:
        settings = get_settings()
    except ValidationError as exc:
        # Report which variables are wrong without echoing their values.
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in exc.errors()
        ]
        logger.error("invalid configuration", extra={"problems": problems})
        return 1
    try:
        asyncio.run(run(settings))
    except StartupError as exc:
        logger.error("startup failed", extra={"reason": str(exc)})
        return 1
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
