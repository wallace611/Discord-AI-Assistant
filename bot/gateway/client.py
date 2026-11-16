"""Discord gateway client: receives events and feeds the work queues."""

import asyncio
import logging
from dataclasses import dataclass

import discord
from discord import app_commands

from bot.gateway.commands import register_memory_commands
from bot.gateway.events import IncomingMessage, to_event, to_stored
from bot.gateway.handler import MessageHandler
from bot.gateway.queue import EventQueue
from bot.gateway.streaming import NO_MENTIONS
from bot.memory import MemoryService, StoredMessage

logger = logging.getLogger(__name__)

BUSY_NOTICE = "目前排隊中的訊息太多，請稍後再試。"


@dataclass(frozen=True, slots=True)
class _ReplyJob:
    message: discord.Message
    event: IncomingMessage


@dataclass(frozen=True, slots=True)
class _MemoryJob:
    message: StoredMessage
    addressed: bool  # the user was talking to the bot, so look for facts about them


class BotClient(discord.Client):
    def __init__(
        self,
        handler: MessageHandler,
        memory: MemoryService,
        *,
        workers: int = 4,
        queue_size: int = 256,
        flush_interval: float = 60.0,
    ) -> None:
        intents = discord.Intents.default()
        # Privileged intent: must also be enabled in the Discord developer portal.
        intents.message_content = True
        super().__init__(intents=intents, allowed_mentions=NO_MENTIONS)
        self._handler = handler
        self._memory = memory
        self._flush_interval = flush_interval
        self._flush_task: asyncio.Task[None] | None = None

        self.tree = app_commands.CommandTree(self)
        register_memory_commands(self.tree, memory.facts)

        self._replies: EventQueue[_ReplyJob] = EventQueue(
            self._reply,
            key=lambda job: job.event.channel_id,
            workers=workers,
            maxsize=queue_size,
        )
        # One worker: memory upkeep shares the local GPU with interactive replies.
        self._memory_jobs: EventQueue[_MemoryJob] = EventQueue(
            self._remember,
            key=lambda job: job.message.channel_id,
            workers=1,
            maxsize=2000,
        )

    async def setup_hook(self) -> None:
        self._replies.start()
        self._memory_jobs.start()
        self._flush_task = asyncio.create_task(self._flush_loop(), name="memory-flush")
        await self.tree.sync()

    async def close(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)
            self._flush_task = None
        await self._replies.stop()
        await self._memory_jobs.stop()
        await super().close()

    async def on_ready(self) -> None:
        logger.info(
            "connected to Discord",
            extra={"bot_user": str(self.user), "guild_count": len(self.guilds)},
        )

    async def on_message(self, message: discord.Message) -> None:
        if self.user is None:
            return
        event = to_event(message, self.user.id, self.user.display_name)
        # The reply is queued first so memory upkeep sees it and waits for it.
        if event is not None and not self._replies.submit(_ReplyJob(message, event)):
            logger.warning("reply queue full, dropping message", extra={"message_id": message.id})
            await message.reply(BUSY_NOTICE, mention_author=False, allowed_mentions=NO_MENTIONS)
        stored = to_stored(message)
        if stored is not None and not self._memory_jobs.submit(
            _MemoryJob(stored, addressed=event is not None)
        ):
            logger.warning(
                "memory queue full, message not stored", extra={"message_id": message.id}
            )

    async def _reply(self, job: _ReplyJob) -> None:
        async with job.message.channel.typing():
            await self._handler.handle(job.event)

    async def _remember(self, job: _MemoryJob) -> None:
        # Storing is cheap and happens at once. The model work (chunking,
        # summarising, fact extraction) shares the GPU with replies, so it waits
        # until no reply is being generated.
        await self._memory.record(job.message)
        await self._replies.wait_idle()
        await self._memory.maintain(job.message.channel_id, job.message.guild_id)
        if job.addressed:
            await self._replies.wait_idle()
            await self._memory.facts.extract(job.message)

    async def _flush_loop(self) -> None:
        """Periodically close chunks in channels that have gone quiet."""
        while True:
            await asyncio.sleep(self._flush_interval)
            try:
                await self._memory.flush()
            except Exception:
                logger.exception("memory flush failed")
