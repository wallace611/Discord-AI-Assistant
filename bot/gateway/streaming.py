"""Streams model output into Discord by editing a message at a fixed interval."""

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import discord

DISCORD_MESSAGE_LIMIT = 2000
# Discord allows roughly 5 edits per 5 seconds per channel; stay well under it.
DEFAULT_EDIT_INTERVAL = 1.2
EMPTY_REPLY = "（沒有產生任何內容）"

NO_MENTIONS = discord.AllowedMentions.none()


class Editable(Protocol):
    """The part of ``discord.Message`` that streaming needs."""

    id: int

    async def edit(self, *, content: str, allowed_mentions: discord.AllowedMentions) -> Any: ...


type SendFn = Callable[[str], Awaitable[Editable]]


def _split(text: str, limit: int) -> tuple[str, str]:
    """Split ``text`` into a head of at most ``limit`` chars and the rest.

    Prefers to cut at a newline, then at a space, and only then mid-word.
    """
    window = text[:limit]
    for separator in ("\n", " "):
        cut = window.rfind(separator)
        if cut > 0:
            return text[:cut], text[cut + 1 :]
    return window, text[limit:]


class StreamingReply:
    """Accumulates text deltas and mirrors them to Discord.

    The first delta is sent immediately; later ones are batched so the message is
    edited at most once per ``edit_interval`` seconds. Text beyond the Discord
    length limit continues in a new message.
    """

    def __init__(
        self,
        send: SendFn,
        *,
        edit_interval: float = DEFAULT_EDIT_INTERVAL,
        limit: int = DISCORD_MESSAGE_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send = send
        self._interval = edit_interval
        self._limit = limit
        self._clock = clock
        self._text = ""  # full text of the message currently being streamed
        self._shown = ""  # what Discord currently displays for that message
        self._message: Editable | None = None
        self._last_flush = -math.inf
        self._sent_any = False
        # Id of the first Discord message this reply produced, once there is one.
        self.first_message_id: int | None = None
        self._status = ""

    async def feed(self, delta: str) -> None:
        """Append ``delta``; push to Discord if the edit interval has elapsed."""
        self._text += delta
        await self._maybe_flush()

    async def set_status(self, status: str) -> None:
        """Show a transient line (such as the tool in use) under the text so far.

        The line disappears on ``finish`` or when set to an empty string.
        """
        self._status = status.strip()
        await self._maybe_flush()

    async def finish(self) -> None:
        """Push any remaining text. Shows a placeholder if nothing was produced."""
        self._status = ""
        await self._flush()
        if self._text.strip():
            return
        if self._message is not None:
            # Only a status line was ever shown; replace it rather than leave it.
            await self._show(EMPTY_REPLY)
        elif not self._sent_any:
            await self._send(EMPTY_REPLY)
            self._sent_any = True

    async def _maybe_flush(self) -> None:
        if self._clock() - self._last_flush >= self._interval:
            await self._flush()

    async def _flush(self) -> None:
        while len(self._text) > self._limit:
            head, self._text = _split(self._text, self._limit)
            await self._show(head)
            # The head is final; the remainder continues in a fresh message.
            self._message = None
            self._shown = ""
        display = self._text
        if self._status:
            display = f"{display}\n\n{self._status}" if display.strip() else self._status
        await self._show(display)
        self._last_flush = self._clock()

    async def _show(self, content: str) -> None:
        # Discord rejects blank messages, and identical edits waste rate limit.
        if not content.strip() or content == self._shown:
            return
        if self._message is None:
            self._message = await self._send(content)
            self._sent_any = True
            if self.first_message_id is None:
                self.first_message_id = self._message.id
        else:
            await self._message.edit(content=content, allowed_mentions=NO_MENTIONS)
        self._shown = content
