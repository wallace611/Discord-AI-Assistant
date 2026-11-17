"""Check that a Py-cord bot can join a voice channel and receive audio under DAVE.

Environment variables:
    DISCORD_TOKEN             bot token
    VERIFY_GUILD_ID           guild that owns the channel
    VERIFY_VOICE_CHANNEL_ID   voice channel to join

Someone must speak in the channel while the script listens. Exit codes:
0 = PASS, 1 = FAIL (see the log for the reason), 2 = bad configuration.
"""

import asyncio
import contextlib
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discord  # noqa: E402  (Py-cord; must come after the path tweak above)
from discord.sinks import Sink  # noqa: E402

from voice_worker.receive_check import (  # noqa: E402
    DAVE_CLOSE_CODE,
    Outcome,
    ReceiveStats,
    Verdict,
    evaluate,
)

logger = logging.getLogger("verify_voice_receive")

LISTEN_SECONDS = 20.0
CONNECT_TIMEOUT = 30.0
READY_TIMEOUT = 60.0


class CountingSink(Sink):
    """Counts packets and PCM bytes per speaker and keeps no audio.

    Py-cord 2.8.1's receive pipeline looks up ``__sink_listeners__`` and calls
    ``walk_children()`` on the sink, but the stock ``Sink`` defines neither, so
    every built-in sink fails with AttributeError when recording starts. Both are
    defined here as empty.
    """

    __sink_listeners__: list[tuple[str, str]] = []

    def __init__(self, stats: ReceiveStats) -> None:
        super().__init__()
        self._stats = stats

    def walk_children(self) -> Any:
        return iter(())

    def write(self, data: Any, user: Any) -> None:
        # Called from Py-cord's packet router thread with a VoiceData object.
        user_id = getattr(user, "id", None)
        name = str(user) if user is not None else "unknown"
        self._stats.record(user_id, name, len(getattr(data, "pcm", b"") or b""))

    def cleanup(self) -> None:
        self.finished = True


class ConfigError(Exception):
    """The environment is missing settings or has invalid ones."""


@dataclass(frozen=True, slots=True)
class Config:
    token: str
    guild_id: int
    channel_id: int


def load_config() -> Config:
    missing = [
        name
        for name in ("DISCORD_TOKEN", "VERIFY_GUILD_ID", "VERIFY_VOICE_CHANNEL_ID")
        if not os.environ.get(name)
    ]
    if missing:
        raise ConfigError(f"missing environment variables: {', '.join(missing)}")
    try:
        guild_id = int(os.environ["VERIFY_GUILD_ID"])
        channel_id = int(os.environ["VERIFY_VOICE_CHANNEL_ID"])
    except ValueError:
        raise ConfigError("VERIFY_GUILD_ID and VERIFY_VOICE_CHANNEL_ID must be integers") from None
    return Config(os.environ["DISCORD_TOKEN"], guild_id, channel_id)


def log_dave_state(voice: Any, label: str) -> None:
    logger.info(
        "DAVE state (%s): dave_session=%s, privacy_code=%s",
        label,
        voice.is_dave_connection(),
        voice.privacy_code,
    )


async def listen(bot: discord.Bot, config: Config, stats: ReceiveStats) -> Outcome:
    channel = await bot.fetch_channel(config.channel_id)
    if not isinstance(channel, discord.VoiceChannel | discord.StageChannel):
        return Outcome(
            Verdict.CONNECT_FAILED, f"channel {config.channel_id} is not a voice channel"
        )
    if channel.guild.id != config.guild_id:
        return Outcome(Verdict.CONNECT_FAILED, "the channel does not belong to VERIFY_GUILD_ID")

    logger.info("Joining #%s in %s", channel.name, channel.guild.name)
    try:
        # reconnect=False so a 4017 close surfaces here instead of being retried.
        voice = await channel.connect(timeout=CONNECT_TIMEOUT, reconnect=False)
    except discord.ConnectionClosed as exc:
        return evaluate(stats, connected=False, close_code=exc.code)
    except (TimeoutError, discord.ClientException, discord.HTTPException) as exc:
        return evaluate(stats, connected=False, error=f"{type(exc).__name__}: {exc}")

    try:
        log_dave_state(voice, "after connect")
        logger.info("Listening for %.0f seconds - speak in the channel now", LISTEN_SECONDS)
        # Py-cord warns that receive is broken under DAVE; the warning is expected.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            voice.start_recording(CountingSink(stats))
        await asyncio.sleep(LISTEN_SECONDS)
        log_dave_state(voice, "after listening")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            voice.stop_recording()
    except discord.ConnectionClosed as exc:
        return evaluate(stats, connected=False, close_code=exc.code)
    except Exception as exc:
        logger.exception("Recording failed")
        return Outcome(Verdict.CONNECT_FAILED, f"recording failed: {type(exc).__name__}: {exc}")
    finally:
        with contextlib.suppress(Exception):
            if voice.is_recording():
                voice.stop_recording()
        with contextlib.suppress(Exception):
            await voice.disconnect(force=True)

    return evaluate(stats, connected=True)


async def run(config: Config) -> Outcome:
    stats = ReceiveStats()
    bot = discord.Bot(intents=discord.Intents(guilds=True, voice_states=True))
    runner = asyncio.create_task(bot.start(config.token))
    try:
        await asyncio.wait_for(bot.wait_until_ready(), READY_TIMEOUT)
        outcome = await listen(bot, config, stats)
    except TimeoutError:
        outcome = Outcome(Verdict.CONNECT_FAILED, "the bot did not become ready in time")
    except discord.LoginFailure:
        outcome = Outcome(Verdict.CONNECT_FAILED, "Discord rejected the token")
    finally:
        await bot.close()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await runner

    for user_id, speaker in stats.speakers().items():
        logger.info(
            "speaker %s (%s): %d packets, %d PCM bytes",
            speaker.name,
            user_id,
            speaker.packets,
            speaker.pcm_bytes,
        )
    return outcome


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    outcome = asyncio.run(run(config))
    if outcome.passed:
        logger.info("RESULT: PASS - %s", outcome.message)
        return 0
    logger.error("RESULT: FAIL (%s) - %s", outcome.verdict.value, outcome.message)
    if outcome.verdict is Verdict.NO_PACKETS:
        logger.error(
            "Py-cord 2.8.1 itself warns that voice reception is broken under DAVE "
            "(https://github.com/Pycord-Development/pycord/issues/3139). If someone really "
            "spoke, this FAIL is likely that known limitation (close code %d did not occur).",
            DAVE_CLOSE_CODE,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
