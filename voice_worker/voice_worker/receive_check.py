"""Pure bookkeeping for the voice-receive verification: counting and the verdict.

Kept free of any Discord import so it can be unit tested without a connection.
"""

import threading
from dataclasses import dataclass
from enum import Enum

# Discord closes the voice websocket with this code when the client does not
# support DAVE (end-to-end encryption), which has been mandatory since 2026-03-01.
DAVE_CLOSE_CODE = 4017


class Verdict(Enum):
    PASS = "pass"
    NO_PACKETS = "no_packets"  # connected, nothing arrived
    NO_PCM = "no_pcm"  # packets arrived but decoded to no audio
    DAVE_UNSUPPORTED = "dave_unsupported"  # closed with 4017
    CONNECT_FAILED = "connect_failed"  # could not join for another reason


@dataclass(slots=True)
class SpeakerStats:
    name: str
    packets: int = 0
    pcm_bytes: int = 0


class ReceiveStats:
    """Per-speaker packet and PCM byte counts. Safe to call from other threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speakers: dict[int | None, SpeakerStats] = {}

    def record(self, user_id: int | None, name: str, pcm_bytes: int) -> None:
        with self._lock:
            speaker = self._speakers.setdefault(user_id, SpeakerStats(name))
            speaker.packets += 1
            speaker.pcm_bytes += pcm_bytes

    def speakers(self) -> dict[int | None, SpeakerStats]:
        with self._lock:
            return {
                user_id: SpeakerStats(s.name, s.packets, s.pcm_bytes)
                for user_id, s in self._speakers.items()
            }

    @property
    def total_packets(self) -> int:
        with self._lock:
            return sum(s.packets for s in self._speakers.values())

    @property
    def total_pcm_bytes(self) -> int:
        with self._lock:
            return sum(s.pcm_bytes for s in self._speakers.values())


@dataclass(frozen=True, slots=True)
class Outcome:
    verdict: Verdict
    message: str

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS


def evaluate(
    stats: ReceiveStats,
    *,
    connected: bool,
    close_code: int | None = None,
    error: str | None = None,
) -> Outcome:
    """Turn what happened during the run into a verdict with a readable reason."""
    if close_code == DAVE_CLOSE_CODE:
        return Outcome(
            Verdict.DAVE_UNSUPPORTED,
            f"Discord closed the voice connection with {DAVE_CLOSE_CODE}: this client does not "
            "support DAVE. Check that 'davey' is installed in this environment.",
        )
    if not connected:
        detail = f" ({error})" if error else ""
        code = f" close code {close_code}," if close_code is not None else ""
        return Outcome(Verdict.CONNECT_FAILED, f"Could not join the voice channel{code}{detail}.")
    if stats.total_packets == 0:
        return Outcome(
            Verdict.NO_PACKETS,
            "Connected, but received 0 audio packets. Either nobody spoke during the window, "
            "or voice receive is broken (decryption/DAVE).",
        )
    if stats.total_pcm_bytes == 0:
        return Outcome(
            Verdict.NO_PCM,
            f"Received {stats.total_packets} packets but none decoded to PCM audio "
            "(decryption or Opus decoding is failing).",
        )
    return Outcome(
        Verdict.PASS,
        f"Received {stats.total_packets} packets / {stats.total_pcm_bytes} PCM bytes "
        f"from {len(stats.speakers())} speaker(s).",
    )
