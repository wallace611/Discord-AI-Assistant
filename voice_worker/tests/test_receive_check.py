import threading

import pytest

from voice_worker.receive_check import (
    DAVE_CLOSE_CODE,
    ReceiveStats,
    Verdict,
    evaluate,
)


def stats_with(*records: tuple[int | None, str, int]) -> ReceiveStats:
    stats = ReceiveStats()
    for user_id, name, pcm in records:
        stats.record(user_id, name, pcm)
    return stats


def test_counts_packets_and_bytes_per_speaker() -> None:
    stats = stats_with((1, "alice", 3840), (1, "alice", 3840), (2, "bob", 1920))

    speakers = stats.speakers()

    assert (speakers[1].packets, speakers[1].pcm_bytes) == (2, 7680)
    assert (speakers[2].packets, speakers[2].pcm_bytes) == (1, 1920)
    assert (stats.total_packets, stats.total_pcm_bytes) == (3, 9600)


def test_unknown_speakers_are_grouped_together() -> None:
    stats = stats_with((None, "unknown", 10), (None, "unknown", 10))

    assert stats.speakers()[None].packets == 2


def test_speakers_returns_a_snapshot() -> None:
    stats = stats_with((1, "alice", 10))

    snapshot = stats.speakers()
    stats.record(1, "alice", 10)

    assert snapshot[1].packets == 1


def test_recording_from_many_threads_loses_nothing() -> None:
    stats = ReceiveStats()

    def worker() -> None:
        for _ in range(1000):
            stats.record(1, "alice", 2)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert (stats.total_packets, stats.total_pcm_bytes) == (8000, 16000)


def test_audio_received_passes() -> None:
    outcome = evaluate(stats_with((1, "alice", 3840)), connected=True)

    assert outcome.verdict is Verdict.PASS
    assert outcome.passed
    assert "1 packets / 3840 PCM bytes" in outcome.message


def test_connected_without_packets_fails() -> None:
    outcome = evaluate(ReceiveStats(), connected=True)

    assert outcome.verdict is Verdict.NO_PACKETS
    assert not outcome.passed


def test_packets_without_pcm_fails_separately() -> None:
    outcome = evaluate(stats_with((1, "alice", 0), (1, "alice", 0)), connected=True)

    assert outcome.verdict is Verdict.NO_PCM
    assert "2 packets" in outcome.message


def test_close_code_4017_is_reported_as_missing_dave_support() -> None:
    outcome = evaluate(ReceiveStats(), connected=False, close_code=DAVE_CLOSE_CODE)

    assert outcome.verdict is Verdict.DAVE_UNSUPPORTED
    assert "DAVE" in outcome.message
    assert "davey" in outcome.message


def test_4017_wins_even_if_audio_was_counted() -> None:
    outcome = evaluate(stats_with((1, "a", 10)), connected=True, close_code=DAVE_CLOSE_CODE)

    assert outcome.verdict is Verdict.DAVE_UNSUPPORTED


@pytest.mark.parametrize("close_code", [None, 4014, 4006])
def test_other_connection_failures_are_not_blamed_on_dave(close_code: int | None) -> None:
    outcome = evaluate(ReceiveStats(), connected=False, close_code=close_code, error="boom")

    assert outcome.verdict is Verdict.CONNECT_FAILED
    assert "boom" in outcome.message
    if close_code is not None:
        assert str(close_code) in outcome.message
