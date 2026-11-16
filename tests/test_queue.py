import asyncio
import logging

import pytest

from bot.gateway.queue import EventQueue

Event = tuple[int, int]  # (channel_id, sequence)


def channel_of(event: Event) -> int:
    return event[0]


async def test_same_channel_events_run_in_order_without_overlap() -> None:
    order: list[int] = []
    active = 0
    max_active = 0

    async def handler(event: Event) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        # Earlier events take longer, so any overlap would reorder completions.
        await asyncio.sleep(0.01 * (5 - event[1]))
        order.append(event[1])
        active -= 1

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, workers=4)
    queue.start()
    for sequence in range(5):
        assert queue.submit((1, sequence))
    await queue.join()
    await queue.stop()

    assert order == [0, 1, 2, 3, 4]
    assert max_active == 1


async def test_different_channels_run_concurrently() -> None:
    both_running = asyncio.Event()
    running: set[int] = set()

    async def handler(event: Event) -> None:
        running.add(event[0])
        if len(running) == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=1)

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, workers=2)
    queue.start()
    queue.submit((1, 0))
    queue.submit((2, 0))
    await asyncio.wait_for(queue.join(), timeout=2)
    await queue.stop()

    assert running == {1, 2}


async def test_handler_exception_is_logged_and_worker_survives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handled: list[int] = []

    async def handler(event: Event) -> None:
        if event[1] == 0:
            raise RuntimeError("boom")
        handled.append(event[1])

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, workers=1)
    queue.start()
    with caplog.at_level(logging.ERROR, logger="bot.gateway.queue"):
        queue.submit((1, 0))
        queue.submit((1, 1))
        await queue.join()
    await queue.stop()

    assert handled == [1]
    assert any("event handler failed" in record.getMessage() for record in caplog.records)


async def test_submit_returns_false_when_full() -> None:
    async def handler(event: Event) -> None:
        return None

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, maxsize=2)
    assert queue.submit((1, 0)) is True
    assert queue.submit((1, 1)) is True
    assert queue.submit((1, 2)) is False


async def test_wait_idle_blocks_while_work_is_queued_or_running() -> None:
    release = asyncio.Event()

    async def handler(event: Event) -> None:
        await release.wait()

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, workers=1)
    assert queue.is_idle is True
    queue.start()

    queue.submit((1, 0))
    queue.submit((1, 1))
    assert queue.is_idle is False
    waiter = asyncio.create_task(queue.wait_idle())
    await asyncio.sleep(0.01)
    assert waiter.done() is False  # first event is running, second is queued

    release.set()
    await asyncio.wait_for(waiter, timeout=1)
    assert queue.is_idle is True
    await queue.stop()


async def test_channel_locks_are_released_after_use() -> None:
    async def handler(event: Event) -> None:
        return None

    queue: EventQueue[Event] = EventQueue(handler, key=channel_of, workers=2)
    queue.start()
    for channel in range(20):
        queue.submit((channel, 0))
    await queue.join()
    await queue.stop()

    assert len(queue._locks) == 0
