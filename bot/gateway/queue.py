"""Bounded event queue with a worker pool and per-key (per-channel) ordering."""

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable, Hashable

logger = logging.getLogger(__name__)


class EventQueue[T]:
    """Runs ``handler`` for each submitted event.

    Events with different keys run concurrently, up to ``workers`` at a time.
    Events with the same key run one at a time, in submission order.
    """

    def __init__(
        self,
        handler: Callable[[T], Awaitable[None]],
        *,
        key: Callable[[T], Hashable],
        workers: int = 4,
        maxsize: int = 256,
    ) -> None:
        self._handler = handler
        self._key = key
        self._workers = workers
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
        # Weak values: a lock disappears once no worker holds or waits on it.
        self._locks: weakref.WeakValueDictionary[Hashable, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def is_idle(self) -> bool:
        """True when nothing is queued or being handled."""
        return self._in_flight == 0 and self._queue.empty()

    async def wait_idle(self) -> None:
        """Block until the queue has nothing queued or in flight."""
        await self._idle.wait()

    def start(self) -> None:
        """Spawn the workers. Must be called from inside a running event loop."""
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._worker(), name=f"event-worker-{index}")
            for index in range(self._workers)
        ]

    def submit(self, event: T) -> bool:
        """Enqueue ``event``. Returns False when the queue is full."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            return False
        self._idle.clear()
        return True

    async def join(self) -> None:
        """Wait until every submitted event has been handled."""
        await self._queue.join()

    async def stop(self) -> None:
        """Cancel the workers. Queued and in-flight events are dropped."""
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _lock_for(self, key: Hashable) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            self._in_flight += 1
            try:
                # No await between dequeue and lock acquisition, and asyncio.Lock
                # is FIFO, so same-key events keep their submission order.
                async with self._lock_for(self._key(event)):
                    await self._handler(event)
            except Exception:
                logger.exception("event handler failed")
            finally:
                self._in_flight -= 1
                self._queue.task_done()
                if self.is_idle:
                    self._idle.set()
