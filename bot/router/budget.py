"""Per-user daily cloud token budget.

In-memory only: counters reset on restart and at UTC midnight. Persistence is
part of the hardening phase.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime


def _utc_today() -> date:
    return datetime.now(UTC).date()


class DailyTokenBudget:
    def __init__(self, limit: int | None, *, today: Callable[[], date] = _utc_today) -> None:
        """``limit=None`` means unlimited; ``limit=0`` disables cloud usage."""
        self._limit = limit
        self._today = today
        self._day = today()
        self._used: dict[int, int] = {}

    def _roll_over(self) -> None:
        current = self._today()
        if current != self._day:
            self._day = current
            self._used.clear()

    def used(self, user_id: int) -> int:
        self._roll_over()
        return self._used.get(user_id, 0)

    def is_exhausted(self, user_id: int) -> bool:
        return self._limit is not None and self.used(user_id) >= self._limit

    def record(self, user_id: int, tokens: int) -> None:
        self._roll_over()
        self._used[user_id] = self._used.get(user_id, 0) + max(tokens, 0)
