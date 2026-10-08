"""Record how long review sessions take (no GUI imports).

One row per reviewer session in ``review_sessions.csv`` beside the review state. Active time
adds up the gaps between consecutive reviewer actions, ignoring any gap longer than the idle
cutoff, so a break does not count as review effort. The cutoff is stored with every row.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.review.storage import atomic_write_dataframe

SESSIONS_FILENAME = "review_sessions.csv"
DEFAULT_IDLE_CUTOFF_SECONDS = 120.0
COLUMNS = (
    "session_id",
    "reviewer",
    "mode",
    "started_at",
    "last_activity_at",
    "n_decisions",
    "active_seconds",
    "idle_cutoff_seconds",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewTimer:
    """Accumulates active review time and keeps its row in the sessions table current."""

    def __init__(
        self,
        directory: Path,
        *,
        reviewer: str,
        mode: str,
        idle_cutoff_seconds: float = DEFAULT_IDLE_CUTOFF_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.path = Path(directory) / SESSIONS_FILENAME
        self.session_id = uuid.uuid4().hex
        self.reviewer = reviewer
        self.mode = mode
        self.idle_cutoff_seconds = float(idle_cutoff_seconds)
        self._clock = clock
        self._last = clock()
        self.started_at = _now()
        self.last_activity_at = self.started_at
        self.n_decisions = 0
        self.active_seconds = 0.0

    def activity(self, *, decision: bool = False) -> None:
        """Note a reviewer action (a decision, undo, or move) and save the running totals."""
        now = self._clock()
        gap = now - self._last
        if 0 <= gap <= self.idle_cutoff_seconds:
            self.active_seconds += gap
        self._last = now
        self.last_activity_at = _now()
        if decision:
            self.n_decisions += 1
        self.save()

    def row(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "reviewer": self.reviewer,
            "mode": self.mode,
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
            "n_decisions": self.n_decisions,
            "active_seconds": round(self.active_seconds, 1),
            "idle_cutoff_seconds": self.idle_cutoff_seconds,
        }

    def save(self) -> None:
        if self.path.is_file():
            table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
            table = table[table["session_id"] != self.session_id]
        else:
            table = pd.DataFrame(columns=list(COLUMNS))
        row = pd.DataFrame([{key: str(value) for key, value in self.row().items()}])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_dataframe(self.path, pd.concat([table, row], ignore_index=True))
