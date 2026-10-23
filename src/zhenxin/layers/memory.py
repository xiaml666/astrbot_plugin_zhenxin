"""Long-term fact bank with per-turn retrieval.

The ten hot slots answer "what matters right now"; this table answers "what was
ever true about this person".  Retrieval drops one or two matching facts into
the per-turn tail, so memory depth no longer costs prefix cache and facts that
fell out of the recent window stop being forgotten.

Everything here is local: no embeddings API, no LLM call.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from ..aux.compression import keywords_for_seed
from ..db import Database
from ..embeddings import LocalScorer

MAX_FACTS = 300
# Calibrated on the bundled dialogue: related questions score 0.20-0.60,
# unrelated ones below 0.17 ("我不吃什么来着，突然忘了" -> "不吃香菜" = 0.204).
MIN_SCORE = 0.18
MAX_CHARS = 60

# Zero-value cleanup.  Everything below is rule-based, so the sweep costs one
# small query and no embeddings/LLM calls.
ARCHIVE_MIN_AGE_DAYS = 30.0
ARCHIVE_SCORE = 0.35
MAX_ARCHIVE_PER_SWEEP = 50


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def fact_value(
    importance: float,
    used_count: int,
    created_at: float,
    last_used: float,
    now: float,
) -> float:
    """0..1 estimate of how much a fact is still worth recalling.

    Importance is what the writer knew at insert time; usage is evidence that
    it actually matched later questions; recency decays with a 30-day half
    life so a stale throwaway line eventually drops out.
    """
    idle_days = max(0.0, (now - (last_used or created_at)) / 86400.0)
    recency = 0.5 ** (idle_days / 30.0)
    usage = min(1.0, math.log1p(max(0, int(used_count))) / math.log(6.0))
    return 0.55 * _clamp(importance) + 0.25 * usage + 0.20 * recency



class FactBank:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        clock: Callable[[], float] = time.time,
        limit: int = MAX_FACTS,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.clock = clock
        self.limit = max(20, int(limit))
        self.min_score = float(min_score)

    def add(
        self, user_key: str, text: str, kind: str = "fact", importance: float = 0.5
    ) -> bool:
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            return False
        existing = self.db.fetchone(
            "SELECT id,importance FROM facts WHERE user_key=? AND text=?",
            (user_key, text),
        )
        if existing is not None:
            # Mentioning it again is weak reinforcement; it does not create a
            # second row, but it should not be treated as zero-value either.
            self.db.execute(
                "UPDATE facts SET importance=MIN(1.0, importance+0.05) WHERE id=?",
                (existing["id"],),
            )
            return False
        now = self.clock()
        self.db.execute(
            "INSERT OR IGNORE INTO facts"
            "(user_key,text,kind,importance,created_at,last_used,used_count)"
            " VALUES(?,?,?,?,?,0,0)",
            (user_key, text, kind, _clamp(importance), now),
        )
        count = int(self.db.scalar("SELECT COUNT(*) FROM facts WHERE user_key=?", (user_key,), 0))
        if count > self.limit:
            self._archive_overflow(user_key, count - self.limit)
        return True

    def _rows(self, user_key: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE user_key=?" if user_key else ""
        params = (user_key,) if user_key else ()
        rows = self.db.fetchall(
            "SELECT id,user_key,text,kind,importance,created_at,last_used,used_count"
            f" FROM facts {where}",
            params,
        )
        return [dict(row) for row in rows]

    def _value(self, row: dict[str, Any], now: float) -> float:
        return fact_value(
            row.get("importance", 0.5),
            row.get("used_count", 0),
            row.get("created_at", now),
            row.get("last_used", 0),
            now,
        )

    def _archive_rows(self, rows: list[dict[str, Any]], reason: str) -> int:
        """Delete full sentences, keep a keyword-only trace."""
        if not rows:
            return 0
        now = self.clock()
        users = set()
        for row in rows:
            users.add(row["user_key"])
            keywords = keywords_for_seed(row["text"], 5)
            self.db.execute(
                "INSERT INTO fact_traces"
                "(user_key,keywords,kind,importance,created_at,archived_at,reason)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    row["user_key"],
                    " / ".join(keywords),
                    row.get("kind", "fact"),
                    row.get("importance", 0.5),
                    row.get("created_at", now),
                    now,
                    reason,
                ),
            )
            self.db.execute("DELETE FROM facts WHERE id=?", (row["id"],))
        # Traces are a paper trail, not a second memory bank: keep them bounded.
        for user_key in users:
            self.db.execute(
                "DELETE FROM fact_traces WHERE user_key=? AND id NOT IN"
                " (SELECT id FROM fact_traces WHERE user_key=?"
                " ORDER BY archived_at DESC, id DESC LIMIT 500)",
                (user_key, user_key),
            )
        return len(rows)

    def _archive_overflow(self, user_key: str, extra: int) -> int:
        now = self.clock()
        rows = sorted(self._rows(user_key), key=lambda row: self._value(row, now))
        return self._archive_rows(rows[: max(1, extra)], "overflow")

    def sweep(
        self,
        user_key: str | None = None,
        *,
        now: float | None = None,
        min_age_days: float = ARCHIVE_MIN_AGE_DAYS,
        score_threshold: float = ARCHIVE_SCORE,
        max_archive: int = MAX_ARCHIVE_PER_SWEEP,
    ) -> int:
        """Drop facts that nobody used and that aged past the freshness floor.

        Called automatically every N rounds (and by the memory page).  No
        embedding and no model call: one SELECT plus a couple of writes.
        """
        now = self.clock() if now is None else float(now)
        cutoff = now - max(0.0, float(min_age_days)) * 86400.0
        candidates = [
            row
            for row in self._rows(user_key)
            if row["created_at"] <= cutoff and self._value(row, now) < score_threshold
        ]
        candidates.sort(key=lambda row: self._value(row, now))
        return self._archive_rows(candidates[: max(0, int(max_archive))], "aged_out")

    def traces(self, user_key: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        where = "WHERE user_key=?" if user_key else ""
        params: tuple[Any, ...] = (user_key,) if user_key else ()
        rows = self.db.fetchall(
            "SELECT user_key,keywords,kind,importance,created_at,archived_at,reason"
            f" FROM fact_traces {where} ORDER BY archived_at DESC LIMIT ?",
            params + (max(1, int(limit)),),
        )
        return [dict(row) for row in rows]

    def trace_count(self, user_key: str | None = None) -> int:
        if user_key:
            return int(
                self.db.scalar(
                    "SELECT COUNT(*) FROM fact_traces WHERE user_key=?", (user_key,), 0
                )
            )
        return int(self.db.scalar("SELECT COUNT(*) FROM fact_traces", (), 0))

    def recall(
        self, user_key: str, query: str, k: int = 2, min_score: float | None = None
    ) -> list[str]:
        if not user_key or not query.strip():
            return []
        threshold = self.min_score if min_score is None else float(min_score)
        rows = self.db.fetchall(
            "SELECT id,text,used_count FROM facts WHERE user_key=?", (user_key,)
        )
        if not rows:
            return []
        scored: list[tuple[float, dict]] = []
        for row in rows:
            score = self.scorer.similarity(query, row["text"])
            if score >= threshold:
                scored.append((score, dict(row)))
        if not scored:
            return []
        # Best match first; among equals, prefer facts that were never used.
        scored.sort(key=lambda item: (-item[0], item[1]["used_count"], len(item[1]["text"])))
        chosen = [item[1] for item in scored[: max(1, k)]]
        now = self.clock()
        for row in chosen:
            self.db.execute(
                "UPDATE facts SET used_count=used_count+1,last_used=? WHERE id=?",
                (now, row["id"]),
            )
        return [row["text"] for row in chosen]

    def count(self, user_key: str) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM facts WHERE user_key=?", (user_key,), 0))
