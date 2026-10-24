"""Long-term fact bank with weighted retrieval and a forgetting cycle.

The ten hot slots answer "what matters right now"; this table answers "what was
ever true about this person".  Retrieval drops one or two matching facts into
the per-turn tail, so memory depth no longer costs prefix cache and facts that
fell out of the recent window stop being forgotten.

Two ideas keep the bank from growing into a second prompt:

* **Weighted retrieval** — semantic similarity alone is not enough.  Stated
  importance, Ebbinghaus retention, usage evidence, exact token overlap and a
  kind prior all contribute.  A name the user gave ("叫我阿岚") is always
  injected, not left to a lucky match.
* **Three pools + recycle bin** — every ``cycle_days`` a fact is re-evaluated:
  high value goes to the treasure pool (slower decay), low value to the shallow
  pool (forgotten first), the middle stays in the cycle.  Retractions
  ("不用提醒了", "别叫我阿岚") move a fact to the recycle bin for a few days
  before permanent deletion.

Everything here is local: no embeddings API, no LLM call.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Callable

from .._aux import zero_value
from .._aux.compression import keywords_for_seed
from ..config import MemoryConfig
from ..db import Database
from ..embeddings import LocalScorer

MAX_FACTS = 300
# Calibrated on the bundled dialogue: related questions score 0.20-0.60,
# unrelated ones below 0.17 ("我不吃什么来着，突然忘了" -> "不吃香菜" = 0.204).
MIN_SCORE = 0.30
MAX_CHARS = 60

# Zero-value cleanup.  Everything below is rule-based, so the sweep costs one
# small query and no embeddings/LLM calls.
ARCHIVE_MIN_AGE_DAYS = 30.0
ARCHIVE_SCORE = 0.30
MAX_ARCHIVE_PER_SWEEP = 50

POOLS = ("cycle", "treasure", "shallow", "trash")
POOL_LABELS = {
    "cycle": "周期池",
    "treasure": "珍藏池",
    "shallow": "浅忆池",
    "trash": "回收池",
}
KIND_PRIOR = {
    "address": 1.0,
    "habit": 0.6,
    "relationship": 0.5,
    "unclosed": 0.45,
    "fact": 0.25,
    "emotion": 0.1,
}
POOL_BONUS = {"treasure": 0.05, "cycle": 0.0, "shallow": -0.10, "trash": -1.0}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _row_get(row: Any, key: str, default: Any = 0) -> Any:
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def retention(row: Any, now: float, *, half_life_days: float = 30.0) -> float:
    """Ebbinghaus retention with a spacing effect.

    Each mention/recall multiplies the memory's "life" a little; the treasure
    pool gets a much longer half-life, the shallow pool decays faster.  This is
    a display/ranking heuristic, not a claim of biological accuracy.
    """
    created = float(_row_get(row, "created_at", now)) or now
    last_used = float(_row_get(row, "last_used", 0)) or 0.0
    last_mentioned = float(_row_get(row, "last_mentioned", 0)) or 0.0
    anchor = max(last_used, last_mentioned, created) or created
    idle_days = max(0.0, (now - anchor) / 86400.0)
    importance = _clamp(_row_get(row, "importance", 0.5))
    mentions = max(0, int(_row_get(row, "mentions", 1)))
    used = max(0, int(_row_get(row, "used_count", 0)))
    repeats = min(12, mentions + used)
    pool = str(_row_get(row, "pool", "cycle") or "cycle")
    half = max(1.0, float(half_life_days))
    life = half * (0.55 + 0.35 * importance) * (1.0 + 0.18 * repeats)
    if pool == "treasure":
        life *= 2.5
    elif pool == "shallow":
        life *= 0.55
    return math.exp(-idle_days / max(0.5, life))


def fact_value(
    importance: float,
    used_count: int,
    created_at: float,
    last_used: float,
    now: float,
    *,
    mentions: int = 0,
    manual_boost: float = 0.0,
    pool: str = "cycle",
    half_life_days: float = 30.0,
) -> float:
    """0..1 estimate of how much a fact is still worth recalling."""
    row = {
        "importance": importance,
        "used_count": used_count,
        "created_at": created_at,
        "last_used": last_used,
        "mentions": max(1, int(mentions or 1)),
        "manual_boost": manual_boost,
        "pool": pool,
    }
    recency = retention(row, now, half_life_days=half_life_days)
    usage = min(1.0, math.log1p(max(0, int(used_count))) / math.log(6.0))
    return _clamp(
        0.48 * _clamp(importance)
        + 0.32 * recency
        + 0.12 * usage
        + 0.08 * _clamp(manual_boost)
    )


class FactBank:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        clock: Callable[[], float] = time.time,
        limit: int = MAX_FACTS,
        min_score: float = MIN_SCORE,
        cfg: MemoryConfig | None = None,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.clock = clock
        self.cfg = cfg or MemoryConfig()
        self.limit = max(20, int(limit if cfg is None else cfg.max_facts))
        self.min_score = float(min_score if cfg is None else cfg.min_score)

    # ------------------------------------------------------------------ write
    def add(
        self,
        user_key: str,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
    ) -> bool:
        if not self.cfg.enabled:
            return False
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            return False
        now = self.clock()
        existing = self.db.fetchone(
            "SELECT id,importance,mentions,pool,manual_boost FROM facts"
            " WHERE user_key=? AND text=?",
            (user_key, text),
        )
        if existing is not None:
            # Mentioning it again is real evidence: raise importance, count the
            # mention, and revive it if it had been swept into the recycle bin.
            self.db.execute(
                "UPDATE facts SET importance=MIN(1.0, importance+0.04),"
                " mentions=mentions+1,last_mentioned=?,"
                " pool=CASE WHEN pool='trash' THEN 'cycle' ELSE pool END,"
                " trashed_at=CASE WHEN pool='trash' THEN 0 ELSE trashed_at END,"
                " trash_reason=CASE WHEN pool='trash' THEN '' ELSE trash_reason END"
                " WHERE id=?",
                (now, existing["id"]),
            )
            return False
        weight = _clamp(0.3 + 0.55 * _clamp(importance))
        self.db.execute(
            "INSERT OR IGNORE INTO facts"
            "(user_key,text,kind,importance,created_at,last_used,used_count,"
            " weight,mentions,pool,cycle_at,trashed_at,trash_reason,"
            " last_mentioned,manual_boost)"
            " VALUES(?,?,?,?,?,0,0,?,1,'cycle',?,0,'',?,0)",
            (
                user_key,
                text,
                kind,
                _clamp(importance),
                now,
                weight,
                now,
                now,
            ),
        )
        count = self.count(user_key)
        if count > self.limit:
            self._archive_overflow(user_key, count - self.limit)
        return True

    # ------------------------------------------------------------------- read
    def _rows(self, user_key: str | None = None, *, include_trash: bool = True) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if user_key:
            where.append("user_key=?")
            params.append(user_key)
        if not include_trash:
            where.append("pool<>'trash'")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.db.fetchall(
            "SELECT id,user_key,text,kind,importance,created_at,last_used,used_count,"
            " weight,mentions,pool,cycle_at,trashed_at,trash_reason,"
            " last_mentioned,manual_boost"
            f" FROM facts {clause}",
            tuple(params),
        )
        return [dict(row) for row in rows]

    def _row_strength(self, row: dict[str, Any], now: float) -> float:
        pool = str(row.get("pool") or "cycle")
        if pool == "trash":
            return 0.0
        return _clamp(
            0.36 * _clamp(row.get("importance", 0.5))
            + 0.44 * retention(row, now, half_life_days=self.cfg.decay_half_life_days)
            + 0.12 * min(1.0, math.log1p(max(0, int(row.get("mentions", 1)))) / math.log(8.0))
            + 0.08 * _clamp(row.get("manual_boost", 0.0))
        )

    def weight_of(self, row: dict[str, Any], now: float | None = None) -> float:
        """0..1 colour depth shown by the memory page."""
        now = self.clock() if now is None else now
        strength = self._row_strength(row, now)
        # Manual clicks should visibly deepen the dot even on an old memory.
        return _clamp(strength + 0.28 * _clamp(row.get("manual_boost", 0.0)))

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
        rows = sorted(self._rows(user_key, include_trash=False), key=lambda row: self._row_strength(row, now))
        return self._archive_rows(rows[: max(1, extra)], "overflow")

    # --------------------------------------------------------------- lifecycle
    def _trash_row(self, row: dict[str, Any], reason: str, now: float) -> bool:
        if str(row.get("pool") or "") == "trash":
            return False
        self.db.execute(
            "UPDATE facts SET pool='trash',trashed_at=?,trash_reason=? WHERE id=?",
            (now, reason[:80], row["id"]),
        )
        return True

    def scan_zero_value(self, user_key: str, text: str) -> int:
        """Move explicitly retracted memories into the recycle bin."""
        reason = zero_value.cue(text)
        if not reason:
            return 0
        now = self.clock()
        removed = 0
        for row in self._rows(user_key, include_trash=False):
            if zero_value.matches(row["text"], row.get("kind", "fact"), text, reason):
                if self._trash_row(row, reason, now):
                    removed += 1
        return removed

    def cycle(
        self,
        user_key: str | None = None,
        *,
        now: float | None = None,
    ) -> dict[str, int]:
        """Run one forgetting cycle and repair the pool sizes.

        The treasure/shallow pools are independent of the cycle afterwards: a
        treasure fact stays a treasure fact until the user retracts it or the
        bank overflows, and a shallow fact is only recalled on a strong match.
        """
        now = self.clock() if now is None else float(now)
        rows = self._rows(user_key, include_trash=False)
        changed = {"treasure": 0, "shallow": 0, "cycle": 0, "purged": 0}
        cycle_seconds = max(0.5, float(self.cfg.cycle_days)) * 86400.0
        for row in rows:
            cycle_at = float(row.get("cycle_at") or 0)
            if cycle_at and now - cycle_at < cycle_seconds:
                continue
            strength = self._row_strength(row, now)
            pool = str(row.get("pool") or "cycle")
            if (
                float(row.get("manual_boost") or 0) >= 0.35
                or int(row.get("mentions") or 1) >= 6
                or strength >= float(self.cfg.treasure_threshold)
            ):
                target = "treasure"
            elif strength <= float(self.cfg.shallow_threshold):
                target = "shallow"
            else:
                target = "cycle"
            self.db.execute(
                "UPDATE facts SET pool=?,cycle_at=?,weight=? WHERE id=?",
                (target, now, self.weight_of({**row, "pool": target}, now), row["id"]),
            )
            if target != pool:
                changed[target] += 1
        # Permanent deletion from the recycle bin.
        cutoff = now - max(0.1, float(self.cfg.trash_ttl_days)) * 86400.0
        trash = [
            row
            for row in self._rows(user_key)
            if row.get("pool") == "trash" and float(row.get("trashed_at") or 0) <= cutoff
        ]
        if trash:
            changed["purged"] = self._archive_rows(trash, "trash_expired")
        self._enforce_limits(user_key)
        return changed

    def _enforce_limits(self, user_key: str | None) -> None:
        now = self.clock()
        users = [user_key] if user_key else [
            row[0]
            for row in self.db.fetchall("SELECT DISTINCT user_key FROM facts")
        ]
        for user in users:
            for pool, limit in (
                ("treasure", int(self.cfg.treasure_limit)),
                ("shallow", int(self.cfg.shallow_limit)),
            ):
                rows = [
                    row
                    for row in self._rows(user)
                    if row.get("pool") == pool
                ]
                if len(rows) <= max(1, limit):
                    continue
                rows.sort(key=lambda row: self.weight_of(row, now))
                overflow = rows[: len(rows) - max(1, limit)]
                if pool == "treasure":
                    # Demote the weakest treasure instead of deleting it.
                    for row in overflow:
                        self.db.execute(
                            "UPDATE facts SET pool='cycle' WHERE id=?", (row["id"],)
                        )
                else:
                    self._archive_rows(overflow, "shallow_overflow")

    def sweep(
        self,
        user_key: str | None = None,
        *,
        now: float | None = None,
        min_age_days: float | None = None,
        score_threshold: float | None = None,
        max_archive: int | None = None,
    ) -> int:
        """Cycle + recycle-bin purge, with a legacy deep-clean option.

        Called automatically every N rounds and by the memory page.  No
        embedding and no model call.
        """
        now = self.clock() if now is None else float(now)
        result = self.cycle(user_key, now=now)
        removed = int(result.get("purged", 0))
        if min_age_days is not None:
            cutoff = now - max(0.0, float(min_age_days)) * 86400.0
            threshold = ARCHIVE_SCORE if score_threshold is None else float(score_threshold)
            candidates = [
                row
                for row in self._rows(user_key, include_trash=False)
                if row.get("pool") == "shallow"
                and float(row.get("created_at") or 0) <= cutoff
                and self._row_strength(row, now) < threshold
            ]
            candidates.sort(key=lambda row: self._row_strength(row, now))
            cap = MAX_ARCHIVE_PER_SWEEP if max_archive is None else int(max_archive)
            removed += self._archive_rows(candidates[: max(0, cap)], "aged_out")
        return removed

    # ------------------------------------------------------------------ query
    def recall(
        self,
        user_key: str,
        query: str,
        k: int | None = None,
        min_score: float | None = None,
    ) -> list[str]:
        if not self.cfg.enabled or not user_key or not query.strip():
            return []
        k = max(1, int(k if k is not None else self.cfg.top_k))
        threshold = self.min_score if min_score is None else float(min_score)
        rows = self._rows(user_key, include_trash=False)
        if not rows:
            return []
        q_tokens = zero_value.meaningful_tokens(query)
        now = self.clock()
        scored: list[tuple[float, dict[str, Any]]] = []
        address_rows: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            semantic = self.scorer.similarity(query, row["text"])
            tokens = zero_value.meaningful_tokens(row["text"])
            overlap = (
                len(q_tokens & tokens) / max(1, min(len(q_tokens), len(tokens)))
                if q_tokens and tokens
                else 0.0
            )
            kind = str(row.get("kind") or "fact")
            # Do not recall on importance/recency alone: at least one semantic
            # or lexical signal is required.
            if semantic < 0.12 and overlap <= 0.0 and kind != "address":
                continue
            rec = retention(row, now, half_life_days=self.cfg.decay_half_life_days)
            usage = min(1.0, math.log1p(max(0, int(row.get("used_count", 0)))) / math.log(8.0))
            score = (
                self.cfg.weight_semantic * semantic
                + self.cfg.weight_importance * _clamp(row.get("importance", 0.5))
                + self.cfg.weight_recency * rec
                + self.cfg.weight_usage * usage
                + self.cfg.weight_kind * KIND_PRIOR.get(kind, 0.2)
                + self.cfg.exact_boost * overlap
                + POOL_BONUS.get(str(row.get("pool") or "cycle"), 0.0)
            )
            if kind == "address":
                score += self.cfg.address_boost
                address_rows.append((score, row))
            if score >= threshold:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], len(item[1]["text"])))
        chosen = [row for _, row in scored[:k]]
        # A stated name is not a retrieval lottery: carry it every turn.
        if self.cfg.address_k > 0 and address_rows:
            address_rows.sort(key=lambda item: -item[0])
            best = address_rows[0][1]
            if not any(row["id"] == best["id"] for row in chosen):
                chosen = chosen[: max(0, k - self.cfg.address_k)] + [best]
        if not chosen:
            return []
        for row in chosen:
            self.db.execute(
                "UPDATE facts SET used_count=used_count+1,last_used=? WHERE id=?",
                (now, row["id"]),
            )
        return [row["text"] for row in chosen]

    def address(self, user_key: str) -> str:
        """The best address fact, injected every turn (or an empty string)."""
        if not self.cfg.enabled or not user_key:
            return ""
        row = self.db.fetchone(
            "SELECT text FROM facts WHERE user_key=? AND kind='address'"
            " AND pool<>'trash' ORDER BY importance DESC, mentions DESC, id DESC LIMIT 1",
            (user_key,),
        )
        return str(row["text"]) if row else ""

    def top_facts(self, user_key: str, limit: int = 3, pools: tuple[str, ...] = ("treasure", "cycle")) -> list[dict[str, Any]]:
        now = self.clock()
        rows = [
            row
            for row in self._rows(user_key, include_trash=False)
            if str(row.get("pool") or "cycle") in pools
        ]
        rows.sort(key=lambda row: (-self.weight_of(row, now), -float(row.get("last_mentioned") or 0)))
        return rows[: max(1, int(limit))]

    # -------------------------------------------------------------- page view
    def list_facts(
        self,
        user_key: str | None = None,
        *,
        date: str = "",
        pool: str = "",
        query: str = "",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        now = self.clock()
        rows = self._rows(user_key)
        out: list[dict[str, Any]] = []
        for row in rows:
            if pool and str(row.get("pool") or "cycle") != pool:
                continue
            if date and _date_of(row.get("created_at", 0)) != date:
                continue
            if query and query not in str(row.get("text", "")):
                continue
            out.append(self._public(row, now))
        out.sort(key=lambda item: (-item["created_at"], -item["weight"]))
        return out[: max(1, int(limit or self.cfg.page_limit))]

    def dates(self, user_key: str | None = None) -> list[dict[str, Any]]:
        rows = self._rows(user_key)
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            day = _date_of(row.get("created_at", 0))
            bucket = buckets.setdefault(day, {"date": day, "total": 0, "trash": 0})
            bucket["total"] += 1
            if row.get("pool") == "trash":
                bucket["trash"] += 1
        return [buckets[key] for key in sorted(buckets, reverse=True)]

    def pool_counts(self, user_key: str | None = None) -> dict[str, int]:
        counts = {pool: 0 for pool in POOLS}
        for row in self._rows(user_key):
            pool = str(row.get("pool") or "cycle")
            counts[pool] = counts.get(pool, 0) + 1
        return counts

    def _public(self, row: dict[str, Any], now: float) -> dict[str, Any]:
        pool = str(row.get("pool") or "cycle")
        weight = self.weight_of(row, now)
        created = float(row.get("created_at") or 0)
        return {
            "id": int(row["id"]),
            "text": row["text"],
            "kind": row.get("kind", "fact"),
            "importance": round(_clamp(row.get("importance", 0.5)), 2),
            "weight": round(weight, 3),
            "color": _color(weight),
            "mentions": int(row.get("mentions") or 1),
            "used": int(row.get("used_count") or 0),
            "pool": pool,
            "pool_label": POOL_LABELS.get(pool, "记忆"),
            "created_at": created,
            "last_used": float(row.get("last_used") or 0),
            "last_mentioned": float(row.get("last_mentioned") or 0),
            "trashed_at": float(row.get("trashed_at") or 0),
            "trash_reason": str(row.get("trash_reason") or ""),
            "manual_boost": round(_clamp(row.get("manual_boost", 0.0)), 2),
            "age_days": round(max(0.0, (now - created) / 86400.0), 1) if created else 0.0,
            "date": _date_of(created),
            "retention": round(retention(row, now, half_life_days=self.cfg.decay_half_life_days), 2),
        }

    def boost(self, fact_id: int, delta: float = 0.12) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM facts WHERE id=?", (int(fact_id),))
        if row is None:
            return None
        now = self.clock()
        current = _clamp(row["manual_boost"])
        new_boost = _clamp(current + max(-0.5, min(0.5, float(delta))))
        importance = _clamp(float(row["importance"]) + (0.03 if delta > 0 else -0.01))
        pool = str(row["pool"] or "cycle")
        if pool == "trash":
            pool = "cycle"
        elif pool == "shallow" and new_boost > 0.05:
            pool = "cycle"
        self.db.execute(
            "UPDATE facts SET manual_boost=?,importance=?,pool=?,weight=?,"
            " trashed_at=0,trash_reason='' WHERE id=?",
            (new_boost, importance, pool, min(1.0, 0.4 + new_boost), int(fact_id)),
        )
        updated = self.db.fetchone("SELECT * FROM facts WHERE id=?", (int(fact_id),))
        return self._public(dict(updated), now) if updated else None

    def trash(self, fact_id: int, reason: str = "manual") -> bool:
        row = self.db.fetchone("SELECT * FROM facts WHERE id=?", (int(fact_id),))
        if row is None:
            return False
        return self._trash_row(dict(row), reason, self.clock())

    def restore(self, fact_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE facts SET pool='cycle',trashed_at=0,trash_reason='' WHERE id=? AND pool='trash'",
            (int(fact_id),),
        )
        return bool(cur.rowcount)

    # ------------------------------------------------------------------ counts
    def count(self, user_key: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM facts WHERE user_key=? AND pool<>'trash'",
                (user_key,),
                0,
            )
        )

    def count_all(self, user_key: str | None = None) -> int:
        if user_key:
            return int(
                self.db.scalar("SELECT COUNT(*) FROM facts WHERE user_key=?", (user_key,), 0)
            )
        return int(self.db.scalar("SELECT COUNT(*) FROM facts", (), 0))

    def distinct_count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(DISTINCT text) FROM facts", (), 0))

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


def _date_of(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError, TypeError):
        return ""


def _color(weight: float) -> str:
    """Map 0..1 weight to a CSS colour used by the dot matrix."""
    stops = (
        (0.0, (203, 213, 225)),
        (0.25, (148, 197, 190)),
        (0.5, (45, 161, 148)),
        (0.75, (15, 118, 110)),
        (1.0, (6, 78, 74)),
    )
    value = _clamp(weight)
    for index in range(1, len(stops)):
        left, right = stops[index - 1], stops[index]
        if value <= right[0]:
            span = max(1e-6, right[0] - left[0])
            ratio = (value - left[0]) / span
            rgb = tuple(
                int(left[1][channel] + (right[1][channel] - left[1][channel]) * ratio)
                for channel in range(3)
            )
            return "#%02x%02x%02x" % rgb
    return "#064e4a"
