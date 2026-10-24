"""L0 clue layer: exactly ten hot slots, a stable, tiny injected prefix.

The injected block is a pure function of (members, seeds, header).  Brightness
changes alone never rewrite bytes.  Membership changes only when:

* the prefix is empty,
* a member seed was corrected/replaced,
* the rebalance interval has elapsed **and** a challenger beats the weakest
  incumbent by ``hysteresis`` **and** the dwell time has elapsed.

That is the main trick that keeps prompt-cache hit rates high while the
underlying salience keeps moving.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from ..config import L0Config
from ..db import Database
from ..embeddings import LocalScorer
from ..models import Ball
from ..normalize import content_tokens, is_ack

HEADER = "【长期线索·L0】"


class L0Layer:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        cfg: L0Config,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.cfg = cfg
        self.clock = clock

    # ------------------------------------------------------------------ slots
    def ensure_slots(self, user_key: str) -> None:
        now = self.clock()
        with self.db.transaction() as conn:
            for slot in range(1, self.cfg.slots + 1):
                conn.execute(
                    "INSERT OR IGNORE INTO hot_slots"
                    "(user_key,slot,content_seed,brightness,last_touched,type,created_at,updated_at) "
                    "VALUES(?,?,'',1.0,?,'fact',?,?)",
                    (user_key, slot, now, now, now),
                )

    def balls(self, user_key: str) -> list[Ball]:
        rows = self.db.fetchall(
            "SELECT slot,content_seed,brightness,last_touched,type FROM hot_slots "
            "WHERE user_key=? ORDER BY slot",
            (user_key,),
        )
        return [
            Ball(
                slot=int(r["slot"]),
                content_seed=r["content_seed"] or "",
                brightness=float(r["brightness"]),
                last_touched=float(r["last_touched"]),
                type=r["type"] or "fact",
            )
            for r in rows
        ]

    def active_balls(self, user_key: str) -> list[Ball]:
        return [b for b in self.balls(user_key) if b.content_seed.strip()]

    def update_ball(self, user_key: str, slot: int, **kwargs: Any) -> None:
        if not kwargs:
            return
        kwargs["updated_at"] = self.clock()
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [user_key, slot]
        self.db.execute(f"UPDATE hot_slots SET {cols} WHERE user_key=? AND slot=?", vals)

    # -------------------------------------------------------------- salience
    def salience(self, ball: Ball, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        age_days = max(0.0, now - ball.last_touched) / 86400.0
        recency = math.exp(-age_days / 7.0) * 0.5
        return ball.brightness + recency + self.cfg.type_bonus.get(ball.type, 0.0)

    def decay(self, user_key: str) -> None:
        self.ensure_slots(user_key)
        self.db.execute(
            "UPDATE hot_slots SET brightness=brightness*? WHERE user_key=? AND content_seed<>''",
            (self.cfg.decay, user_key),
        )
        self.db.execute(
            "UPDATE warm_slots SET brightness=brightness*? WHERE user_key=?",
            (self.cfg.warm_decay, user_key),
        )

    def touch_best(self, user_key: str, text: str, amount: float = 0.5) -> Ball | None:
        if len((text or "").strip()) <= 1 or is_ack(text):
            return None
        balls = self.active_balls(user_key)
        if not balls:
            return None
        best = max(balls, key=lambda b: self.scorer.similarity(text, b.content_seed))
        rel = self.scorer.similarity(text, best.content_seed)
        if rel < 0.18:
            return None
        now = self.clock()
        self.update_ball(
            user_key,
            best.slot,
            brightness=min(8.0, max(0.0, best.brightness + amount)),
            last_touched=now,
        )
        best.brightness = min(8.0, best.brightness + amount)
        best.last_touched = now
        return best

    def boost_recent(self, user_key: str, amount: float = 0.5) -> Ball | None:
        """Affirmation without content ("嗯/好") reinforces the last clue."""
        balls = self.active_balls(user_key)
        if not balls:
            return None
        ball = max(balls, key=lambda b: b.last_touched)
        now = self.clock()
        self.update_ball(
            user_key,
            ball.slot,
            brightness=min(8.0, ball.brightness + amount),
            last_touched=now,
        )
        ball.brightness = min(8.0, ball.brightness + amount)
        ball.last_touched = now
        return ball

    def best_match(self, user_key: str, text: str) -> tuple[Ball | None, float]:
        balls = self.active_balls(user_key)
        if not balls:
            return None, 0.0
        scored = [(b, self.scorer.similarity(text, b.content_seed)) for b in balls]
        return max(scored, key=lambda kv: kv[1])

    # ------------------------------------------------------------------ write
    def insert_candidate(
        self,
        user_key: str,
        seed: str,
        ball_type: str,
        *,
        brightness: float = 1.6,
        topic_key: str = "",
        source: str = "l2",
    ) -> dict[str, Any]:
        """Insert or reinforce a seed.  Returns a small decision record."""
        seed = seed.strip()[: self.cfg.max_seed_chars]
        if not seed:
            return {"action": "skip"}
        self.ensure_slots(user_key)
        now = self.clock()
        balls = self.balls(user_key)
        active = [b for b in balls if b.content_seed]
        if active:
            best = max(active, key=lambda b: self.scorer.similarity(seed, b.content_seed))
            if self.scorer.similarity(seed, best.content_seed) >= 0.65:
                self.update_ball(
                    user_key,
                    best.slot,
                    brightness=min(8.0, best.brightness + 0.5),
                    last_touched=now,
                )
                return {"action": "reinforce", "slot": best.slot, "old": best.content_seed}

        empties = [b for b in balls if not b.content_seed]
        old: Ball | None = None
        if empties:
            slot = empties[0].slot
        else:
            # Never evict an injected member for a new fact: that would rewrite
            # the stable prefix.  Prefer a non-injected hot ball; only fall back
            # to the weakest member if all ten are injected (k <= 3, so rare).
            members = set(self.db.get_state(user_key, "l0_members", []) or [])
            pool = [b for b in active if b.slot not in members] or active
            old = min(pool, key=lambda b: self.salience(b, now))
            # Fold the displaced node down one tier before overwriting it.
            self.add_warm(user_key, old.content_seed, old.brightness, old.type)
            slot = old.slot
        self.update_ball(
            user_key,
            slot,
            content_seed=seed,
            brightness=brightness,
            last_touched=now,
            type=ball_type,
        )
        return {"action": "replace" if old else "insert", "slot": slot, "old": old.content_seed if old else None}

    def correct_best(self, user_key: str, text: str, new_seed: str) -> dict[str, Any]:
        ball, rel = self.best_match(user_key, text)
        if ball is None or rel < 0.06:
            return self.insert_candidate(user_key, new_seed, "fact", brightness=1.0, source="correction")
        now = self.clock()
        old = ball.content_seed
        self.update_ball(
            user_key,
            ball.slot,
            content_seed=new_seed[: self.cfg.max_seed_chars],
            brightness=1.0,
            last_touched=now,
            type=ball.type,
        )
        # A correction is a material prefix event.
        self.db.set_state(user_key, "l0_force_rebalance", True)
        return {"action": "correct", "slot": ball.slot, "old": old}

    def correct_by_terms(self, user_key: str, old_terms: list[str], new_seed: str) -> dict[str, Any]:
        """Correct the ball that contains one of the terms the user rejected.

        This is still a local string match; it exists because "不是累，是烦"
        should update the *tired* clue even when the word "疲惫" is used in the
        stored seed.
        """
        terms = [t.strip() for t in old_terms if t and t.strip()]
        if not terms:
            return {"action": "skip"}
        for ball in self.active_balls(user_key):
            if any(term in ball.content_seed for term in terms):
                now = self.clock()
                old = ball.content_seed
                self.update_ball(
                    user_key,
                    ball.slot,
                    content_seed=new_seed[: self.cfg.max_seed_chars],
                    brightness=1.0,
                    last_touched=now,
                    type=ball.type,
                )
                self.db.set_state(user_key, "l0_force_rebalance", True)
                return {"action": "correct", "slot": ball.slot, "old": old}
        return {"action": "skip"}

    def add_warm(self, user_key: str, seed: str, brightness: float, ball_type: str = "fact") -> None:
        if not seed.strip():
            return
        now = self.clock()
        existing = self.db.fetchone(
            "SELECT id,brightness FROM warm_slots WHERE user_key=? AND content_seed=?",
            (user_key, seed),
        )
        if existing:
            self.db.execute(
                "UPDATE warm_slots SET brightness=?,last_touched=? WHERE id=?",
                (max(float(existing["brightness"]), brightness), now, existing["id"]),
            )
            return
        count = int(self.db.scalar("SELECT COUNT(*) FROM warm_slots WHERE user_key=?", (user_key,), 0))
        if count >= self.cfg_limit:
            self._evict_warm(user_key)
        self.db.execute(
            "INSERT OR IGNORE INTO warm_slots(user_key,content_seed,brightness,last_touched,type,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (user_key, seed[: self.cfg.max_seed_chars], brightness, now, ball_type, now),
        )

    @property
    def cfg_limit(self) -> int:
        # Warm limit lives in L2Config but L0 owns tier mechanics.  The engine
        # injects the effective value at construction (default 30).
        return getattr(self, "_warm_limit", 30)

    def set_warm_limit(self, limit: int) -> None:
        self._warm_limit = max(1, int(limit))

    def _evict_warm(self, user_key: str) -> None:
        row = self.db.fetchone(
            "SELECT id,content_seed,brightness,type FROM warm_slots WHERE user_key=? "
            "ORDER BY brightness ASC,last_touched ASC LIMIT 1",
            (user_key,),
        )
        if row is None:
            return
        self.archive_cold(
            user_key,
            row["content_seed"],
            thin_context=f"曾经活跃的线索：{row['content_seed']}",
            topic_key="",
            source="warm_evict",
        )
        self.db.execute("DELETE FROM warm_slots WHERE id=?", (row["id"],))

    def swap_warm(self, user_key: str) -> dict[str, Any] | None:
        hot = self.active_balls(user_key)
        if not hot:
            return None
        coldest = min(hot, key=lambda b: b.brightness)
        warm = self.db.fetchone(
            "SELECT id,content_seed,brightness,type FROM warm_slots WHERE user_key=? "
            "ORDER BY brightness DESC,last_touched DESC LIMIT 1",
            (user_key,),
        )
        if warm is None or float(warm["brightness"]) <= coldest.brightness + 0.35:
            return None
        now = self.clock()
        # Physically delete the warm row, then write it into the hot slot; the
        # displaced hot seed is folded into warm (which may in turn evict).
        self.db.execute("DELETE FROM warm_slots WHERE id=?", (warm["id"],))
        self.add_warm(user_key, coldest.content_seed, coldest.brightness, coldest.type)
        self.update_ball(
            user_key,
            coldest.slot,
            content_seed=warm["content_seed"],
            brightness=float(warm["brightness"]),
            last_touched=now,
            type=warm["type"],
        )
        self.db.set_state(user_key, "l0_force_rebalance", True)
        return {"slot": coldest.slot, "out": coldest.content_seed, "in": warm["content_seed"]}

    def archive_cold(
        self,
        user_key: str,
        seed: str,
        *,
        thin_context: str = "",
        topic_key: str = "",
        source: str = "l2",
        tokens: str | None = None,
    ) -> None:
        seed = seed.strip()
        if not seed:
            return
        tok = tokens if tokens is not None else " ".join(content_tokens(seed))
        thin = (thin_context or seed)[:100]
        existing = self.db.fetchone(
            "SELECT id,access_count FROM cold_seeds WHERE user_key=? AND seed_tokens=?",
            (user_key, tok),
        )
        now = self.clock()
        if existing:
            self.db.execute(
                "UPDATE cold_seeds SET thin_context=?,last_access=? WHERE id=?",
                (thin, now, existing["id"]),
            )
            return
        self.db.execute(
            "INSERT INTO cold_seeds(user_key,seed_tokens,thin_context,topic_key,source,created_at,last_access) "
            "VALUES(?,?,?,?,?,?,?)",
            (user_key, tok, thin, topic_key, source, now, now),
        )
        self.db.execute(
            "DELETE FROM cold_seeds WHERE user_key=? AND id NOT IN "
            "(SELECT id FROM cold_seeds WHERE user_key=? ORDER BY id DESC LIMIT ?)",
            (user_key, user_key, max(10, int(getattr(self, "_cold_limit", 800)))),
        )

    def set_cold_limit(self, limit: int) -> None:
        self._cold_limit = max(10, int(limit))

    # --------------------------------------------------------------- prefix
    def rebalance(self, user_key: str, round_no: int, force: bool = False) -> list[int]:
        if not self.cfg.enabled:
            return []
        self.ensure_slots(user_key)
        now = self.clock()
        balls = self.active_balls(user_key)
        if not balls:
            self._persist_members(user_key, [], "", round_no)
            return []
        k = max(1, min(self.cfg.inject, self.cfg.slots))
        members = list(self.db.get_state(user_key, "l0_members", []) or [])
        member_map = {b.slot: b for b in balls}
        current = [int(s) for s in members if int(s) in member_map]

        force = force or bool(self.db.get_state(user_key, "l0_force_rebalance", False))
        last_changed = int(self.db.get_state(user_key, "l0_changed_round", 0) or 0)
        due = (round_no - last_changed) >= self.cfg.min_dwell_rounds and (
            round_no % max(1, self.cfg.rebalance_interval) == 0
        )
        k_target = min(k, len(balls))
        # If the prefix is under-filled, add clues immediately: showing one
        # ball when three are available is worse than one early prefix change.
        if len(current) >= k_target and not force and not due:
            return current

        # Local discovery: salience sets the long-term ranking, while a small
        # relevance bonus lets the current message break ties at the rare
        # rebalance points.  It is deliberately not applied every turn.
        query = self.db.get_state(user_key, "last_inbound_text", "") or ""
        scores = {
            b.slot: self.salience(b, now)
            + (0.25 * self.scorer.similarity(query, b.content_seed) if query else 0.0)
            for b in balls
        }
        if not current or force:
            desired = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        else:
            selected = set(current[:k])
            # Fill spaces first.
            for slot, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                if len(selected) >= k:
                    break
                selected.add(slot)
            # Hysteresis: challenger must beat the weakest incumbent clearly.
            changed = True
            while changed:
                changed = False
                weak = min(selected, key=lambda s: scores.get(s, 0.0))
                for slot, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
                    if slot in selected:
                        continue
                    if score > scores.get(weak, 0.0) + self.cfg.hysteresis:
                        selected.remove(weak)
                        selected.add(slot)
                        changed = True
                        break
            desired = sorted(selected, key=lambda s: scores.get(s, 0.0), reverse=True)

        # Stable order in the prompt is by slot number, regardless of salience.
        desired = sorted(desired, key=lambda s: s)
        block = self._render_block(desired, member_map)
        old_block = self.db.get_state(user_key, "l0_block", "") or ""
        if block != old_block or desired != current:
            self._persist_members(user_key, desired, block, round_no)
            return desired
        self.db.set_state(user_key, "l0_force_rebalance", False)
        return desired

    def _persist_members(self, user_key: str, members: list[int], block: str, round_no: int) -> None:
        self.db.set_state(user_key, "l0_members", members)
        self.db.set_state(user_key, "l0_block", block)
        self.db.set_state(user_key, "l0_changed_round", round_no)
        self.db.set_state(user_key, "l0_force_rebalance", False)

    def _render_block(self, slots: list[int], member_map: dict[int, Ball]) -> str:
        lines: list[str] = []
        for slot in sorted(slots):
            ball = member_map.get(slot)
            if ball is None or not ball.content_seed:
                continue
            lines.append(f"- {ball.content_seed[: self.cfg.max_seed_chars]}")
        if not lines:
            return ""
        block = HEADER + "\n" + "\n".join(lines)
        if len(block) > self.cfg.max_inject_chars:
            while lines and len(HEADER) + 1 + len("\n".join(lines)) > self.cfg.max_inject_chars:
                lines.pop()
            block = HEADER + "\n" + "\n".join(lines)
        return block

    def current_block(self, user_key: str, round_no: int) -> tuple[str, list[int]]:
        if not self.cfg.enabled:
            return "", []
        force = bool(self.db.get_state(user_key, "l0_force_rebalance", False))
        members = self.rebalance(user_key, round_no, force=force)
        block = self.db.get_state(user_key, "l0_block", "") or ""
        # A changed member seed must be reflected even if membership is equal.
        member_map = {b.slot: b for b in self.active_balls(user_key)}
        expected = self._render_block(members, member_map)
        if expected != block:
            block = expected
            self._persist_members(user_key, members, block, round_no)
        return block, members

    def injected_relevance(self, user_key: str, text: str, slots: list[int] | None = None) -> tuple[float, list[tuple[int, float]]]:
        balls = {b.slot: b for b in self.active_balls(user_key)}
        if slots is None:
            slots = [int(s) for s in (self.db.get_state(user_key, "l0_members", []) or [])]
        ranked = []
        for slot in slots:
            ball = balls.get(slot)
            if ball:
                ranked.append((slot, self.scorer.similarity(text, ball.content_seed)))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return (ranked[0][1] if ranked else 0.0), ranked
