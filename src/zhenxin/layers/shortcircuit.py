"""Habit solidification (the short-circuit).

``(user, signature) -> response_shape``.  Shape parameters only; the original
user text is never stored.  Confidence and negative streaks are updated by
counts and regexes, never by a model.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import ShortCircuitConfig
from ..db import Database
from ..embeddings import LocalScorer
from ..models import SolidPath
from ..normalize import (
    TEMPLATES,
    Signature,
    compact,
    is_ack,
    is_cold,
    is_correction,
    is_negation,
    is_question,
    pattern_signature,
    response_shape,
)


_META_NEG_RE = re.compile(r"模板|别老说|机械|复读|机器人|AI味|太假|敷衍", re.I)


def is_meta_negative(text: str) -> bool:
    return bool(_META_NEG_RE.search(text or ""))


@dataclass
class Match:
    path: SolidPath
    signature: Signature
    created: bool
    repeated: bool
    hit: bool
    template: str | None = None


class ShortCircuit:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        cfg: ShortCircuitConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.cfg = cfg
        self.clock = clock

    # ------------------------------------------------------------------- rows
    def get_path(self, user_key: str, signature: str) -> SolidPath | None:
        row = self.db.fetchone(
            "SELECT * FROM solidified_paths WHERE user_key=? AND signature=?",
            (user_key, signature),
        )
        if row is None:
            return None
        try:
            shape = json.loads(row["response_shape"])
        except (TypeError, json.JSONDecodeError):
            shape = {}
        return SolidPath(
            signature=row["signature"],
            shape=shape,
            confidence=float(row["confidence"]),
            last_hit=float(row["last_hit"]) if row["last_hit"] else None,
            last_seen_round=int(row["last_seen_round"]),
            neg_streak=int(row["neg_streak"]),
            degraded_until=float(row["degraded_until"]),
            hits=int(row["hits"]),
            misses=int(row["misses"]),
        )

    def _save(self, user_key: str, path: SolidPath) -> None:
        self.db.execute(
            "INSERT INTO solidified_paths"
            "(user_key,signature,response_shape,confidence,last_hit,last_seen_round,neg_streak,degraded_until,hits,misses) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(user_key,signature) DO UPDATE SET "
            "response_shape=excluded.response_shape,confidence=excluded.confidence,last_hit=excluded.last_hit,"
            "last_seen_round=excluded.last_seen_round,neg_streak=excluded.neg_streak,"
            "degraded_until=excluded.degraded_until,hits=excluded.hits,misses=excluded.misses",
            (
                user_key,
                path.signature,
                json.dumps(path.shape, ensure_ascii=False, separators=(",", ":")),
                path.confidence,
                path.last_hit,
                path.last_seen_round,
                path.neg_streak,
                path.degraded_until,
                path.hits,
                path.misses,
            ),
        )

    # --------------------------------------------------------------- feedback
    def feedback(
        self,
        user_key: str,
        text: str,
        previous_signature: str | None,
        previous_user_text: str,
        previous_was_solid: bool,
        round_no: int,
    ) -> dict[str, Any]:
        if not previous_signature:
            return {"applied": False}
        path = self.get_path(user_key, previous_signature)
        if path is None:
            return {"applied": False}
        now = self.clock()
        # Cooldown expiry: allow the path to be learned again.
        if path.degraded_until and now >= path.degraded_until:
            path.neg_streak = 0
            path.degraded_until = 0.0
        kind = self._classify(text, previous_user_text, previous_was_solid)
        if kind == "ack":
            path.confidence = min(10.0, path.confidence + self.cfg.ack_boost)
            path.neg_streak = max(0, path.neg_streak - 1)
        elif kind == "continue":
            path.confidence = min(10.0, path.confidence + self.cfg.continue_boost)
            path.neg_streak = max(0, path.neg_streak - 1)
        elif kind == "negative":
            path.confidence = max(0.0, path.confidence - self.cfg.negative_penalty)
            path.neg_streak += 1
            if path.neg_streak >= self.cfg.neg_degrade_threshold:
                was_degraded = path.degraded_until > now
                path.degraded_until = max(path.degraded_until, now + self.cfg.degrade_hours * 3600.0)
                if not was_degraded:
                    self.db.incr_state(user_key, "solid_degrade_events", 1)
        path.last_seen_round = round_no
        self._save(user_key, path)
        return {
            "applied": True,
            "kind": kind,
            "signature": previous_signature,
            "confidence": path.confidence,
            "neg_streak": path.neg_streak,
            "degraded_until": path.degraded_until,
        }

    def _classify(self, text: str, previous_user_text: str, previous_was_solid: bool) -> str:
        if is_correction(text):
            return "negative"
        if is_cold(text):
            return "negative"
        if is_ack(text):
            return "ack"
        if is_question(text):
            return "neutral"
        if is_negation(text) or is_meta_negative(text):
            return "negative"
        if previous_user_text:
            sig_prev = pattern_signature(previous_user_text, self.cfg.max_literal_chars).key
            sig_now = pattern_signature(text, self.cfg.max_literal_chars).key
            if sig_prev == sig_now:
                return "continue"
            if self.scorer.similarity(text, previous_user_text) >= 0.22:
                return "continue"
        # A topic switch after a shaped reply is a negative signal.  After a
        # full reply it is ordinary conversation and is left neutral.
        if previous_was_solid:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------ match
    def match(
        self,
        user_key: str,
        text: str,
        round_no: int,
        *,
        last_path_signature: str | None = None,
    ) -> Match:
        sig = pattern_signature(text, self.cfg.max_literal_chars)
        path = self.get_path(user_key, sig.key)
        created = False
        repeated = False
        now = self.clock()
        if path is None:
            shape = response_shape(sig.intent, sig.key)
            path = SolidPath(
                signature=sig.key,
                shape=shape,
                confidence=self.cfg.initial_confidence,
                last_hit=None,
                last_seen_round=round_no,
                neg_streak=0,
                degraded_until=0.0,
                hits=0,
                misses=0,
            )
            created = True
        else:
            repeated = True
            expired = bool(path.degraded_until and now >= path.degraded_until)
            if expired:
                path.neg_streak = 0
                path.degraded_until = 0.0
            # Do not double-count the same turn: if the previous turn used this
            # exact path, its feedback was already applied above.  Feedback for
            # a *different* path must not suppress this path's repeat boost.
            if last_path_signature != sig.key and not path.degraded_until:
                path.confidence = min(10.0, path.confidence + self.cfg.repeat_boost)
            path.last_seen_round = round_no
        hit = self._can_hit(path, text, now)
        if hit:
            path.hits += 1
            path.last_hit = now
        else:
            path.misses += 1
        self._save(user_key, path)
        return Match(path=path, signature=sig, created=created, repeated=repeated, hit=hit)

    def _can_hit(self, path: SolidPath, text: str, now: float) -> bool:
        if not self.cfg.enabled:
            return False
        if path.neg_streak > 0 or path.degraded_until > now:
            return False
        if path.confidence < self.cfg.hit_threshold:
            return False
        if is_question(text) or is_correction(text):
            return False
        # Shaped generation is for short habits, not planning statements.
        if len(compact(text)) > self.cfg.max_hit_chars:
            return False
        return True

    # ----------------------------------------------------------------- render
    def render(self, user_key: str, match: Match, small_generator: Callable[[dict[str, Any]], str] | None = None) -> str:
        path = match.path
        shape = path.shape or response_shape(match.signature.intent, path.signature)
        intent = str(shape.get("intent") or match.signature.intent)
        pool = TEMPLATES.get(intent) or TEMPLATES["generic"]
        digest = hashlib.sha1(f"{user_key}|{path.signature}".encode("utf-8")).hexdigest()
        index = (int(digest[:8], 16) + path.hits) % len(pool)
        reply = pool[index]

        if (
            self.cfg.use_small_model
            and small_generator is not None
            and path.confidence < self.cfg.template_confidence
        ):
            try:
                generated = small_generator(
                    {
                        "intent": intent,
                        "shape": shape,
                        "template": reply,
                        "signature": path.signature,
                    }
                )
                if generated and generated.strip():
                    reply = generated.strip()
            except Exception:
                pass

        if shape.get("ask_followup") and "？" not in reply and intent in ("sad", "meal"):
            reply = reply.rstrip("。") + "？"
        if shape.get("use_emoji") and not any(ord(ch) > 0x1F000 for ch in reply):
            reply += "🙂"
        return reply[: self.cfg.max_reply_chars]

    def stats(self, user_key: str) -> dict[str, Any]:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS paths, SUM(hits) AS hits, SUM(misses) AS misses, "
            "SUM(CASE WHEN neg_streak>=? THEN 1 ELSE 0 END) AS degraded "
            "FROM solidified_paths WHERE user_key=?",
            (self.cfg.neg_degrade_threshold, user_key),
        )
        if row is None:
            return {"paths": 0, "hits": 0, "misses": 0, "degraded": 0}
        return {
            "paths": int(row["paths"] or 0),
            "hits": int(row["hits"] or 0),
            "misses": int(row["misses"] or 0),
            "degraded": int(row["degraded"] or 0),
            "degrade_events": int(self.db.get_state(user_key, "solid_degrade_events", 0) or 0),
        }
