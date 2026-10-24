"""Optional proactive messages ("主动找你说话").

This is deliberately opt-in and conservative: an unsolicited message is a
product decision.  The scheduler only considers conversations that were active
inside ``active_window_hours``, respects quiet hours, a per-conversation idle
threshold, a minimum interval and a daily cap.  Every send is persisted, so a
restart cannot reset the cap and spam the user.

Message material comes from the plugin's own memory: an unfinished open loop
first ("稿子交了吗"), then a high-value fact, then a time-of-day greeting bank
("吃饭了没" / "在干嘛呀" / "还没睡？").  Every third message skips the memory
material on purpose, so the stream does not read like a reminder bot.  An
optional LLM call can rewrite the line in the user's style; the local template
is always the fallback.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Callable

from .config import ProactiveConfig
from .db import Database

STATE_LAST_USER = "last_user_at"

# Casual check-ins, grouped by time of day.  These are the "你在干嘛呀" /
# "吃饭了没" half of proactive messages: small talk is a feature, not noise.
GREETINGS: dict[str, tuple[str, ...]] = {
    "dawn": (
        "醒了？早安。",
        "早呀，今天起得挺早。",
        "早上好，今天有什么安排？",
        "刚醒，早。",
    ),
    "morning": (
        "在忙吗？",
        "今天忙不忙？",
        "上午还顺利吗？",
        "在干嘛呀？",
    ),
    "noon": (
        "吃饭了没？",
        "到饭点了，今天吃什么？",
        "午饭吃了吗？别又随便对付。",
        "中午好，吃了没？",
    ),
    "afternoon": (
        "在干嘛呀？",
        "下午犯困不？",
        "忙完了吗？",
        "刚泡了杯茶，想起你了。",
    ),
    "evening": (
        "下班了吗？",
        "晚饭吃了吗？",
        "今天累不累？",
        "忙一天了，歇会儿吧。",
    ),
    "night": (
        "晚上好呀，今天怎么样？",
        "吃了吗？在忙啥？",
        "在做什么呢？",
        "今天过得怎么样？",
    ),
    "late": (
        "还没睡？",
        "这么晚还不睡呀。",
        "睡不着？",
        "别熬太晚了。",
    ),
}

GENERIC_GREETINGS: tuple[str, ...] = (
    "突然想起你了。",
    "在干嘛呀？",
    "最近怎么样？",
    "好久没聊，你还好吗？",
    "刚看到个东西，想到你了。",
)


def _band(hour: int) -> str:
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 11:
        return "morning"
    if 11 <= hour < 13:
        return "noon"
    if 13 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 19:
        return "evening"
    if 19 <= hour < 23:
        return "night"
    return "late"


def is_group_session(user_key: str) -> bool:
    lowered = (user_key or "").lower()
    return "group" in lowered or "群" in lowered


def _hour(now: float) -> int:
    try:
        return datetime.fromtimestamp(now).hour
    except (OSError, OverflowError, ValueError):
        return 12


def _in_quiet(hour: int, start: int, end: int) -> bool:
    start = int(start) % 24
    end = int(end) % 24
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


class ProactiveScheduler:
    def __init__(
        self,
        db: Database,
        cfg: ProactiveConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.clock = clock

    # --------------------------------------------------------------- sessions
    def candidates(self, now: float | None = None, limit: int = 50) -> list[dict[str, Any]]:
        now = self.clock() if now is None else now
        rows = self.db.fetchall(
            "SELECT user_key, MAX(ts) last FROM ("
            " SELECT user_key,created_at ts FROM metrics"
            " UNION ALL SELECT user_key,created_at FROM turns WHERE role='user'"
            ") GROUP BY user_key ORDER BY last DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        out: list[dict[str, Any]] = []
        window = max(0.5, float(self.cfg.active_window_hours)) * 3600.0
        for row in rows:
            key = str(row["user_key"])
            last = max(float(row["last"] or 0), float(self.db.get_state(key, STATE_LAST_USER, 0) or 0))
            if not last or now - last > window:
                continue
            out.append({"user": key, "last_user_at": last})
        return out

    # ------------------------------------------------------------------ gates
    def _last_proactive(self, user_key: str) -> float:
        return float(
            self.db.scalar(
                "SELECT MAX(created_at) FROM proactive_log WHERE user_key=?",
                (user_key,),
                0,
            )
            or 0
        )

    def _today_count(self, user_key: str, now: float) -> int:
        start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM proactive_log WHERE user_key=? AND created_at>=?",
                (user_key, start.timestamp()),
                0,
            )
            or 0
        )

    def gate(self, user_key: str, last_user_at: float, now: float | None = None) -> tuple[bool, str]:
        now = self.clock() if now is None else now
        if not self.cfg.enabled:
            return False, "未开启"
        if self.cfg.only_private and is_group_session(user_key):
            return False, "群聊不发"
        if _in_quiet(_hour(now), self.cfg.quiet_start, self.cfg.quiet_end):
            return False, "静默时段"
        idle = max(0.0, now - float(last_user_at or 0))
        if idle < max(1, int(self.cfg.min_idle_minutes)) * 60:
            return False, "对方刚说过话"
        last = self._last_proactive(user_key)
        if last and now - last < max(0.1, float(self.cfg.min_interval_hours)) * 3600:
            return False, "离上次太近"
        if self._today_count(user_key, now) >= max(0, int(self.cfg.max_per_day)):
            return False, "今天已经够了"
        return True, ""

    # --------------------------------------------------------------- compose
    def _open_loop(self, user_key: str) -> dict[str, Any] | None:
        return self.db.fetchone(
            "SELECT summary,updated_at FROM open_loops WHERE user_key=? AND status='open'"
            " ORDER BY updated_at DESC LIMIT 1",
            (user_key,),
        )

    def _top_fact(self, user_key: str) -> str:
        rows = self.db.fetchall(
            "SELECT text,importance,mentions FROM facts WHERE user_key=?"
            " AND pool IN ('treasure','cycle') AND kind NOT IN ('emotion','address')"
            " ORDER BY importance DESC,mentions DESC LIMIT 3",
            (user_key,),
        )
        for row in rows:
            text = str(row["text"] or "").strip()
            if 2 <= len(text) <= 40 and "更正" not in text:
                return text
        return ""

    def _sent_count(self, user_key: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM proactive_log WHERE user_key=?",
                (user_key,),
                0,
            )
            or 0
        )

    def _last_text(self, user_key: str) -> str:
        row = self.db.fetchone(
            "SELECT content FROM proactive_log WHERE user_key=?"
            " ORDER BY id DESC LIMIT 1",
            (user_key,),
        )
        return str(row["content"] or "") if row else ""

    def _greeting(self, user_key: str, now: float) -> str:
        """Pick one casual check-in that fits the hour and never repeats."""
        pool = GREETINGS[_band(_hour(now))] + GENERIC_GREETINGS
        # Deterministic per user/day/send-count: a restart does not replay the
        # same line, and tests stay reproducible without touching ``random``.
        seed = f"{user_key}|{int(now // 86400)}|{_hour(now)}|{self._sent_count(user_key)}"
        index = sum(ord(ch) for ch in seed) % len(pool)
        text = pool[index]
        if text == self._last_text(user_key):
            text = pool[(index + 1) % len(pool)]
        return text

    def compose(self, user_key: str, now: float | None = None) -> tuple[str, str]:
        """Return ``(text, kind)`` using local memory only.

        Material order: unfinished open loop, high-value fact, then a casual
        greeting for the current time of day.  Every third message skips the
        memory material on purpose, so the stream also contains plain small
        talk instead of only reminders.
        """
        now = self.clock() if now is None else now
        template = str(self.cfg.template or "").strip()
        greeting = self._greeting(user_key, now)
        casual_turn = self._sent_count(user_key) % 3 == 2

        loop = None if casual_turn else self._open_loop(user_key)
        if loop is not None:
            summary = str(loop["summary"] or "").strip()
            # Clean rows written by older versions ("周五·周五要交稿").
            summary = re.sub(r"^([^·]{1,6})·\1", r"\1", summary)
            text = template or "对了，{loop}后来怎么样了？"
            text = text.replace("{loop}", summary or "上次说的那件事")
            text = text.replace("{greeting}", greeting)
            return self._clip(text), "open_loop"
        fact = "" if casual_turn else self._top_fact(user_key)
        if fact:
            text = template or "突然想到，{fact}。"
            text = text.replace("{fact}", fact).replace("{greeting}", greeting)
            return self._clip(text), "memory"
        text = template or "{greeting}"
        text = text.replace("{greeting}", greeting)
        # A custom template may still hold {loop}/{fact}; drop the empty slots
        # rather than sending a literal placeholder to the user.
        text = text.replace("{loop}", "").replace("{fact}", "")
        text = " ".join(text.strip(" ，,。；;：:").split())
        return self._clip(text or greeting), "presence"

    def _clip(self, text: str) -> str:
        text = " ".join(str(text or "").split())
        limit = max(10, int(self.cfg.max_chars))
        return text[:limit]

    def mark_sent(self, user_key: str, text: str, kind: str = "memory", now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self.db.execute(
            "INSERT INTO proactive_log(user_key,content,kind,created_at) VALUES(?,?,?,?)",
            (user_key, text, kind, now),
        )
        self.db.execute(
            "DELETE FROM proactive_log WHERE user_key=? AND id NOT IN"
            " (SELECT id FROM proactive_log WHERE user_key=? ORDER BY id DESC LIMIT 200)",
            (user_key, user_key),
        )
        self.db.set_state(user_key, "proactive_last_at", now)

    def stats(self, user_key: str | None = None, limit: int = 20) -> dict[str, Any]:
        where = "WHERE user_key=?" if user_key else ""
        params: tuple[Any, ...] = (user_key,) if user_key else ()
        rows = self.db.fetchall(
            f"SELECT user_key,content,kind,created_at FROM proactive_log {where}"
            " ORDER BY created_at DESC LIMIT ?",
            params + (max(1, int(limit)),),
        )
        count = int(
            self.db.scalar(
                f"SELECT COUNT(*) FROM proactive_log {where}", params, 0
            )
            or 0
        )
        return {"total": count, "recent": [dict(row) for row in rows]}

    def preview(self, user_key: str) -> dict[str, Any]:
        text, kind = self.compose(user_key)
        return {"user": user_key, "text": text, "kind": kind}
