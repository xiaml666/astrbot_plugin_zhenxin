"""Unclosed-things queue: max 5, time/emotion pointers pop it, age-out to cold."""

from __future__ import annotations

import re
from typing import Callable

from ..db import Database
from ..embeddings import LocalScorer
from ..normalize import COMPLETION_RE, FUTURE_ACTION_RE, TIME_RE, content_tokens, is_question

_ACTION_RE = re.compile(
    r"(交稿|截止|面试|考试|看医生|复诊|搬家|出差|开会|还钱|修理|旅行|生日|约了|赶稿|通宵|早起|"
    r"提交|汇报|答辩|体检|接人|送人|买|订)"
)
_SUMMARY_SPLIT_RE = re.compile(r"[，,。；;！!？?\n]")


class OpenLoops:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        limit: int = 5,
        ttl_rounds: int = 25,
        enabled: bool = True,
        archive: Callable[..., None] | None = None,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.limit = max(1, int(limit))
        self.ttl_rounds = max(1, int(ttl_rounds))
        self.enabled = enabled
        self.archive = archive

    def rows(self, user_key: str) -> list[dict]:
        return [
            {
                "id": int(r["id"]),
                "summary": r["summary"],
                "keywords": r["keywords"],
                "rounds_alive": int(r["rounds_alive"]),
                "due_hint": r["due_hint"] or "",
            }
            for r in self.db.fetchall(
                "SELECT id,summary,keywords,rounds_alive,due_hint FROM open_loops "
                "WHERE user_key=? AND status='open' ORDER BY id ASC",
                (user_key,),
            )
        ]

    def observe(self, user_key: str, text: str, round_no: int) -> dict:
        if not self.enabled:
            return {"hint": "", "touched": [], "created": None, "closed": []}
        closed = self._close_completed(user_key, text)
        self._age(user_key, round_no)
        touched = self._touch(user_key, text)
        created = self._maybe_create(user_key, text, round_no)
        hint = self.hint(user_key)
        return {"hint": hint, "touched": touched, "created": created, "closed": closed}

    def _close_completed(self, user_key: str, text: str) -> list[str]:
        if not COMPLETION_RE.search(text):
            return []
        out: list[str] = []
        for row in self.rows(user_key):
            if self.scorer.similarity(text, row["keywords"] or row["summary"]) >= 0.16:
                out.append(row["summary"])
                # Completed tasks are removed, not moved to cold: recalling
                # them later as "unfinished" would be worse than forgetting.
                self.db.execute("DELETE FROM open_loops WHERE id=?", (row["id"],))
        return out

    def _age(self, user_key: str, round_no: int) -> None:
        self.db.execute(
            "UPDATE open_loops SET rounds_alive=rounds_alive+1 WHERE user_key=? AND status='open'",
            (user_key,),
        )
        expired = self.db.fetchall(
            "SELECT id,summary FROM open_loops WHERE user_key=? AND status='open' AND rounds_alive>?",
            (user_key, self.ttl_rounds),
        )
        for row in expired:
            if self.archive:
                self.archive(
                    user_key,
                    row["summary"],
                    thin_context=f"尚未了结：{row['summary']}",
                    source="loop_expired",
                )
            self.db.execute("DELETE FROM open_loops WHERE id=?", (row["id"],))

    def _touch(self, user_key: str, text: str) -> list[str]:
        out: list[str] = []
        for row in self.rows(user_key):
            sim = self.scorer.similarity(text, row["keywords"] or row["summary"])
            time_match = bool(TIME_RE.search(text) and row["due_hint"] and row["due_hint"] in text)
            if sim >= 0.2 or time_match:
                out.append(row["summary"])
                self.db.execute(
                    "UPDATE open_loops SET updated_at=strftime('%s','now') WHERE id=?",
                    (row["id"],),
                )
        return out

    def _maybe_create(self, user_key: str, text: str, round_no: int) -> str | None:
        if is_question(text) and not _ACTION_RE.search(text):
            return None
        if not (TIME_RE.search(text) or FUTURE_ACTION_RE.search(text)):
            return None
        action = _ACTION_RE.search(text)
        if not action:
            return None
        clauses = [c.strip() for c in _SUMMARY_SPLIT_RE.split(text) if c.strip()]
        relevant = next((c for c in clauses if action.group(1) in c), action.group(1))
        due = TIME_RE.search(text)
        summary = ((due.group(1) + "·") if due else "") + relevant
        summary = summary[:40]
        keywords = " ".join(content_tokens(summary)[:6]) or summary
        # Deduplicate by lexical similarity.
        for row in self.rows(user_key):
            if self.scorer.similarity(summary, row["summary"]) >= 0.55:
                return None
        count = int(self.db.scalar("SELECT COUNT(*) FROM open_loops WHERE user_key=? AND status='open'", (user_key,), 0))
        if count >= self.limit:
            oldest = self.rows(user_key)[0]
            if self.archive:
                self.archive(
                    user_key,
                    oldest["summary"],
                    thin_context=f"尚未了结：{oldest['summary']}",
                    source="loop_overflow",
                )
            self.db.execute("DELETE FROM open_loops WHERE id=?", (oldest["id"],))
        self.db.execute(
            "INSERT INTO open_loops(user_key,summary,keywords,created_at,updated_at,rounds_alive,due_hint,status) "
            "VALUES(?,?,?,strftime('%s','now'),strftime('%s','now'),0,?,'open')",
            (user_key, summary, keywords, due.group(1) if due else ""),
        )
        return summary

    def hint(self, user_key: str) -> str:
        rows = self.rows(user_key)
        if not rows:
            return ""
        bits = [f"{r['summary']}（第{r['rounds_alive']}轮）" for r in rows[:2]]
        return "未闭合：" + "；".join(bits)

    def count(self, user_key: str) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM open_loops WHERE user_key=? AND status='open'", (user_key,), 0))
