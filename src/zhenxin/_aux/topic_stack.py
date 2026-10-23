"""A tiny topic stack (depth 3-5).  Only the top label enters the context."""

from __future__ import annotations

import hashlib

from ..db import Database
from ..embeddings import LocalScorer, keyword_set
from ..normalize import compact, is_ack, is_cold


class TopicStack:
    def __init__(self, db: Database, scorer: LocalScorer, max_depth: int = 5, enabled: bool = True) -> None:
        self.db = db
        self.scorer = scorer
        self.max_depth = max(3, min(5, int(max_depth)))
        self.enabled = enabled

    def rows(self, user_key: str) -> list[dict]:
        return [
            {"depth": int(r["depth"]), "key": r["topic_key"], "label": r["label"]}
            for r in self.db.fetchall(
                "SELECT depth,topic_key,label FROM topic_stack WHERE user_key=? ORDER BY depth",
                (user_key,),
            )
        ]

    def top(self, user_key: str) -> dict | None:
        rows = self.rows(user_key)
        return rows[-1] if rows else None

    def observe(self, user_key: str, text: str) -> dict | None:
        if not self.enabled:
            return None
        text = (text or "").strip()
        if not text or is_ack(text) or is_cold(text) or len(compact(text)) < 2:
            return self.top(user_key)
        rows = self.rows(user_key)
        top = rows[-1] if rows else None
        if top:
            sim = self.scorer.similarity(text, top["label"])
            if sim >= 0.34:
                self.db.execute(
                    "UPDATE topic_stack SET pushed_at=strftime('%s','now') WHERE user_key=? AND depth=?",
                    (user_key, top["depth"]),
                )
                return top
        # Return to an older topic: pop everything above it.
        for row in reversed(rows[:-1] if top else []):
            if self.scorer.similarity(text, row["label"]) >= 0.46:
                self.db.execute(
                    "DELETE FROM topic_stack WHERE user_key=? AND depth>?",
                    (user_key, row["depth"]),
                )
                return row
        label = self._label(text)
        depth = (top["depth"] + 1) if top else 1
        self.db.execute(
            "INSERT OR REPLACE INTO topic_stack(user_key,depth,topic_key,label,pushed_at) "
            "VALUES(?,?,?,?,strftime('%s','now'))",
            (user_key, depth, self._key(label), label),
        )
        if depth > self.max_depth:
            # Drop the oldest (lowest depth) and shift down; stack stays <= 5.
            self.db.execute(
                "DELETE FROM topic_stack WHERE user_key=? AND depth IN "
                "(SELECT depth FROM topic_stack WHERE user_key=? ORDER BY depth ASC LIMIT 1)",
                (user_key, user_key),
            )
            rows = self.rows(user_key)
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM topic_stack WHERE user_key=?", (user_key,))
                for i, row in enumerate(rows, start=1):
                    conn.execute(
                        "INSERT INTO topic_stack(user_key,depth,topic_key,label,pushed_at) "
                        "VALUES(?,?,?,?,strftime('%s','now'))",
                        (user_key, i, row["key"], row["label"]),
                    )
            depth = min(depth, self.max_depth)
        return {"depth": depth, "key": self._key(label), "label": label}

    def hint(self, user_key: str) -> str:
        if not self.enabled:
            return ""
        top = self.top(user_key)
        if not top:
            return ""
        return f"当前话题：{top['label'][:12]}"

    @staticmethod
    def _label(text: str) -> str:
        words = keyword_set(text, 3)
        if words:
            return "·".join(words[:2])[:20]
        return compact(text)[:12]

    @staticmethod
    def _key(label: str) -> str:
        return hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
