"""Negative-pattern blacklist: learn which reply shapes to avoid."""

from __future__ import annotations

import re

from ..db import Database
from ..normalize import is_cold

_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\ufe0f]+")
_QUESTION_RE = re.compile(r"[?？]|吗|呢|么")

_META_NEGATIVE = (
    ("template", re.compile(r"模板|机械|复读|机器人|AI味|别老说|又是这句")),
    ("too_long", re.compile(r"太长|说太多|别写小作文")),
    ("too_many_questions", re.compile(r"别问了|问太多|查户口")),
)


class NegativePatterns:
    def __init__(self, db: Database, enabled: bool = True) -> None:
        self.db = db
        self.enabled = enabled

    def record(self, user_key: str, text: str, last_reply: str, now: float) -> None:
        if not self.enabled:
            return
        features: list[str] = []
        if is_cold(text) or any(w in text for w in ("哦", "随便", "算了", "不聊了", "行吧")):
            features.extend(self._features(last_reply))
        for name, pattern in _META_NEGATIVE:
            if pattern.search(text):
                features.append(name)
        seen: set[str] = set()
        for feature in features:
            if feature in seen:
                continue
            seen.add(feature)
            self.db.execute(
                "INSERT INTO negative_patterns(user_key,feature,count,last_seen) VALUES(?,?,1,?) "
                "ON CONFLICT(user_key,feature) DO UPDATE SET count=count+1,last_seen=excluded.last_seen",
                (user_key, feature, now),
            )

    @staticmethod
    def _features(reply: str) -> list[str]:
        out: list[str] = []
        if len(reply) > 80:
            out.append("too_long")
        if len(_QUESTION_RE.findall(reply)) >= 2:
            out.append("too_many_questions")
        if len(_EMOJI_RE.findall(reply)) >= 2:
            out.append("too_many_emoji")
        return out or ["neutral_cold"]

    def hint(self, user_key: str, limit: int = 2) -> str:
        if not self.enabled:
            return ""
        rows = self.db.fetchall(
            "SELECT feature,count FROM negative_patterns WHERE user_key=? AND count>0 "
            "ORDER BY count DESC,last_seen DESC LIMIT ?",
            (user_key, limit),
        )
        if not rows:
            return ""
        labels = {
            "too_long": "回复别太长",
            "too_many_questions": "别连续追问",
            "too_many_emoji": "少用表情",
            "template": "别像模板/复读",
            "neutral_cold": "别显得冷淡",
        }
        return "避开：" + "、".join(labels.get(r["feature"], r["feature"]) for r in rows)
