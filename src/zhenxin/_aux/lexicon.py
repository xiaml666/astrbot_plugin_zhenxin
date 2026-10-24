"""User vocabulary mapping (user word <-> standard word), updated by rules."""

from __future__ import annotations

import re

from ..db import Database

_PAREN_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{1,10})[（(]([\u4e00-\u9fffA-Za-z0-9]{1,10})[）)]")
# Only metalinguistic naming, not ordinary corrections ("不是累，是烦").
_EXPLICIT_RE = re.compile(r"(?:我)?(?:不叫|管它叫|管这个叫)(.{1,8})[，, ]*(?:叫|是)(.{1,8})")


class Lexicon:
    def __init__(self, db: Database, aliases: dict[str, str] | None = None, enabled: bool = True) -> None:
        self.db = db
        self.enabled = enabled
        self.aliases = dict(aliases or {})

    def mappings(self, user_key: str, limit: int = 8) -> list[tuple[str, str]]:
        if not self.enabled:
            return []
        rows = self.db.fetchall(
            "SELECT user_word,standard_word,count FROM lexicon WHERE user_key=? "
            "ORDER BY count DESC,updated_at DESC LIMIT ?",
            (user_key, limit),
        )
        out = [(r["user_word"], r["standard_word"]) for r in rows]
        for user, standard in self.aliases.items():
            if (user, standard) not in out:
                out.append((user, standard))
        return out[:limit]

    def learn(self, user_key: str, text: str, now: float) -> list[tuple[str, str]]:
        if not self.enabled:
            return []
        found: list[tuple[str, str]] = []
        # "不是X，是Y" / "不叫X，叫Y" -> X is the user's word, Y standard.
        for m in _EXPLICIT_RE.finditer(text):
            user_word, standard = m.group(1).strip(), m.group(2).strip()
            if user_word and standard and user_word != standard:
                found.append((user_word, standard))
        # "X（Y）" -> X user word, Y standard.
        for m in _PAREN_RE.finditer(text):
            user_word, standard = m.group(1).strip(), m.group(2).strip()
            if user_word and standard and user_word != standard:
                found.append((user_word, standard))
        for user_word, standard in found:
            if not (1 <= len(user_word) <= 12 and 1 <= len(standard) <= 12):
                continue
            self.db.execute(
                "INSERT INTO lexicon(user_key,user_word,standard_word,count,updated_at) VALUES(?,?,?,1,?) "
                "ON CONFLICT(user_key,user_word,standard_word) DO UPDATE SET count=count+1,updated_at=excluded.updated_at",
                (user_key, user_word, standard, now),
            )
        return found

    def hint(self, user_key: str, limit: int = 3) -> str:
        maps = self.mappings(user_key, limit)
        if not maps:
            return ""
        bits = [f"{u}（对方指{s}）" for u, s in maps]
        return "用对方的词：" + "；".join(bits)

    def apply(self, user_key: str, reply: str, enabled: bool = True) -> str:
        if not enabled:
            return reply
        for user_word, standard in sorted(self.mappings(user_key), key=lambda kv: len(kv[1]), reverse=True):
            if standard and standard in reply and user_word not in reply:
                reply = reply.replace(standard, user_word)
        return reply
