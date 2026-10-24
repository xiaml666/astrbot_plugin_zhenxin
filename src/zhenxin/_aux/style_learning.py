"""Learn to talk like a real person — slowly, locally, with a hard cap.

Two sources feed one small knowledge base:

* the user's own messages: sentence length, particles, laughter, punctuation,
  emoji, question rate, favourite fillers and slang;
* periodic web-search snippets about natural human conversation, distilled by
  rules (no model call) into at most a few hundred characters.

The result is capped (500 characters by default) and the per-turn hint only
uses a 110-character slice, so this feature can never grow into a second
system prompt.  Web access is best-effort: on a blocked network the local
profile still works and the last good online note is kept.
"""

from __future__ import annotations

import html
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Callable

from ..config import StyleLearningConfig
from ..db import Database

_PARTICLES = ("啊", "呀", "吧", "呢", "哦", "嘛", "哈", "诶", "唉", "嗯", "呃", "哇", "啦", "咯", "呗")
_FILLERS = ("就是", "然后", "其实", "反正", "感觉", "有点", "可能", "应该", "真的", "不过")
_SLANG = ("笑死", "绝了", "离谱", "上头", "救命", "好家伙", "无语", "麻了", "服了", "破防", "emo")
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\u2b00-\u2bff\ufe0f]+"
)
_LAUGH_RE = re.compile(r"哈{2,}|嘿嘿|hhh|233")
_QUESTION_RE = re.compile(r"[?？]|吗|呢|么")
_ELLIPSIS_RE = re.compile(r"\.{3,}|。{2,}|…")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")

_QUERIES = (
    "site:zhihu.com 聊天 共情 倾听 说话习惯",
    "site:zhihu.com 微信聊天 语气词 短句 表情",
    "高情商 聊天 回复 共情 自然 技巧",
)

# Used when search returns nothing usable (offline / blocked / SEO spam).  It
# is deliberately short: the whole point is a small, stable knowledge base.
SEED_KNOWLEDGE = (
    "真人聊天常识：先接住情绪，再给建议；多用短句，一次只说一件事；"
    "少用客服腔和总结句；对方说累时先认同，不急着解决；"
    "适度用“嗯、啊、吧”等语气词；不连续追问；不堆表情；"
    "不确定就直说，不编造；偶尔自嘲，但不过度。"
)
_BAD_WORDS = (
    "英语",
    "英文",
    "口语",
    "美式",
    "单词",
    "背单词",
    "雅思",
    "托福",
    "翻译",
    "how's",
    "english",
    "grammar",
)

SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
BING_URL = "https://www.bing.com/search?q={query}&setlang=zh-CN"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

STATE_NEXT = "style_learn_next"
STATE_ONLINE_NEXT = "style_online_next"
STATE_ERROR = "style_learn_error"
STATE_UPDATED = "style_learn_updated"


def _clean(text: str) -> str:
    text = html.unescape(_TAG_RE.sub(" ", text or ""))
    return _WS_RE.sub(" ", text).strip()


def _fetch(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=max(2.0, float(timeout))) as response:
        raw = response.read(700_000)
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _snippets(page: str) -> list[str]:
    out: list[str] = []
    for pattern in (
        r'class="result__snippet"[^>]*>(.*?)</a>',
        r"<li[^>]*class=\"b_algo\"[^>]*>.*?<p[^>]*>(.*?)</p>",
        r'<p class="b_lineclamp[^"]*"[^>]*>(.*?)</p>',
        r'<meta[^>]+name="description"[^>]+content="([^"]+)"',
    ):
        for match in re.finditer(pattern, page, re.S | re.I):
            text = _clean(match.group(1))
            if 20 <= len(text) <= 260:
                out.append(text)
            if len(out) >= 12:
                break
        if len(out) >= 8:
            break
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        key = item[:40]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:10]


def search_notes(
    queries: tuple[str, ...] = _QUERIES,
    *,
    timeout: float = 8.0,
    endpoint: str = "",
) -> tuple[list[str], str]:
    """Best-effort public search; returns ``(snippets, error)``."""
    collected: list[str] = []
    errors: list[str] = []
    deadline = time.time() + max(4.0, float(timeout) * 3.0)
    for query in queries:
        if time.time() >= deadline:
            break
        encoded = urllib.parse.quote(query)
        urls = []
        if endpoint:
            urls.append(endpoint.format(query=encoded) if "{query}" in endpoint else f"{endpoint}{encoded}")
        else:
            urls.extend([SEARCH_URL.format(query=encoded), BING_URL.format(query=encoded)])
        for url in urls:
            if time.time() >= deadline:
                break
            try:
                page = _fetch(url, timeout)
            except Exception as exc:  # noqa: BLE001 - network is best effort
                errors.append(f"{url.split('/')[2]}: {exc}")
                continue
            found = _snippets(page)
            if found:
                collected.extend(found)
                break
        if len(collected) >= 8:
            break
    return collected, "; ".join(errors[-2:])


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "") if part.strip()]


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    return cjk / max(1, len(text))


def _distill_web(snippets: list[str], limit: int) -> str:
    """Keep only sentences that describe conversational behaviour.

    Public search is full of English-learning and SEO pages.  A result must be
    substantially Chinese, must not mention language learning, and must contain
    a behaviour keyword; otherwise the knowledge base would fill up with
    "How's it going" instead of how people actually chat.
    """
    keywords = (
        "说话",
        "聊天",
        "对话",
        "沟通",
        "语气",
        "共情",
        "倾听",
        "回应",
        "短句",
        "口头",
        "自然",
        "表情",
        "提问",
        "称呼",
        "打断",
        "节奏",
        "边界",
        "情绪",
    )
    picked: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        lowered = snippet.lower()
        if any(bad in lowered for bad in _BAD_WORDS):
            continue
        for sentence in _sentences(snippet):
            sentence = re.sub(r"^[#*\-\s\d.、]+", "", sentence).strip()
            if len(sentence) < 10 or len(sentence) > 80:
                continue
            if _cjk_ratio(sentence) < 0.45:
                continue
            if not any(word in sentence for word in keywords):
                continue
            normal = re.sub(r"\s+", "", sentence)
            key = normal[:18]
            if key in seen:
                continue
            seen.add(key)
            picked.append(sentence)
            if sum(len(item) for item in picked) >= max(60, limit):
                break
        if sum(len(item) for item in picked) >= max(60, limit):
            break
    return _trim("；".join(picked), limit)


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit - 1)]
    for sep in ("；", "。", "，", " "):
        index = cut.rfind(sep)
        if index >= max(20, limit // 3):
            cut = cut[:index]
            break
    return cut.rstrip("；。，, ") + "…"


class HumanStyleLearner:
    def __init__(
        self,
        db: Database,
        cfg: StyleLearningConfig,
        style: Any,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.style = style
        self.clock = clock

    # ------------------------------------------------------------------ read
    def knowledge(self, user_key: str) -> str:
        rows = self.db.fetchall(
            "SELECT source,content FROM style_knowledge WHERE user_key=?"
            " ORDER BY CASE source WHEN 'local' THEN 0 WHEN 'seed' THEN 1 ELSE 2 END",
            (user_key,),
        )
        parts = [str(row["content"]).strip() for row in rows if str(row["content"]).strip()]
        return _trim("；".join(parts), max(120, int(self.cfg.max_chars)))

    def hint(self, user_key: str) -> str:
        if not self.enabled:
            return ""
        text = self.knowledge(user_key)
        if not text:
            return ""
        text = re.sub(r"^(?:用户习惯|真人聊天常识)[:：]\s*", "", text)
        text = _trim(text, max(40, int(self.cfg.hint_chars)))
        return "〔像真人一点：" + text + "〕"

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def status(self, user_key: str) -> dict[str, Any]:
        rows = self.db.fetchall(
            "SELECT source,content,chars,updated_at FROM style_knowledge WHERE user_key=?",
            (user_key,),
        )
        sources = {str(row["source"]): dict(row) for row in rows}
        turns = int(
            self.db.scalar(
                "SELECT COUNT(*) FROM turns WHERE user_key=? AND role='user'", (user_key,), 0
            )
        )
        return {
            "enabled": self.enabled,
            "user": user_key,
            "turns": turns,
            "knowledge": self.knowledge(user_key),
            "chars": len(self.knowledge(user_key)),
            "max_chars": int(self.cfg.max_chars),
            "updated_at": float(self.db.get_state(user_key, STATE_UPDATED, 0) or 0),
            "next_at": float(self.db.get_state(user_key, STATE_NEXT, 0) or 0),
            "online_next_at": float(self.db.get_state(user_key, STATE_ONLINE_NEXT, 0) or 0),
            "last_error": str(self.db.get_state(user_key, STATE_ERROR, "") or ""),
            "online": bool(self.cfg.online),
            "sources": {
                key: {
                    "chars": int(value.get("chars") or 0),
                    "updated_at": float(value.get("updated_at") or 0),
                    "preview": _trim(str(value.get("content") or ""), 160),
                }
                for key, value in sources.items()
            },
        }

    def recent_users(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT user_key,COUNT(*) turns,MAX(created_at) last FROM turns"
            " WHERE role='user' GROUP BY user_key ORDER BY last DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        out = []
        for row in rows:
            key = str(row["user_key"])
            profile = self.style._load(key) if hasattr(self.style, "_load") else None
            out.append(
                {
                    "user": key,
                    "turns": int(row["turns"] or 0),
                    "last": float(row["last"] or 0),
                    "samples": int(getattr(profile, "samples", 0) or 0),
                }
            )
        return out

    def due(self, user_key: str, now: float | None = None) -> bool:
        if not self.enabled:
            return False
        now = self.clock() if now is None else now
        next_at = float(self.db.get_state(user_key, STATE_NEXT, 0) or 0)
        if next_at and now < next_at:
            return False
        turns = int(
            self.db.scalar(
                "SELECT COUNT(*) FROM turns WHERE user_key=? AND role='user'", (user_key,), 0
            )
        )
        return turns >= max(1, int(self.cfg.min_turns))

    def online_due(self, user_key: str, now: float | None = None) -> bool:
        if not (self.enabled and self.cfg.online):
            return False
        now = self.clock() if now is None else now
        next_at = float(self.db.get_state(user_key, STATE_ONLINE_NEXT, 0) or 0)
        return not next_at or now >= next_at

    def due_users(self, limit: int = 20) -> list[str]:
        rows = self.db.fetchall(
            "SELECT user_key,COUNT(*) c FROM turns WHERE role='user'"
            " GROUP BY user_key ORDER BY MAX(created_at) DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [str(row["user_key"]) for row in rows if self.due(str(row["user_key"]))]

    # ----------------------------------------------------------------- write
    def _store(self, user_key: str, source: str, content: str) -> None:
        now = self.clock()
        content = _trim(content, int(self.cfg.max_chars))
        self.db.execute(
            "INSERT INTO style_knowledge(user_key,source,content,chars,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?) ON CONFLICT(user_key,source) DO UPDATE SET"
            " content=excluded.content,chars=excluded.chars,updated_at=excluded.updated_at",
            (user_key, source, content, len(content), now, now),
        )

    def _local_profile(self, user_key: str) -> str:
        rows = self.db.fetchall(
            "SELECT content FROM turns WHERE user_key=? AND role='user'"
            " ORDER BY id DESC LIMIT 160",
            (user_key,),
        )
        texts = [str(row["content"]).strip() for row in reversed(rows) if str(row["content"]).strip()]
        if len(texts) < 3:
            return ""
        lengths = [len(text) for text in texts if len(text) >= 1]
        avg = sum(lengths) / max(1, len(lengths))
        short_ratio = sum(1 for length in lengths if length <= 12) / max(1, len(lengths))
        emoji = sum(1 for text in texts if _EMOJI_RE.search(text)) / len(texts)
        question = sum(1 for text in texts if _QUESTION_RE.search(text)) / len(texts)
        laugh = sum(1 for text in texts if _LAUGH_RE.search(text)) / len(texts)
        ellipsis = sum(1 for text in texts if _ELLIPSIS_RE.search(text)) / len(texts)
        particles = Counter()
        for text in texts:
            for particle in _PARTICLES:
                particles[particle] += text.count(particle)
        fillers = Counter({word: sum(text.count(word) for text in texts) for word in _FILLERS})
        slang = Counter({word: sum(text.count(word) for text in texts) for word in _SLANG})
        top_particles = [word for word, count in particles.most_common(3) if count >= max(2, len(texts) // 12)]
        top_fillers = [word for word, count in fillers.most_common(2) if count >= max(2, len(texts) // 10)]
        top_slang = [word for word, count in slang.most_common(2) if count >= max(2, len(texts) // 14)]

        bits: list[str] = []
        if short_ratio >= 0.55 or avg <= 14:
            bits.append(f"多用短句（平均{avg:.0f}字）")
        elif avg >= 45:
            bits.append(f"说得比较细（平均{avg:.0f}字）")
        else:
            bits.append(f"句子不长不短（平均{avg:.0f}字）")
        if top_particles:
            bits.append("常用" + "、".join(f"“{word}”" for word in top_particles) + "收尾")
        if top_fillers:
            bits.append("开口常说" + "、".join(f"“{word}”" for word in top_fillers))
        if top_slang:
            bits.append("会用" + "、".join(f"“{word}”" for word in top_slang))
        if emoji >= 0.2:
            bits.append("偶尔带表情")
        else:
            bits.append("很少用表情")
        if laugh >= 0.15:
            bits.append("常发“哈哈”")
        if ellipsis >= 0.15:
            bits.append("爱用省略号")
        if question >= 0.35:
            bits.append("常主动追问")
        elif question <= 0.08:
            bits.append("很少提问")
        if short_ratio >= 0.55:
            bits.append("先接一句再讲正事")
        bits.append("像微信聊天一样随意，不写小作文")
        return _trim("用户习惯：" + "；".join(bits), max(120, int(self.cfg.max_chars * 0.7)))

    def refresh(self, user_key: str, *, online: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "message": "说话学习已关闭"}
        error = ""
        if not self.db.fetchone(
            "SELECT 1 FROM style_knowledge WHERE user_key=? AND source='seed' LIMIT 1",
            (user_key,),
        ):
            self._store(user_key, "seed", SEED_KNOWLEDGE)
        if self.cfg.learn_user:
            local = self._local_profile(user_key)
            if local:
                self._store(user_key, "local", local)
        if online and self.cfg.online:
            snippets, error = search_notes(
                timeout=float(self.cfg.search_timeout_s), endpoint=str(self.cfg.search_endpoint or "")
            )
            note = ""
            if snippets:
                note = _distill_web(snippets, max(80, int(self.cfg.max_chars * 0.8)))
                if note:
                    self._store(user_key, "web", note)
            if not note and not error:
                error = "未获取到可用的中文结果"
        now = self.clock()
        interval = max(0.2, float(self.cfg.interval_days if online else self.cfg.local_interval_hours / 24.0))
        self.db.set_state(user_key, STATE_NEXT, now + interval * 86400.0)
        if online:
            self.db.set_state(
                user_key,
                STATE_ONLINE_NEXT,
                now + max(0.2, float(self.cfg.interval_days)) * 86400.0,
            )
        self.db.set_state(user_key, STATE_UPDATED, now)
        self.db.set_state(user_key, STATE_ERROR, error)
        return {"ok": True, "status": self.status(user_key), "error": error}
