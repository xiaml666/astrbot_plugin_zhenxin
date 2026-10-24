"""Reply rhythm controller: emotion baseline x relationship warmth x topic type.

Produces soft constraints (a very short hint) and an optional local
post-processor.  All arithmetic is rule/count based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..db import Database
from ..normalize import EMOTION_RE, EMOTION_WORDS, is_ack, is_cold, is_negation

_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\u2b00-\u2bff\ufe0f]+")
_QUESTION_RE = re.compile(r"[?？]|吗|呢|么")

WORK_WORDS = ("项目", "甲方", "交稿", "加班", "方案", "线稿", "稿子", "工作", "老板", "客户")
HEALTH_WORDS = ("医院", "医生", "病", "吐", "疼", "不舒服", "吃药")
REL_WORDS = ("团子", "阿哲", "妈妈", "妈", "朋友", "对象", "猫")
CASUAL_WORDS = ("电影", "吃", "喝", "周末", "睡", "玩", "咖啡")


@dataclass
class Rhythm:
    max_chars: int
    followup_allowed: bool
    emoji_allowed: bool
    tone: str
    warmth: float
    topic_type: str
    hint: str


def render_hint(rhythm: Rhythm) -> str:
    """Build the tail hint from the (possibly adjusted) rhythm fields."""
    bits = [
        f"{rhythm.max_chars}字内",
        {
            "steady": "少追问",
            "soft": "先接情绪",
            "warm": "自然收住",
            "bright": "轻快",
            "neutral": "简洁",
        }[rhythm.tone],
    ]
    if not rhythm.followup_allowed:
        bits.append("不追问")
    if not rhythm.emoji_allowed:
        bits.append("少表情")
    return "〔节奏：" + "，".join(bits) + "〕"


class RhythmController:
    def __init__(self, db: Database, enabled: bool = True, postprocess: bool = True) -> None:
        self.db = db
        self.enabled = enabled
        self.postprocess_enabled = postprocess

    def _state(self, user_key: str) -> dict:
        raw = self.db.get_state(user_key, "rhythm", {}) or {}
        return {
            "pos": int(raw.get("pos", 0)),
            "neg": int(raw.get("neg", 0)),
            "emotions": list(raw.get("emotions", []))[-8:],
        }

    def observe(self, user_key: str, text: str) -> dict:
        if not self.enabled:
            return self._state(user_key)
        state = self._state(user_key)
        if is_ack(text):
            state["pos"] += 1
        elif is_cold(text) or is_negation(text):
            state["neg"] += 1
        emo = self._emotion(text)
        if emo:
            state["emotions"].append(emo)
            state["emotions"] = state["emotions"][-8:]
        self.db.set_state(user_key, "rhythm", state)
        return state

    @staticmethod
    def _emotion(text: str) -> str:
        m = EMOTION_RE.search(text or "")
        return EMOTION_WORDS.get(m.group(0), "") if m else ""

    @staticmethod
    def _topic_type(text: str) -> str:
        if any(w in text for w in WORK_WORDS):
            return "work"
        if any(w in text for w in HEALTH_WORDS):
            return "health"
        if any(w in text for w in REL_WORDS):
            return "relationship"
        if any(w in text for w in CASUAL_WORDS):
            return "casual"
        return "general"

    def compute(self, user_key: str, text: str) -> Rhythm:
        if not self.enabled:
            return Rhythm(80, True, False, "neutral", 0.5, "general", "")
        state = self._state(user_key)
        pos, neg = state["pos"], state["neg"]
        warmth = max(0.0, min(1.0, 0.5 + (pos - neg) / 20.0))
        emotions = state["emotions"]
        emo = self._emotion(text) or (emotions[-1] if emotions else "")
        topic = self._topic_type(text)
        if emo in ("烦躁", "生气", "无语", "崩溃"):
            max_chars, tone, follow = 48, "steady", False
        elif emo in ("低落", "委屈", "孤单", "想念"):
            max_chars, tone, follow = 62, "soft", True
        elif emo == "疲惫":
            max_chars, tone, follow = 46, "warm", False
        elif emo == "开心":
            max_chars, tone, follow = 50, "bright", False
        else:
            max_chars, tone, follow = 72, "neutral", True
        if topic == "work":
            max_chars = min(max_chars, 52)
            follow = False
        elif topic == "health":
            max_chars = min(max_chars, 70)
            follow = True
        emoji = warmth >= 0.62 and tone in ("bright", "warm", "soft") and neg == 0
        if warmth < 0.35:
            max_chars = min(max_chars, 38)
            follow = False
            emoji = False
        result = Rhythm(max_chars, follow, emoji, tone, warmth, topic, "")
        result.hint = render_hint(result)
        return result

    def postprocess(self, reply: str, rhythm: Rhythm) -> str:
        if not self.enabled or not self.postprocess_enabled or not reply:
            return reply
        if not rhythm.emoji_allowed:
            reply = _EMOJI_RE.sub("", reply).strip()
        if not rhythm.followup_allowed:
            # Remove a trailing question sentence beyond the first clause.
            parts = re.split(r"(?<=[。！？!?])", reply)
            kept = [p for p in parts if p.strip()]
            if len(kept) > 1:
                kept = [p for p in kept if not _QUESTION_RE.search(p)] or kept[:1]
            reply = "".join(kept).strip()
        if len(reply) > rhythm.max_chars:
            cut = reply[: rhythm.max_chars]
            boundary = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"), cut.rfind("，"))
            if boundary >= int(rhythm.max_chars * 0.55):
                cut = cut[: boundary + 1]
            reply = cut.rstrip("，,；; ")
            if not reply.endswith(("。", "！", "？", "…")):
                reply += "。"
        return reply

    def apply_lexicon(self, user_key: str, reply: str, enabled: bool = True) -> str:
        return reply
