"""Long-term speaking-style adaptation, without any model call.

The rhythm controller reacts to the current message (emotion/topic).  This
module learns the user's *habitual* style across the conversation: how long
their messages are, whether they use emoji, whether they ask questions, and
whether they speak colloquially or formally.

It only tightens the existing rhythm limits (shorter replies, no emoji, fewer
follow-up questions), never loosens them, so it cannot increase the output
budget.  The learned hint is a short tail block: no system-prompt change, no
prefix-cache break, and no extra LLM call.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Callable

from ..db import Database
from .rhythm import Rhythm, render_hint

_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\u2b00-\u2bff\ufe0f]+"
)
_QUESTION_RE = re.compile(r"[?？]|吗|呢|么")
_COLLOQUIAL_RE = re.compile(r"啦|呀|嘛|哦|哈|哎|诶|嘿嘿|哈哈|呗|嘞")
_FORMAL_RE = re.compile(r"您|请问|麻烦您|谢谢您")
_SHORT_RE = re.compile(r"说短|短点|简短|太长|别那么长|少说|话少|简洁")
_LONG_RE = re.compile(r"多说|详细|展开|太短|多讲")
_NO_EMOJI_RE = re.compile(r"别(用)?表情|不要表情|少发表情|表情少")
_NO_QUESTION_RE = re.compile(r"别(老)?问|不要问|少问|别追问")
_CASUAL_RE = re.compile(r"别这么正式|自然点|随便点|别文绉绉")

STATE_KEY = "style_profile"
ALPHA = 0.18
MIN_SAMPLES = 5


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class Style:
    samples: int = 0
    avg_len: float = 30.0
    emoji_rate: float = 0.0
    question_rate: float = 0.0
    colloquial_rate: float = 0.0
    formal_rate: float = 0.0
    short_bias: float = 0.0
    no_emoji: bool = False
    no_question: bool = False
    updated_at: float = 0.0


class StyleLearner:
    def __init__(
        self,
        db: Database,
        enabled: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.enabled = bool(enabled)
        self.clock = clock

    def _load(self, user_key: str) -> Style:
        raw = self.db.get_state(user_key, STATE_KEY, {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        try:
            return Style(
                samples=int(raw.get("samples", 0) or 0),
                avg_len=_bounded(raw.get("avg_len", 30.0), 1.0, 400.0),
                emoji_rate=_bounded(raw.get("emoji_rate", 0.0), 0.0, 1.0),
                question_rate=_bounded(raw.get("question_rate", 0.0), 0.0, 1.0),
                colloquial_rate=_bounded(raw.get("colloquial_rate", 0.0), 0.0, 1.0),
                formal_rate=_bounded(raw.get("formal_rate", 0.0), 0.0, 1.0),
                short_bias=_bounded(raw.get("short_bias", 0.0), 0.0, 1.0),
                no_emoji=bool(raw.get("no_emoji", False)),
                no_question=bool(raw.get("no_question", False)),
                updated_at=float(raw.get("updated_at", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            return Style()

    def _save(self, user_key: str, style: Style) -> None:
        self.db.set_state(
            user_key,
            STATE_KEY,
            {
                "samples": style.samples,
                "avg_len": round(style.avg_len, 2),
                "emoji_rate": round(style.emoji_rate, 3),
                "question_rate": round(style.question_rate, 3),
                "colloquial_rate": round(style.colloquial_rate, 3),
                "formal_rate": round(style.formal_rate, 3),
                "short_bias": round(style.short_bias, 3),
                "no_emoji": style.no_emoji,
                "no_question": style.no_question,
                "updated_at": style.updated_at,
            },
        )

    def observe(self, user_key: str, text: str) -> Style:
        """Update the profile from one user message (user's own style)."""
        if not self.enabled:
            return self._load(user_key)
        text = (text or "").strip()
        if not text:
            return self._load(user_key)
        style = self._load(user_key)
        length = float(len(text))
        emoji = 1.0 if _EMOJI_RE.search(text) else 0.0
        question = 1.0 if _QUESTION_RE.search(text) else 0.0
        colloquial = 1.0 if _COLLOQUIAL_RE.search(text) else 0.0
        formal = 1.0 if _FORMAL_RE.search(text) else 0.0

        if style.samples == 0:
            style.avg_len = length
            style.emoji_rate = emoji
            style.question_rate = question
            style.colloquial_rate = colloquial
            style.formal_rate = formal
        else:
            style.avg_len = (1 - ALPHA) * style.avg_len + ALPHA * length
            style.emoji_rate = (1 - ALPHA) * style.emoji_rate + ALPHA * emoji
            style.question_rate = (1 - ALPHA) * style.question_rate + ALPHA * question
            style.colloquial_rate = (
                (1 - ALPHA) * style.colloquial_rate + ALPHA * colloquial
            )
            style.formal_rate = (1 - ALPHA) * style.formal_rate + ALPHA * formal

        # Explicit feedback is a strong, immediate signal.
        if _SHORT_RE.search(text):
            style.avg_len = min(style.avg_len, 16.0)
            style.short_bias = min(1.0, style.short_bias + 0.4)
        if _NO_EMOJI_RE.search(text):
            style.no_emoji = True
            style.emoji_rate = 0.0
        if _NO_QUESTION_RE.search(text):
            style.no_question = True
            style.question_rate = 0.0
        if _CASUAL_RE.search(text):
            style.colloquial_rate = max(style.colloquial_rate, 0.6)
        if _LONG_RE.search(text):
            # Budget-first: a request for longer replies does not raise the
            # rhythm cap.  We only avoid tightening it further.
            style.short_bias = 0.0

        style.samples += 1
        style.updated_at = self.clock()
        self._save(user_key, style)
        return style

    def adjust(self, user_key: str, rhythm: Rhythm) -> Rhythm:
        """Tighten a rhythm object using long-term style (never loosen it)."""
        if not self.enabled:
            return rhythm
        style = self._load(user_key)
        if style.samples < MIN_SAMPLES:
            return rhythm
        max_chars = rhythm.max_chars
        if style.avg_len <= 14 or style.short_bias >= 0.2:
            max_chars = min(max_chars, 46)
        follow = rhythm.followup_allowed and not (
            style.no_question
            or (style.samples >= 8 and style.question_rate < 0.05)
        )
        emoji = rhythm.emoji_allowed and not (
            style.no_emoji or (style.samples >= 8 and style.emoji_rate < 0.03)
        )
        adjusted = replace(
            rhythm,
            max_chars=max_chars,
            followup_allowed=follow,
            emoji_allowed=emoji,
        )
        adjusted.hint = render_hint(adjusted)
        return adjusted

    def hint(self, user_key: str, rhythm: Rhythm | None = None) -> str:
        """One short tail block describing the learned style."""
        if not self.enabled:
            return ""
        style = self._load(user_key)
        if style.samples < MIN_SAMPLES:
            return ""
        bits: list[str] = []
        if style.avg_len <= 14 or style.short_bias >= 0.2:
            bits.append("对方习惯短句")
        elif style.avg_len >= 45:
            bits.append("对方说得比较细")
        if style.formal_rate >= 0.3:
            bits.append("用“您”")
        elif style.colloquial_rate >= 0.25:
            bits.append("口语化一点")
        if rhythm is not None:
            if style.emoji_rate >= 0.2 and rhythm.emoji_allowed:
                bits.append("可以偶尔一个表情")
            elif (
                style.no_emoji
                or (style.samples >= 8 and style.emoji_rate < 0.03)
            ) and not rhythm.emoji_allowed:
                bits.append("对方不爱表情")
            if style.question_rate >= 0.3 and rhythm.followup_allowed:
                bits.append("可以顺着问一句")
            elif (
                style.no_question
                or (style.samples >= 8 and style.question_rate < 0.05)
            ) and not rhythm.followup_allowed:
                bits.append("少追问")
        # Keep the tail tiny: at most three traits.
        bits = bits[:3]
        if not bits:
            return ""
        return "〔他的习惯：" + "，".join(bits) + "〕"

    def describe(self, user_key: str) -> str:
        """Human-readable profile for the /zhenxin command."""
        style = self._load(user_key)
        if style.samples < MIN_SAMPLES:
            return f"还在观察（{style.samples}/{MIN_SAMPLES} 条）"
        bits: list[str] = []
        if style.avg_len <= 14 or style.short_bias >= 0.2:
            bits.append("短句")
        elif style.avg_len >= 45:
            bits.append("偏长")
        if style.formal_rate >= 0.3:
            bits.append("正式")
        elif style.colloquial_rate >= 0.25:
            bits.append("口语化")
        if style.no_emoji or style.emoji_rate < 0.03:
            bits.append("少表情")
        if style.no_question or style.question_rate < 0.05:
            bits.append("少追问")
        return "、".join(bits) or "普通"
