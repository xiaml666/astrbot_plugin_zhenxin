"""Per-turn directives and host capability notes.

Two problems are handled here, both without an extra model call:

* **capabilities** — the model used to answer "我没有嗓子" even though the host
  has TTS enabled.  When the host reports a voice provider, a stable line is
  added to the system prefix so the model knows the voice is its own.
* **redo requests** — "重来 / 重说一遍" turns often produced the redone answer
  plus an apology or a parenthetical note ("这次真的只有这三个字").  A short
  tail hint asks for the content only.

The texts are constants on purpose: they are part of the prompt contract and
can be asserted in tests.
"""

from __future__ import annotations

import re

# A redo is a *short* instruction.  A long message that merely mentions "重来"
# is a normal turn, so the length guard keeps the hint from hijacking it.
_REDO_RE = re.compile(r"(重来|重说|重做|重发|重新说|重新回|重新来|再来一遍|再说一遍)")
# Asking what the word means is not an instruction to redo anything.
_META_QUESTION_RE = re.compile(r"(是什么|什么意思|怎么说|怎么用|这个词)")
_REDO_MAX_CHARS = 60

REDO_HINT = (
    "〔本轮〕对方在要求你重做上一条回复：只输出重做后的内容本身。"
    "不要解释、道歉、总结，不要提“这次/重来/真的/只有这些/我懂了”，"
    "也不要加括号补充说明。说完就停。"
)

VOICE_LINE = (
    "你在这个会话里可以发语音：对方要语音时，系统会把你说的话合成成声音发出去，"
    "那就是你的声音。不要说没有嗓子、发不出语音、只能打字，也不要把语音说成是别人"
    "或系统替你念的；直接说要说的话，不要解释语音是怎么来的。"
)

# A request for voice is a short chat instruction, not a technical question.
_VOICE_RE = re.compile(
    r"(语音|声音|用声|发个声|出声|念出来|念一遍|读出来|嗓子|唱一句|唱一段|说给我听)"
)

VOICE_HINT = (
    "〔本轮〕对方要的是语音回复：直接说要说的话本身，不要写“发了/听到了吗/"
    "这次是真声音”这类旁白，也不要说明语音是怎么生成的；系统会把你的话合成语音发出去。"
    "对方要求慢一点时，用短句和停顿让它自然变慢，不要承诺具体的语速数值。"
)

_VOICE_STOP_RE = re.compile(r"(不用|不要|别|停止|取消|关掉).{0,4}(语音|声音)|打字|用文字|发文字")


def is_voice_stop(text: str) -> bool:
    """True when the user turns voice off ("别发语音了", "打字就行")."""
    value = str(text or "").strip()
    if not value:
        return False
    return bool(_VOICE_STOP_RE.search(value))


def is_voice_request(text: str) -> bool:
    """True when the user asks for a spoken reply ("用语音说…", "语音")."""
    value = str(text or "").strip()
    if not value:
        return False
    return bool(_VOICE_RE.search(value))


def is_redo_request(text: str) -> bool:
    """True for "重来 / 重说一遍" style instructions."""
    value = str(text or "").strip()
    if not value or len(value) > _REDO_MAX_CHARS:
        return False
    if _META_QUESTION_RE.search(value):
        return False
    return bool(_REDO_RE.search(value))


def capability_block(capabilities: dict[str, bool] | None) -> str:
    """Stable system-prefix block for the host features that are available."""
    caps = capabilities or {}
    lines: list[str] = []
    if caps.get("voice"):
        lines.append(VOICE_LINE)
    if not lines:
        return ""
    return "【能力】" + "\n".join(lines)
