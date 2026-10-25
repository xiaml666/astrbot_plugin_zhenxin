"""Split a finished reply into a few chat bubbles.

This is a transport-only feature: the assistant turn is still stored as one
full text, the model is not called again, and nothing is added to the context.
A tiny, rule-based splitter keeps it predictable:

* split only on sentence punctuation / newlines;
* never cut inside quotes, brackets or fenced code blocks;
* merge fragments shorter than ``min_segment_chars`` into a neighbour;
* cap the number of bubbles (the tail is merged into the last one).

The caller decides when to use it (``split.enabled``) and how long to pause
between bubbles.
"""

from __future__ import annotations

from typing import Iterable

# Opening -> closing.  Closing characters are ignored unless they match the
# innermost opening, so a stray ")" cannot unbalance the whole split.
PAIRS = {
    "“": "”",
    "‘": "’",
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
    "「": "」",
    "『": "』",
    "《": "》",
    "{": "}",
}
DELIMITERS = "。！？!?…\n"
_TRAILING = " \t\r\n"


def _cut(text: str) -> list[str]:
    """Cut at delimiters that are outside quotes/brackets/code fences."""
    out: list[str] = []
    buf: list[str] = []
    stack: list[str] = []
    fence = False
    index = 0
    while index < len(text):
        if text.startswith("```", index):
            fence = not fence
            buf.append("```")
            index += 3
            continue
        char = text[index]
        if not fence:
            if char in PAIRS:
                stack.append(PAIRS[char])
            elif stack and char == stack[-1]:
                stack.pop()
        buf.append(char)
        index += 1
        if not fence and not stack and char in DELIMITERS:
            while index < len(text) and text[index] in DELIMITERS:
                buf.append(text[index])
                index += 1
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [part.strip(_TRAILING) for part in out if part.strip(_TRAILING)]


def _merge_short(parts: Iterable[str], min_segment_chars: int) -> list[str]:
    merged: list[str] = []
    for part in parts:
        part = part.strip(_TRAILING)
        if not part:
            continue
        if merged and len(part) < min_segment_chars:
            merged[-1] = merged[-1].rstrip(_TRAILING) + part
        elif merged and len(merged[-1]) < min_segment_chars:
            merged[-1] = merged[-1].rstrip(_TRAILING) + part
        else:
            merged.append(part)
    return merged


def split_reply(
    text: str,
    *,
    max_segments: int = 3,
    min_chars: int = 40,
    min_segment_chars: int = 8,
) -> list[str]:
    """Return ``[text]`` when it should stay one bubble, else 2..max segments."""
    raw = (text or "").strip()
    if not raw:
        return []
    cap = max(1, int(max_segments))
    if cap <= 1 or len(raw) < max(1, int(min_chars)):
        return [raw]
    parts = _merge_short(_cut(raw), max(1, int(min_segment_chars)))
    if len(parts) > cap:
        # Merge the overflow into the last allowed bubble so no text is lost.
        parts = parts[: cap - 1] + ["".join(parts[cap - 1:])]
    return parts or [raw]
