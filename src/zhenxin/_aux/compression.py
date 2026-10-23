"""Rule-based inbound compression for long messages (no model call)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..normalize import EMOTION_RE, EMOTION_WORDS, TIME_RE, content_tokens, is_question

_SENT_RE = re.compile(r"(?<=[。！？!?；;\n])")
_QUOTED_RE = re.compile(r"[“\"「『]([^”\"」』]{1,40})[”\"」』]")
_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_.+\-]{2,30})(?![A-Za-z0-9_])")
_INSTRUCTION_RE = re.compile(r"你|帮我|记得|必须|不要|别|几点|在哪|怎么|为什么|能不能|可以吗|需要我")
_FILLER_RE = re.compile(r"^(那个|就是|然后|其实|我觉得|感觉|就是说|话说|对了|欸|哎|唉)+")


@dataclass
class CompressedInbound:
    text: str
    compressed: bool
    original_len: int
    emotion: str = ""
    entities: list[str] = field(default_factory=list)
    intent: str = "statement"
    kept_sentences: int = 0


def extract_entities(text: str) -> list[str]:
    out: list[str] = []
    for match in _QUOTED_RE.finditer(text):
        out.append(match.group(1).strip())
    for match in _IDENT_RE.finditer(text):
        out.append(match.group(1))
    for match in TIME_RE.finditer(text):
        out.append(match.group(1))
    # Concrete nouns that recur in the conversation domain.
    for word in ("海盐", "团子", "阿哲", "甲方", "线稿", "绘本", "冰美式", "香菜", "电影", "医院", "中药"):
        if word in text:
            out.append(word)
    seen: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen


def detect_emotion(text: str) -> str:
    match = EMOTION_RE.search(text)
    return EMOTION_WORDS.get(match.group(0), "") if match else ""


def compress_inbound(text: str, threshold: int = 150, enabled: bool = True) -> CompressedInbound:
    text = (text or "").strip()
    entities = extract_entities(text)
    emotion = detect_emotion(text)
    if not enabled or len(text) <= threshold:
        return CompressedInbound(
            text=text,
            compressed=False,
            original_len=len(text),
            emotion=emotion,
            entities=entities,
            intent="question" if is_question(text) else "statement",
        )

    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    keep: list[str] = []
    kept_chars = 0
    for sentence in sentences:
        strong = bool(
            is_question(sentence)
            or _INSTRUCTION_RE.search(sentence)
            or _QUOTED_RE.search(sentence)
            or _IDENT_RE.search(sentence)
        )
        if strong:
            pieces = [sentence]
        else:
            # Keep the emotional/time clause, not the surrounding narrative.
            clauses = [c.strip() for c in re.split(r"[，,]", sentence) if c.strip()]
            pieces = [
                c
                for c in clauses
                if EMOTION_RE.search(c) or TIME_RE.search(c) or _IDENT_RE.search(c)
            ]
        if not pieces:
            continue
        for piece in pieces:
            if kept_chars + len(piece) > 200:
                break
            keep.append(piece)
            kept_chars += len(piece)
    if not keep:
        # Fall back to the first and last clause, still verbatim.
        clauses = [c.strip() for c in re.split(r"[，,。；;！!？?]", text) if c.strip()]
        keep = [c for c in (clauses[0] if clauses else "", clauses[-1] if clauses else "") if c]
    compressed_text = "".join(keep)
    if len(compressed_text) < len(text) * 0.92:
        compressed_text = compressed_text.rstrip("，,；;") + "〔叙述省略〕"
    else:
        compressed_text = text
    return CompressedInbound(
        text=compressed_text,
        compressed=len(compressed_text) < len(text),
        original_len=len(text),
        emotion=emotion,
        entities=entities,
        intent="question" if is_question(text) else "statement",
        kept_sentences=len(keep),
    )


def keywords_for_seed(text: str, limit: int = 4) -> list[str]:
    out = [e for e in extract_entities(text) if len(e) >= 2]
    for tok in content_tokens(text):
        if len(tok) >= 2 and tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out[:limit]
