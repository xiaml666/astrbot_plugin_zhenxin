"""Reply segmentation, ported from ``astrbot_plugin_splitter``.

The proven behaviour is kept:

* walk the *whole* message chain, not just plain text: a ``Reply`` quote stays
  on the first bubble, images can go alone, ``At``/face components follow the
  next/inline text instead of breaking the chain;
* smart split: paired quotes/brackets, fenced code, ``<think>`` blocks and
  Markdown tables are never cut;
* sentence-first: every natural boundary (``。！？；`` / newline) ends a bubble,
  however short — a two-character complete answer such as "有啊" is a message,
  not a fragment; commas are only used when a single sentence has grown past
  ``2 × min_segment_chars`` and would be unwieldy as one bubble;
* ``max_segments`` caps the bubbles and merges the tail without losing text;
* a bubble never ends on a clause-continuing mark (``，、；：``).

It is transport-only: the assistant turn is stored as one full text, no model
call is added, and nothing enters the prompt.  Component objects are duck-typed
(``type(comp).__name__`` / ``.text``) so the module is testable without
AstrBot; the caller passes a ``plain_factory`` that builds a real ``Plain``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

PAIR_MAP = {
    '"': '"',
    "“": "”",
    "‘": "’",
    "《": "》",
    "（": "）",
    "(": ")",
    "[": "]",
    "{": "}",
    "'": "'",
    "【": "】",
    "「": "」",
    "『": "』",
    "<": ">",
}
QUOTE_CHARS = {'"', "'", "`"}
SECONDARY_PATTERN = re.compile(r"[，,、；;]+")
# Punctuation that means "the sentence continues".  A bubble must never end
# with one of these: the break itself is the pause, and a trailing "，" reads
# as if the message was cut off (the model is even told not to end a line that
# way).  Only the transport copy is trimmed; the stored turn keeps the text.
_TRAILING_CONTINUATION_RE = re.compile(r"[，,、；;：:]+[ \t]*$")
# Defaults mirror the splitter's "simple" mode: sentence punctuation and
# semicolons/newlines; commas only split in balanced mode when needed.
DEFAULT_SPLIT_CHARS = ("。", "？", "！", "?", "!", "；", ";", "\n")
STRATEGIES = {"image": "单独", "at": "跟随下段", "face": "嵌入", "default": "跟随下段"}


def _kind(component: Any) -> str:
    return type(component).__name__.lower()


def _is_plain(component: Any) -> bool:
    return _kind(component) == "plain"


def _text(component: Any) -> str:
    return str(getattr(component, "text", "") or "")


def _split_pattern(chars: Iterable[str]) -> str:
    parts = [re.escape(str(char)) for char in chars if str(char)]
    parts.sort(key=len, reverse=True)
    return "(?:{})+".format("|".join(parts)) if parts else r"[\n]+"


def _process_text(
    text: str,
    pattern: str,
    segments: list[list[Any]],
    buffer: list[Any],
    plain_factory: Callable[[str], Any],
    *,
    start_weight: int,
    comma_threshold: int,
    last_break_primary: bool,
) -> tuple[int, bool]:
    """Smart split one plain-text component.

    Returns ``(weight, last_break_primary)``.  The flag tells the caller
    whether the most recent bubble boundary was a *natural* one (sentence
    punctuation / newline) or a comma cut, so a short final sentence is not
    merged away.
    """
    stack: list[str] = []
    compiled = re.compile(pattern)
    index = 0
    length = len(text)
    chunk = ""
    weight = start_weight

    while index < length:
        # Fenced code block.
        if text.startswith("```", index) and (index == 0 or text[index - 1] == "\n"):
            end = text.find("```", index + 3)
            if end != -1:
                chunk += text[index : end + 3]
                weight += end + 3 - index
                index = end + 3
            else:
                chunk += text[index:]
                weight += length - index
            continue

        # Reasoning block from reasoning models.
        if text.startswith("<think>", index) and (index == 0 or text[index - 1] == "\n"):
            end = text.find("</think>", index + 7)
            if end != -1:
                chunk += text[index : end + 8]
                weight += end + 8 - index
                index = end + 8
            else:
                chunk += text[index:]
                weight += length - index
            continue

        # Markdown table.
        if (index == 0 or text[index - 1] == "\n") and index < length and text[index] == "|":
            table_end = index
            position = index
            while position < length:
                line_end = text.find("\n", position)
                if line_end == -1:
                    line_end = length
                line = text[position:line_end].strip()
                if line.startswith("|") or (line and all(c in "-| :" for c in line)):
                    table_end = line_end + 1 if line_end < length else length
                    position = table_end
                else:
                    break
            if table_end > index + 1:
                table_text = text[index:table_end]
                chunk += table_text
                weight += sum(1 for c in table_text if not c.isspace())
                index = table_end
                continue

        match = compiled.match(text, pos=index)
        if match:
            delimiter = match.group()
            # Natural boundaries (sentence punctuation, newline) always end a
            # bubble, however short it is: "有啊。" is a complete message.  Two
            # guards only: never split while inside quotes/brackets, and never
            # emit a punctuation-only leading bubble.
            should_split = not stack or "\n" in delimiter
            if should_split and not chunk.strip():
                should_split = False
            # English decimal/abbreviation protection: don't cut "3.14" or "U.S.".
            if (
                should_split
                and "\n" not in delimiter
                and re.match(r"^[ \t.?!,;:\-']+$", delimiter)
            ):
                previous = text[index - 1] if index > 0 else ""
                following = text[index + len(delimiter)] if index + len(delimiter) < length else ""
                if re.match(r"^[a-zA-Z0-9 \t.?!,;:\-']$", previous) and re.match(
                    r"^[a-zA-Z0-9 \t.?!,;:\-']$", following
                ):
                    should_split = False
            if should_split:
                chunk += delimiter
                buffer.append(plain_factory(chunk))
                segments.append(buffer[:])
                buffer.clear()
                chunk = ""
                weight = 0
                last_break_primary = True
                index += len(delimiter)
            else:
                chunk += delimiter
                weight += len(delimiter)
                index += len(delimiter)
            continue

        # A comma only ends a bubble when the sentence has already grown past
        # ^comma_threshold chars.  The old "fill up to max_segments" rule cut
        # ordinary 15~20-char sentences in half; a clause break is a poor
        # substitute for a sentence break.
        if comma_threshold > 0 and weight >= comma_threshold and not stack:
            secondary = SECONDARY_PATTERN.match(text, pos=index)
            if secondary and chunk.strip():
                delimiter = secondary.group()
                chunk += delimiter
                buffer.append(plain_factory(chunk))
                segments.append(buffer[:])
                buffer.clear()
                chunk = ""
                weight = 0
                last_break_primary = False
                index += len(delimiter)
                continue

        char = text[index]
        if char in QUOTE_CHARS:
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                stack.append(char)
        elif not stack and char in PAIR_MAP:
            stack.append(char)
        elif stack and char == PAIR_MAP.get(stack[-1]):
            stack.pop()
        chunk += char
        index += 1
        weight += 0 if char.isspace() else 1

    if chunk:
        buffer.append(plain_factory(chunk))
    return weight, last_break_primary


def _merge_adjacent_plain(components: list[Any], plain_factory: Callable[[str], Any]) -> list[Any]:
    merged: list[Any] = []
    for component in components:
        if merged and _is_plain(component) and _is_plain(merged[-1]):
            merged[-1] = plain_factory(_text(merged[-1]) + _text(component))
        else:
            merged.append(component)
    return merged


def _trim_edge_blank_lines(components: list[Any]) -> None:
    first = next((c for c in components if _is_plain(c)), None)
    last = next((c for c in reversed(components) if _is_plain(c)), None)
    if first is not None:
        first.text = re.sub(r"^(?:[ \t]*\r?\n)+", "", _text(first))
    if last is not None:
        last.text = re.sub(r"(?:\r?\n[ \t]*)+$", "", _text(last))


def _strip_trailing_continuation(components: list[Any]) -> None:
    """Drop a clause-continuing mark (``，、；：``) from the bubble's edge.

    Segmentation can leave one there two ways: a model that ends a line with
    ``，`` and a newline boundary is honoured, or the balanced mode deliberately
    cuts at a comma to fill the bubble quota.  Both look like a truncated
    message, so the mark is dropped from the outgoing bubble.  The full text is
    still what gets stored for the next turn.
    """
    for component in reversed(components):
        if not _is_plain(component):
            continue
        text = _text(component)
        stripped = _TRAILING_CONTINUATION_RE.sub("", text)
        # Never turn a bubble into pure punctuation; that is worse than the
        # dangling comma and the empty-segment cleanup would drop it.
        if stripped.strip():
            component.text = stripped
        return


def _segment_text(components: Sequence[Any]) -> str:
    return "".join(_text(c) for c in components if _is_plain(c))


def _has_non_text(components: Sequence[Any]) -> bool:
    return any(not _is_plain(c) for c in components)


def split_chain(
    chain: Sequence[Any],
    *,
    plain_factory: Callable[[str], Any],
    max_segments: int = 2,
    min_segment_chars: int = 10,
    balanced: bool = True,
    enable_reply: bool = True,
    split_chars: Sequence[str] = DEFAULT_SPLIT_CHARS,
) -> list[list[Any]]:
    """Split a message chain into 1..``max_segments`` component lists.

    Guarantee for the caller: no bubble's text ends on a clause-continuing
    mark (``，、；：``), including when segmentation is configured off with
    ``max_segments=1``.  The stored turn keeps the model's original text.
    """
    cap = max(1, int(max_segments))
    minimum = max(1, int(min_segment_chars))
    pattern = _split_pattern(split_chars)
    components = [c for c in chain if c is not None]
    if cap <= 1:
        if not components:
            return []
        # The caller may still send this as one message, so apply the same edge
        # rule as for a real split.
        _strip_trailing_continuation(components)
        return [components]

    text_weight = sum(len(_text(c).replace(" ", "")) for c in components if _is_plain(c))
    solo_media = sum(
        1
        for c in components
        if not _is_plain(c) and STRATEGIES.get(_kind(c), STRATEGIES["default"]) == "单独"
    )
    # A comma may only end a bubble once that sentence is long enough to be
    # worth two chat bubbles; otherwise the natural boundaries decide.  The
    # floor is twice min_segment_chars (20 chars by default), so a normal
    # one-sentence reply is never cut at its comma.
    comma_threshold = 0
    if balanced and text_weight:
        target = max(1, cap - solo_media)
        comma_threshold = max(2 * minimum, math.ceil(text_weight / target))

    segments: list[list[Any]] = []
    buffer: list[Any] = []
    weight = 0
    last_break_primary = True
    for component in components:
        if _is_plain(component):
            text = _text(component)
            if not text:
                continue
            weight, last_break_primary = _process_text(
                text,
                pattern,
                segments,
                buffer,
                plain_factory,
                start_weight=weight,
                comma_threshold=comma_threshold,
                last_break_primary=last_break_primary,
            )
            continue
        kind = _kind(component)
        if "reply" in kind:
            if enable_reply:
                buffer.append(component)
            continue
        strategy = STRATEGIES.get(kind, STRATEGIES["default"])
        if strategy == "单独":
            if buffer:
                segments.append(buffer[:])
                buffer.clear()
            segments.append([component])
            weight = 0
            last_break_primary = True
        elif strategy == "跟随上段":
            if buffer:
                buffer.append(component)
                segments.append(buffer[:])
                buffer.clear()
                weight = 0
                last_break_primary = True
            elif segments:
                segments[-1].append(component)
            else:
                segments.append([component])
        elif strategy in ("跟随下段", "接下文"):
            if buffer:
                segments.append(buffer[:])
                buffer.clear()
                weight = 0
                last_break_primary = True
            buffer.append(component)
        else:
            buffer.append(component)
    if buffer:
        segments.append(buffer)
    segments = [s for s in segments if s]
    if not segments:
        return []

    for segment in segments:
        _trim_edge_blank_lines(segment)
    # No bubble may end on a clause-continuing mark (see the helper).  Run this
    # before the empty-Plain cleanup so a punctuation-only fragment is dropped
    # instead of being sent on its own.
    for segment in segments:
        _strip_trailing_continuation(segment)
    # Blank-line trimming can leave an empty Plain; drop those so an empty
    # bubble is never sent.  Segments with media components are kept.
    cleaned: list[list[Any]] = []
    for segment in segments:
        kept = [c for c in segment if not _is_plain(c) or _text(c) != ""]
        if kept and (
            any(not _is_plain(c) for c in kept)
            or any(_text(c).strip() for c in kept)
        ):
            cleaned.append(kept)
    segments = cleaned
    if not segments:
        return []

    # Cap the number of bubbles: merge the tail, keeping component order and
    # joining adjacent Plain pieces so no text is lost.
    if len(segments) > cap:
        tail: list[Any] = []
        for segment in segments[cap - 1:]:
            tail.extend(segment)
        segments = segments[: cap - 1] + [_merge_adjacent_plain(tail, plain_factory)]

    # A short *continuation* left by a comma cut ("……，而已") reads like a
    # stray bubble and is merged back.  A short bubble that came from a natural
    # boundary is a complete answer ("有啊。", "嗯。") and must stay, however few
    # characters it has.
    if len(segments) >= 2 and not last_break_primary:
        last = segments[-1]
        last_text = _segment_text(last).strip()
        if (
            0 < len(last_text) < minimum
            and not _has_non_text(last)
            and not _has_non_text(segments[-2])
        ):
            tail = segments.pop()
            segments[-1] = _merge_adjacent_plain(segments[-1] + tail, plain_factory)

    return segments


def segment_text(components: Sequence[Any]) -> str:
    """Text used by the caller for delay calculation and logging."""
    return _segment_text(components)


def calculate_delay(mode: str, next_text: str) -> float:
    """Delay before the *next* bubble.

    ``natural`` mirrors the splitter's linear speed (0.5s + 0.1s per char),
    ``fast`` 0.3s, ``slow`` 2.5s.  A hard ceiling keeps a long paragraph from
    stalling the pipeline for too long.
    """
    value = str(mode or "natural").strip().lower()
    if value == "fast":
        delay = 0.3
    elif value == "slow":
        delay = 2.5
    else:
        delay = 0.5 + 0.1 * len(next_text or "")
    return max(0.0, min(5.0, delay))


def merge_segments(segments: Sequence[Sequence[Any]], plain_factory: Callable[[str], Any]) -> list[Any]:
    """Flatten several segments into one component list (failure fallback)."""
    merged: list[Any] = []
    for segment in segments:
        merged.extend(segment)
    return _merge_adjacent_plain(merged, plain_factory)
