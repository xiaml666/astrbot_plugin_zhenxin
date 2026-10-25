"""L3 cache-aware context pruning.

Three segments:

STABLE_PREFIX  system + persona  (never touched by pruning; contains no
               per-round or per-clue content)
RECENT_WINDOW  last N turns, evicted from the oldest end in coarse steps
EPHEMERAL_TAIL per-request hints and the L0 clue block, appended to the
               current user message and never persisted into history

Provider prompt caches are *prefix* caches: the first differing token makes
everything after it uncached.  Two rules follow, and both are enforced here:

1. anything that can change per round (L0 clues, recall, hints) belongs at the
   very end of the request, never in the system prompt;
2. the history window must stay byte-identical for several rounds at a time.
   Sliding it one turn per round would leave the cache with only the system
   prompt, so the window grows append-only up to a high-water mark and is then
   cut back down to ``recent_turns``.

When ``cache_aware_pruning`` is false this degrades to a plain sliding window:
no character budget and the previous ephemeral hint is retained.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import L3Config
from ..db import Database


BOUNDARY = "【边界】只依据已知线索和最近对话回答；不知道就直说不知道，不要编造对方没说过的事。"


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    other = max(0, len(text) - cjk)
    return cjk + (other + 3) // 4


@dataclass
class BuiltContext:
    stable_prefix: str
    prefix_hash: str
    recent_messages: list[dict[str, str]]
    prompt: str
    ephemeral_tail: str
    l0_block: str
    l0_chars: int
    l0_slots: list[int]
    recent_chars: int
    ephemeral_chars: int
    injected_tokens: int
    l0_in_tail: bool = False


class ContextBuilder:
    def __init__(self, db: Database, cfg: L3Config) -> None:
        self.db = db
        self.cfg = cfg

    def build_prefix(
        self,
        system_prompt: str,
        persona: str,
        l0_block: str,
        capabilities: str = "",
    ) -> str:
        parts = [system_prompt.strip()]
        if persona.strip():
            parts.append("【人格锚点】" + persona.strip())
        # Constant, so it costs one cached block and never breaks the prefix.
        # Without it the model answers a guessed clue as if the user had said
        # it (see reports: "阿哲是不是搞摄影的" -> fabricated role).
        parts.append(BOUNDARY)
        # Stable host capability notes (e.g. "voice is available").  They only
        # change when the host configuration changes, so the cache stays warm.
        if capabilities.strip():
            parts.append(capabilities.strip())
        if l0_block:
            parts.append(l0_block)
        return "\n\n".join(p for p in parts if p)

    def build(
        self,
        *,
        user_key: str,
        text: str,
        stable_prefix: str,
        l0_block: str,
        l0_slots: list[int],
        ephemeral_parts: list[str],
        previous_ephemeral: str = "",
        l0_tail: str = "",
    ) -> BuiltContext:
        hints = "\n".join(p for p in ephemeral_parts if p).strip()
        if self.cfg.include_ephemeral and hints:
            hints = hints[: self.cfg.ephemeral_max_chars]
        else:
            hints = ""
        # The clue block is never truncated by the hint budget: hints are
        # droppable, long-term clues are the point of the layer.
        ephemeral = "\n\n".join(p for p in (l0_tail, hints) if p)

        if not self.cfg.enabled:
            rows = self.db.recent_turns(user_key, self._floor_messages())
        elif self.cfg.cache_aware_pruning:
            rows = self._window(user_key)
        else:
            rows = self.db.recent_turns(user_key, self._floor_messages())
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]

        prompt = text
        if not self.cfg.cache_aware_pruning and previous_ephemeral:
            prompt = prompt + "\n\n〔上一轮附带提示〕\n" + previous_ephemeral[: self.cfg.ephemeral_max_chars]
        if ephemeral:
            prompt = prompt + "\n\n〔本轮提示，不进历史〕\n" + ephemeral

        recent_chars = sum(len(m["content"]) for m in messages)
        total_tokens = estimate_tokens(stable_prefix) + estimate_tokens(ephemeral) + estimate_tokens(prompt)
        for m in messages:
            total_tokens += estimate_tokens(m["content"]) + 4
        return BuiltContext(
            stable_prefix=stable_prefix,
            prefix_hash=hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest(),
            recent_messages=messages,
            prompt=prompt,
            ephemeral_tail=ephemeral,
            l0_block=l0_block,
            l0_chars=len(l0_block),
            l0_slots=list(l0_slots),
            recent_chars=recent_chars,
            ephemeral_chars=len(ephemeral),
            injected_tokens=total_tokens,
            l0_in_tail=bool(l0_tail),
        )

    # --------------------------------------------------------------- window
    def _floor_messages(self) -> int:
        return max(2, max(1, int(self.cfg.recent_turns)) * 2)

    def _window(self, user_key: str):
        """Return the recent window, growing append-only between cuts."""
        floor_msgs = self._floor_messages()
        high_msgs = floor_msgs + 2 * max(0, int(self.cfg.window_grow_turns))
        first_id = int(self.db.get_state(user_key, "window_first_id", 0) or 0)
        rows = self.db.turns_from(user_key, first_id) if first_id else []
        if not rows:
            rows = self.db.recent_turns(user_key, floor_msgs)
        rows = self._trim_turns(list(rows), floor_msgs, high_msgs)
        if rows:
            try:
                self.db.set_state(user_key, "window_first_id", int(rows[0]["id"]))
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        return rows

    def _trim_turns(self, rows: list, floor_msgs: int, high_msgs: int) -> list:
        max_chars = int(self.cfg.recent_max_chars)
        if max_chars > 0 and rows:
            total = sum(len(r["content"]) for r in rows)
            if total > max_chars:
                low_water = min(1.0, max(0.1, float(self.cfg.recent_char_low_water)))
                target = max(1, int(max_chars * low_water))
                while len(rows) > 1 and total > target:
                    drop = 1
                    if (
                        len(rows) > 1
                        and rows[0]["role"] == "user"
                        and rows[1]["role"] == "assistant"
                    ):
                        drop = 2
                    total -= sum(len(r["content"]) for r in rows[:drop])
                    rows = rows[drop:]
        if len(rows) > high_msgs:
            rows = rows[-floor_msgs:] if floor_msgs else rows[-2:]
        return rows

    @staticmethod
    def _trim_by_chars(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
        if max_chars <= 0:
            return []
        total = sum(len(m["content"]) for m in messages)
        out = list(messages)
        while out and total > max_chars:
            # Evict the oldest complete turn (user+assistant) when possible.
            removed = out.pop(0)
            total -= len(removed["content"])
            if out and removed["role"] == "user" and out[0]["role"] == "assistant":
                nxt = out.pop(0)
                total -= len(nxt["content"])
        # At least keep the most recent message, unless it alone exceeds budget.
        if not out and messages:
            out = [messages[-1]]
        return out
