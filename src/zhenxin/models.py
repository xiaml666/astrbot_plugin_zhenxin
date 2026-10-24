from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Ball:
    slot: int
    content_seed: str
    brightness: float
    last_touched: float
    type: str = "fact"


@dataclass
class SolidPath:
    signature: str
    shape: dict[str, Any]
    confidence: float
    last_hit: float | None
    last_seen_round: int
    neg_streak: int
    degraded_until: float
    hits: int
    misses: int


@dataclass
class ContextPlan:
    user_key: str
    round_no: int
    text: str
    compressed_text: str
    stable_prefix: str
    l0_block: str
    l0_chars: int
    l0_slots: list[int]
    recent_messages: list[dict[str, str]]
    prompt: str
    ephemeral_tail: str
    prefix_hash: str
    wake: bool = False
    wake_trigger: str = ""
    recall: str = ""
    solid_reply: str | None = None
    solid_signature: str | None = None
    solid_confidence: float | None = None
    mode: str = "full"
    shadow: bool = False
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> list[dict[str, str]]:
        msgs = list(self.recent_messages)
        msgs.append({"role": "user", "content": self.prompt})
        return msgs

    @property
    def shortcircuit(self) -> bool:
        return self.solid_reply is not None


@dataclass
class TurnTrace:
    user_key: str
    round_no: int
    mode: str
    wake: bool
    wake_trigger: str
    prefix_hash: str
    prefix_stable: bool
    l0_chars: int
    l0_balls: int
    recent_chars: int
    ephemeral_chars: int
    injected_tokens: int
    shortcircuit_hit: bool
    solid_signature: str | None
    solid_confidence: float | None
    l2_llm_calls: int
    main_llm_calls: int
    decision_llm_calls: int
    reply_chars: int
    created_at: float
    # Actual provider usage, when AstrBot exposes LLMResponse.usage.
    input_tokens_actual: int = 0
    cached_tokens_actual: int = 0
    output_tokens_actual: int = 0
    actual_usage_available: bool = False
    shadow: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = self.__dict__.copy()
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@dataclass
class TurnResult:
    reply: str
    trace: TurnTrace
    plan: ContextPlan
