"""LLM boundary with a hard policy gate.

Only four stages are allowed to touch a model:

* ``main_generate``    - the final reply
* ``solidify_small``   - optional shaped generation for a solidified path
* ``l1_rerank``        - optional rerank of already-retrieved local candidates
* ``style_screen``     - low-frequency screening of style-knowledge candidates

Everything else is a policy violation.  The per-turn write path never even
receives a gateway, which makes the zero-call guarantee structural as well as
audited.  ``style_screen`` runs only inside the periodic learner (at most a few
times per day/week), never per turn.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import ModelConfig

FORBIDDEN_STAGES = {
    "l2_write",
    "l2_distill",
    "l2_bootstrap",
    "l0_score",
    "l0_rebalance",
    "brightness",
    "confidence",
    "shortcircuit_match",
    "shortcircuit_feedback",
    "l1_trigger",
    "l1_retrieve",
    "rhythm",
    "topic_stack",
    "open_loop",
    "lexicon",
    "negative_pattern",
    "prune",
    "prune_l3",
}
ALLOWED_STAGES = {"main_generate", "solidify_small", "l1_rerank", "style_screen"}


class PolicyViolation(RuntimeError):
    pass


@dataclass
class LLMCall:
    stage: str
    model: str
    at: float
    tokens: int = 0
    ok: bool = True


@dataclass
class LLMCallLedger:
    calls: list[LLMCall] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, stage: str, model: str = "", tokens: int = 0, ok: bool = True) -> None:
        if stage in FORBIDDEN_STAGES or stage not in ALLOWED_STAGES:
            raise PolicyViolation(f"LLM call from forbidden stage: {stage}")
        with self._lock:
            self.calls.append(LLMCall(stage=stage, model=model, at=time.time(), tokens=tokens, ok=ok))

    def count(self, stage: str | None = None) -> int:
        with self._lock:
            if stage is None:
                return len(self.calls)
            return sum(1 for c in self.calls if c.stage == stage)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for call in self.calls:
                out[call.stage] = out.get(call.stage, 0) + 1
            return out

    def mark(self) -> int:
        with self._lock:
            return len(self.calls)

    def since(self, mark: int) -> list[LLMCall]:
        with self._lock:
            return list(self.calls[mark:])


class CallableProvider:
    """Injected provider used by adapters/tests."""

    def __init__(self, fn: Callable[[list[dict[str, str]], dict[str, Any]], str], name: str = "callable") -> None:
        self.fn = fn
        self.name = name
        self.model = name

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.fn(messages, kwargs)


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions client (stdlib only).

    The client is synchronous by design so the standalone library has no
    third-party dependencies.  In the AstrBot adapter the whole ``begin_turn``
    pipeline is executed via ``asyncio.to_thread``; optional small-model and
    rerank calls therefore never block AstrBot's event loop.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.name = cfg.provider
        self.model = cfg.name or "main"

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self.cfg.base_url:
            raise RuntimeError("main_model.base_url is empty")
        url = self.cfg.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {
            "model": kwargs.get("model") or self.cfg.name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.cfg.temperature),
            "max_tokens": kwargs.get("max_tokens", self.cfg.max_tokens),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        key = self.cfg.resolve_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:  # noqa: S310 - configured endpoint
            body = json.loads(resp.read().decode("utf-8"))
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected completion response: {body}") from exc


class LLMGateway:
    def __init__(
        self,
        main_provider: CallableProvider | OpenAICompatibleProvider | None,
        decision_provider: CallableProvider | OpenAICompatibleProvider | None,
        ledger: LLMCallLedger,
        *,
        strict: bool = True,
    ) -> None:
        self.main_provider = main_provider
        self.decision_provider = decision_provider
        self.ledger = ledger
        self.strict = strict

    def generate_main(self, messages: list[dict[str, str]], system: str) -> str:
        return self._call("main_generate", self.main_provider, messages, system)

    def generate_solidified(self, prompt: str, shape: dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "按给定形状补一句自然的中文回复。不要解释。"
                    f"目标长度 {shape.get('min_chars', 6)}-{shape.get('max_chars', 40)} 字；"
                    f"情绪 {shape.get('emotion', 'neutral')}；"
                    f"{'可以追问' if shape.get('ask_followup') else '不要追问'}；"
                    f"{'可以用表情' if shape.get('use_emoji') else '不要用表情'}。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._call("solidify_small", self.decision_provider, messages, "")

    def rerank(self, query: str, candidates: Sequence[str]) -> list[int]:
        prompt = (
            "只返回候选编号的 JSON 数组，按与用户消息的相关性排序，最多 3 个。\n"
            f"用户：{query}\n候选：{json.dumps(list(candidates), ensure_ascii=False)}"
        )
        text = self._call(
            "l1_rerank",
            self.decision_provider,
            [{"role": "user", "content": prompt}],
            "",
        )
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [int(x) for x in data]
        except Exception:
            pass
        return []

    def screen(self, prompt: str, system: str = "") -> str:
        """Low-frequency model screen for the style knowledge base.

        Prefers the small decision model; falls back to the main one when no
        decision model is configured.  Raises when neither exists, so the
        caller can keep the previous knowledge instead of storing raw text.
        """
        provider = self.decision_provider or self.main_provider
        return self._call(
            "style_screen",
            provider,
            [{"role": "user", "content": prompt}],
            system,
        )

    def _call(
        self,
        stage: str,
        provider: CallableProvider | OpenAICompatibleProvider | None,
        messages: list[dict[str, str]],
        system: str,
    ) -> str:
        if provider is None:
            raise RuntimeError(f"no provider configured for stage {stage}")
        # Raises before any network I/O if a forbidden stage tries to call out.
        self.ledger.record(stage, getattr(provider, "model", ""), 0, True)
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        return provider.chat(msgs)
