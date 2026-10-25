"""Configuration model and loaders.

Every behavioural module has an explicit switch.  Defaults are intentionally
conservative: local-only decision making, cache-aware pruning on, optional
small-model rerank off.  The main model is only ever addressed by the host
adapter (or by the standalone runner); no update path receives an LLM client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

try:  # PyYAML is optional; the bundled subset loader is enough for this config.
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - exercised on bare installs
    _yaml = None

from . import _mini_yaml


@dataclass
class ModelConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str = "ZHENXIN_MAIN_API_KEY"
    name: str = ""
    temperature: float = 0.8
    max_tokens: int = 512
    timeout_s: float = 60.0

    def resolve_api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class EmbeddingConfig:
    # auto -> ONNX when both runtime and files exist, otherwise hashing fallback.
    backend: str = "auto"
    model_path: str = "models/bge-small-zh.onnx"
    tokenizer_path: str = "models/tokenizer.json"
    max_length: int = 256
    dims: int = 384


@dataclass
class L0Config:
    enabled: bool = True
    slots: int = 10
    inject: int = 3
    # True: the clue block rides in the per-request tail instead of the system
    # prompt.  Provider prompt caches are prefix caches, so a block that sits
    # inside the system prompt invalidates everything after it whenever a clue
    # changes.  Measured on the live instance: one L0 change cost ~7.9k uncached
    # input tokens (~37% of a 32k-session's uncached input).
    tail_injection: bool = True
    max_seed_chars: int = 30
    max_inject_chars: int = 260
    decay: float = 0.99
    warm_decay: float = 0.995
    hysteresis: float = 2.0
    min_dwell_rounds: int = 8
    rebalance_interval: int = 8
    type_bonus: dict[str, float] = field(
        default_factory=lambda: {
            # Durable, user-declared things outrank transient mood: with only
            # three injected slots, "不吃香菜" must beat "最近生气（甲方）".
            "habit": 0.35,
            "fact": 0.25,
            "relationship": 0.15,
            "unclosed": 0.2,
            "emotion": 0.1,
        }
    )


@dataclass
class ShortCircuitConfig:
    enabled: bool = True
    hit_threshold: float = 3.0
    template_confidence: float = 5.0
    initial_confidence: float = 2.0
    repeat_boost: float = 1.0
    ack_boost: float = 0.5
    continue_boost: float = 1.0
    negative_penalty: float = 1.0
    neg_degrade_threshold: int = 2
    degrade_hours: float = 24.0
    max_literal_chars: int = 24
    max_hit_chars: int = 8
    use_small_model: bool = False
    max_reply_chars: int = 80


@dataclass
class L1Config:
    enabled: bool = True
    low_l0_threshold: float = 0.3
    candidate_min_score: float = 0.10
    recall_max_chars: int = 100
    max_recalls: int = 2
    wake_cooldown_rounds: int = 8
    rerank: bool = False
    rerank_top_k: int = 5


@dataclass
class L2Config:
    enabled: bool = True
    warm_limit: int = 30
    max_candidates: int = 4
    open_loop_limit: int = 5
    open_loop_ttl_rounds: int = 25
    cold_max_rows: int = 800
    bootstrap_seeds: int = 2


@dataclass
class L3Config:
    enabled: bool = True
    cache_aware_pruning: bool = True
    recent_turns: int = 8
    # The window grows append-only up to ``recent_turns + window_grow_turns``
    # and is then cut back to ``recent_turns``.  Evicting one turn per round
    # makes the history prefix differ every round, so the provider cache can
    # never extend past the system prompt; coarse eviction keeps the prefix
    # byte-identical for ~3 of every 4 rounds.
    window_grow_turns: int = 4
    recent_max_chars: int = 2200
    # When the char budget is hit, cut down to this fraction so the next
    # eviction is several rounds away instead of next round.
    recent_char_low_water: float = 0.7
    ephemeral_max_chars: int = 420
    include_ephemeral: bool = True
    # Assumption used only by the cache-hit estimate / dashboards.
    cache_ttl_seconds: int = 300
    # replace: plugin owns the recent window. off: never touch req.contexts.
    # (A legacy "prepend" value is treated as off; duplicating the host window
    # and the plugin window doubled the context tokens, so it was removed.)
    host_context_mode: str = "replace"


@dataclass
class AuxConfig:
    topic_stack: bool = True
    compression: bool = True
    rhythm: bool = True
    lexicon: bool = True
    open_loops: bool = True
    negative_patterns: bool = True
    compression_threshold: int = 150
    topic_depth: int = 5
    rhythm_postprocess: bool = True
    lexicon_postprocess: bool = True
    lexicon_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Long-term memory retrieval weights and the forgetting cycle.

    Retrieval is deliberately not "semantic top-k only": the score blends
    semantic similarity, declared importance, Ebbinghaus retention, usage
    evidence, exact token overlap and a small kind prior.  Explicit address
    facts ("叫我阿岚") get an extra boost and are always injected, because a
    name the user stated once should never need a lucky retrieval to survive.
    """

    enabled: bool = True
    max_facts: int = 300
    top_k: int = 3
    min_score: float = 0.30
    address_k: int = 1
    # Retrieval score = sum(weight * component) + bonuses.
    weight_semantic: float = 0.50
    weight_importance: float = 0.18
    weight_recency: float = 0.12
    weight_usage: float = 0.10
    weight_kind: float = 0.05
    exact_boost: float = 0.12
    address_boost: float = 0.22
    # Forgetting cycle (Ebbinghaus-style retention with spacing effect).
    cycle_days: float = 14.0
    decay_half_life_days: float = 30.0
    treasure_threshold: float = 0.62
    shallow_threshold: float = 0.30
    treasure_limit: int = 120
    shallow_limit: int = 240
    trash_ttl_days: float = 7.0
    # A daily view is what the page opens on; the total view is one click away.
    page_limit: int = 400


@dataclass
class StyleLearningConfig:
    """Slow, periodic, model-screened speech-preference knowledge base.

    Two sources: sentences where the user explicitly says how they want to be
    talked to, and periodic snippets from forums / chat / social platforms.
    Nothing is stored before a model screen approves it; the learner never
    imitates the user's wording.  The result is capped hard (500 chars by
    default) so the per-turn hint can never grow into a second prompt.
    """

    enabled: bool = True
    online: bool = True
    interval_days: float = 7.0
    local_interval_hours: float = 24.0
    min_turns: int = 6
    max_chars: int = 500
    hint_chars: int = 110
    # Built-in public search only (forums/chat platforms + result-domain check).
    search_timeout_s: float = 8.0


@dataclass
class SplitConfig:
    """Optional reply segmentation ("对话分段").

    Defaults follow the proven ``astrbot_plugin_splitter`` simple mode: up to
    5 bubbles, fragments under 10 chars merged, natural increasing delay.
    This is transport-only: no extra model call, no extra context tokens, and
    the stored assistant turn remains the full text.
    """

    enabled: bool = False
    max_segments: int = 5
    # Fragments shorter than this are merged into a neighbour.
    min_segment_chars: int = 10
    # natural: 0.5s + 0.1s per char of the next bubble; fast: 0.3s; slow: 2.5s.
    delay_mode: str = "natural"


@dataclass
class ProactiveConfig:
    """Optional proactive messages ("主动找你说话").

    Off by default: an unsolicited message is a product decision, not a
    technical default.  All timing is per conversation and persisted in the
    ``state`` table, so a restart cannot spam a session.
    """

    enabled: bool = False
    min_idle_minutes: int = 180
    min_interval_hours: float = 12.0
    max_per_day: int = 2
    active_window_hours: float = 72.0
    quiet_start: int = 23
    quiet_end: int = 8
    only_private: bool = True
    use_llm: bool = False
    max_chars: int = 80
    template: str = ""


@dataclass
class FeedbackConfig:
    enabled: bool = True
    github_issues: str = "https://github.com/xiaml666/astrbot_plugin_zhenxin/issues/new"


@dataclass
class MetricsConfig:
    enabled: bool = True
    jsonl_path: str = "logs/metrics.jsonl"
    log_traces: bool = True
    # Used only to price actual provider usage when the host exposes it.
    currency: str = "CNY"
    input_price_per_million: float = 2.0
    cached_input_price_per_million: float = 0.5
    output_price_per_million: float = 8.0


@dataclass
class Config:
    db_path: str = "data/zhenxin.sqlite3"
    system_prompt: str = (
        "你是对方熟悉的朋友，说话自然、简短，不说教。根据长期线索和最近对话回应，"
        "不要声称自己拥有数据库或检索系统。"
    )
    persona: str = "语气温和但有边界，偶尔冷幽默；不连续追问，不堆表情。"
    shadow_mode: bool = False
    # Canary escape hatch: while shadow_mode is on, these user_key prefixes
    # bypass shadow and run live.  Keep it narrow (e.g. one webchat session).
    live_allowlist: list[str] = field(default_factory=list)
    strict_zero_llm: bool = True
    main_model: ModelConfig = field(default_factory=ModelConfig)
    decision_model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    l0: L0Config = field(default_factory=L0Config)
    shortcircuit: ShortCircuitConfig = field(default_factory=ShortCircuitConfig)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    aux: AuxConfig = field(default_factory=AuxConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    style_learning: StyleLearningConfig = field(default_factory=StyleLearningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        return _build_dataclass(cls, data or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            text = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".json":
                data = json.loads(text)
            elif _yaml is not None:
                data = _yaml.safe_load(text) or {}
            else:
                data = _mini_yaml.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"config root must be a mapping: {path}")
        cfg = cls.from_dict(data)
        cfg.apply_env()
        return cfg

    def apply_env(self, environ: dict[str, str] | None = None) -> "Config":
        env = environ if environ is not None else os.environ
        # Convenience aliases.
        aliases = {
            "ZHENXIN_DB_PATH": ("db_path",),
            "ZHENXIN_SHADOW_MODE": ("shadow_mode",),
            "ZHENXIN_STRICT_ZERO_LLM": ("strict_zero_llm",),
            "ZHENXIN_CACHE_AWARE_PRUNING": ("l3", "cache_aware_pruning"),
        }
        for key, path_parts in aliases.items():
            if key in env:
                _set_path(self, path_parts, _parse_env_value(env[key]))
        prefix = "ZHENXIN__"
        for key, raw in env.items():
            if not key.startswith(prefix):
                continue
            parts = [p.lower() for p in key[len(prefix):].split("__") if p]
            if not parts:
                continue
            if parts[0] == "main_model" and parts[-1] == "api_key":
                continue  # secrets are read by api_key_env, never stored in config
            _set_path(self, tuple(parts), _parse_env_value(raw))
        return self

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(self)


def _parse_env_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        hint = hints.get(f.name)
        if isinstance(hint, type) and is_dataclass(hint) and isinstance(val, dict):
            kwargs[f.name] = _build_dataclass(hint, val)
        elif is_dataclass(val):
            kwargs[f.name] = val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _set_path(obj: Any, parts: tuple[str, ...], value: Any) -> None:
    cur = obj
    for part in parts[:-1]:
        if not hasattr(cur, part):
            return
        cur = getattr(cur, part)
    last = parts[-1]
    if hasattr(cur, last):
        current = getattr(cur, last)
        if isinstance(current, bool) and not isinstance(value, bool):
            value = bool(value)
        setattr(cur, last, value)
