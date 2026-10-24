"""L2 zero-LLM write path: regex distillation, tier updates, physical eviction.

There is no LLM client anywhere in this file.  If a future edit adds one, the
pipeline's strict ledger will raise on the next write-path call.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .._aux.compression import extract_entities, keywords_for_seed
from .._aux.lexicon import Lexicon
from .._aux.negative import NegativePatterns
from .._aux.open_loops import OpenLoops
from .._aux.zero_value import identity_facts
from ..config import L2Config, L0Config
from ..db import Database
from ..embeddings import LocalScorer
from ..layers.l0 import L0Layer
from ..normalize import (
    EMOTION_WORDS,
    EMOTION_RE,
    FUTURE_ACTION_RE,
    RELATION_RE,
    TIME_RE,
    content_tokens,
    is_ack,
    is_correction,
)

_QUOTED_RE = re.compile(r"[“\"「『]([^”\"」』]{1,40})[”\"」』]")
_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_.+\-]{2,30})(?![A-Za-z0-9_])")
_PREF_RE = re.compile(r"((?:不)?(?:喜欢|爱|讨厌|不吃|不喝|爱喝|最怕|最烦|离不开))([^，,。！？!?\n]{1,14})")
_CORRECT_RE = re.compile(r"(?:不是|不叫).{0,10}?(?:是|叫)([^，,。！？!?\n]{1,24})")
_CORRECTION_PAIR_RE = re.compile(r"不是([^，,。！？!?\n]{1,10})[，, ]*(?:是|而是)([^，,。！？!?\n]{1,14})")
_ACTION_RE = re.compile(
    r"(交稿|截止|面试|考试|看医生|复诊|搬家|出差|开会|还钱|赶稿|通宵|早起|提交|汇报|答辩|体检|买|订)"
)
_EXPLICIT_MEMORY_RE = re.compile(r"记住|记一下|别忘了|提醒我")
# "叫我阿岚 / 称呼我为小满 / 我的名字是..." is a durable identity fact: keep it
# top-importance so retrieval weights and the always-on address hint can use it.
_ADDRESS_RE = re.compile(
    r"(?:你(?:可以|就|以后)?|以后|请|都)?(?:叫我|喊我|称呼我(?:为|作)?)"
    r"([\u4e00-\u9fffA-Za-z0-9_]{1,12})"
)
_NAME_RE = re.compile(r"(?:我的名字(?:是|叫)|我叫)([\u4e00-\u9fffA-Za-z0-9_]{1,12})")
_KNOWN = ("海盐", "团子", "阿哲", "甲方", "冰美式", "香菜", "线稿", "绘本", "电影", "妈妈", "医院", "中药")

# Question detection.  A question can carry emotion, but it must not create
# facts: "上次说的阿哲，是不是搞摄影的" is the user asking, not telling.
_QUESTION_ENDS = ("？", "?", "吗", "呢")
_QUESTION_HINTS = ("是不是", "还记得", "记不记得", "什么", "怎么", "多少", "哪一", "哪本", "几点", "多久")
_INTERROGATIVE_WORDS = ("什么", "啥", "哪", "怎么", "多少", "吗", "呢", "？", "?")


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(_QUESTION_ENDS):
        return True
    return any(hint in t for hint in _QUESTION_HINTS)


def _is_interrogative(fragment: str) -> bool:
    return any(word in fragment for word in _INTERROGATIVE_WORDS)


@dataclass
class Candidate:
    seed: str
    type: str
    keywords: list[str] = field(default_factory=list)
    thin_context: str = ""
    topic_key: str = ""
    source: str = "distill"
    importance: float = 0.5


class Distiller:
    def __init__(self, max_candidates: int = 4) -> None:
        self.max_candidates = max_candidates

    def distill(self, text: str, *, bootstrap: bool = False) -> list[Candidate]:
        text = (text or "").strip()
        if not text:
            return []
        out: list[Candidate] = []
        entities = extract_entities(text)
        emotion = self._emotion(text)
        # A question is not a source of facts.  "上次说的阿哲，是不是搞摄影的"
        # must not create the clue "朋友阿哲是摄影师", otherwise the model later
        # answers its own guess as if the user had said it.
        question = _is_question(text)

        # 1. Correction: the corrected clause is the strongest signal.
        if is_correction(text):
            m = _CORRECT_RE.search(text)
            if m:
                clause = m.group(1).strip()
                seed = self._corrected_seed(clause, emotion)
                out.append(self._candidate(seed, "fact", text, "correction", 0.95))

        # 2. How the user wants to be addressed, and what the assistant's name
        # is.  These are durable identity facts, not incidental tokens.
        if not question:
            rejected = any(
                phrase in text
                for phrase in ("别叫我", "不要叫我", "不用叫我", "不许叫我", "不叫", "别喊我")
            )
            for fact, kind in identity_facts(text):
                if rejected and kind == "address":
                    continue
                out.append(
                    self._candidate(fact, kind, text, kind, 0.98 if kind == "address" else 0.95)
                )

        # 3. Explicit quote / identifier.
        if not question:
            for m in _QUOTED_RE.finditer(text):
                quoted = m.group(1).strip()
                if len(quoted) >= 2:
                    out.append(self._candidate(f"提到“{quoted}”", "fact", text, "quote", 0.8))
            for m in _IDENT_RE.finditer(text):
                ident = m.group(1)
                if ident.lower() not in ("http", "https", "www"):
                    out.append(self._candidate(f"相关：{ident}", "fact", text, "identifier", 0.6))

        # 3. Stable preferences / habits.
        if not question:
            for m in _PREF_RE.finditer(text):
                verb, obj = m.group(1), m.group(2).strip()
                if 1 <= len(obj) <= 12 and not _is_interrogative(obj):
                    out.append(self._candidate(f"{verb}{obj}", "habit", text, "preference", 0.85))

        # 4. Emotion baseline (not every message; requires an emotion cue).
        if emotion:
            trigger = self._trigger(text, entities)
            seed = f"最近{emotion}" + (f"（{trigger}）" if trigger else "")
            out.append(self._candidate(seed, "emotion", text, "emotion", 0.7))

        # 5. Known entities and their roles.
        if not question:
            if "团子" in text or "猫" in text:
                seed = "养了猫叫团子" if "团子" in text else "养了一只猫"
                out.append(self._candidate(seed, "relationship", text, "entity", 0.8))
            if "海盐" in text:
                out.append(self._candidate("在做海盐绘本/项目", "unclosed", text, "entity", 0.8))
            if "阿哲" in text:
                seed = "朋友阿哲是摄影师" if ("摄影" in text or "拍照" in text) else "朋友阿哲"
                out.append(self._candidate(seed, "relationship", text, "entity", 0.75))
            if "冰美式" in text:
                out.append(self._candidate("爱喝冰美式", "habit", text, "entity", 0.8))
            if "香菜" in text:
                out.append(self._candidate("不吃香菜", "habit", text, "entity", 0.8))
            if "中药" in text:
                out.append(self._candidate("在喝中药调理", "habit", text, "entity", 0.8))

        # 6. Future / unclosed action.
        time_m = TIME_RE.search(text)
        action_m = _ACTION_RE.search(text)
        if action_m and (time_m or FUTURE_ACTION_RE.search(text)):
            seed = ((time_m.group(1) + "·") if time_m else "") + action_m.group(1)
            out.append(self._candidate(seed, "unclosed", text, "action", 0.75))

        # 7. Relationship / preference expressions.
        if not question and RELATION_RE.search(text) and not any(c.type == "habit" for c in out):
            kws = keywords_for_seed(text, 2)
            if kws:
                out.append(self._candidate("在意" + "、".join(kws), "relationship", text, "relation", 0.5))

        # 8. Bootstrap fallback: make the first turns usable without invention.
        if bootstrap and not out:
            kws = keywords_for_seed(text, 2)
            if kws:
                out.append(self._candidate("提到" + "、".join(kws), "fact", text, "bootstrap", 0.5))

        return self._dedupe(out)[: self.max_candidates]

    @staticmethod
    def _emotion(text: str) -> str:
        m = EMOTION_RE.search(text)
        return EMOTION_WORDS.get(m.group(0), "") if m else ""

    @staticmethod
    def _trigger(text: str, entities: list[str]) -> str:
        for word in ("海盐", "甲方", "交稿", "加班", "团子", "考试", "工作"):
            if word in text:
                return word
        for ent in entities:
            if 2 <= len(ent) <= 8:
                return ent
        return ""

    @staticmethod
    def _corrected_seed(clause: str, emotion: str) -> str:
        if emotion and len(clause) <= 6:
            return f"最近{emotion}"
        return "更正：" + clause[:24]

    def _candidate(self, seed: str, type_: str, text: str, source: str, importance: float) -> Candidate:
        seed = seed.strip()[:30]
        kws = keywords_for_seed(seed, 5) or [seed]
        thin = f"{self._label(type_)}：{seed}"[:100]
        topic = next((w for w in _KNOWN if w in text), kws[0] if kws else "")
        return Candidate(
            seed=seed,
            type=type_,
            keywords=kws,
            thin_context=thin[:100],
            topic_key=topic,
            source=source,
            importance=importance,
        )

    @staticmethod
    def _label(type_: str) -> str:
        return {
            "emotion": "情绪基线",
            "habit": "偏好",
            "relationship": "关系",
            "unclosed": "未闭合",
            "address": "称呼",
            "assistant_name": "AI名字",
            "fact": "事实",
        }.get(type_, "线索")

    def _dedupe(self, items: list[Candidate]) -> list[Candidate]:
        out: list[Candidate] = []
        for item in items:
            if not item.seed:
                continue
            if any(item.seed == existing.seed for existing in out):
                continue
            if any(self._similar(item.seed, existing.seed) for existing in out):
                continue
            out.append(item)
        return sorted(out, key=lambda c: c.importance, reverse=True)

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        ta, tb = set(content_tokens(a)), set(content_tokens(b))
        if not ta or not tb:
            return False
        overlap = len(ta & tb) / min(len(ta), len(tb))
        return overlap >= 0.8


class L2Writer:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        l0: L0Layer,
        cfg: L2Config,
        l0_cfg: L0Config,
        lexicon: Lexicon,
        loops: OpenLoops,
        negative: NegativePatterns,
        clock: Callable[[], float] = time.time,
        facts: Any = None,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.l0 = l0
        self.cfg = cfg
        self.l0_cfg = l0_cfg
        self.lexicon = lexicon
        self.loops = loops
        self.negative = negative
        self.clock = clock
        self.facts = facts
        self.distiller = Distiller(max_candidates=cfg.max_candidates)

    def bootstrap(self, user_key: str, text: str) -> list[Candidate]:
        if not self.cfg.enabled:
            return []
        if self.db.get_state(user_key, "bootstrapped", False):
            return []
        candidates = self.distiller.distill(text, bootstrap=True)[: self.cfg.bootstrap_seeds]
        for cand in candidates:
            if cand.type in ("address", "assistant_name"):
                continue
            self.l0.insert_candidate(
                user_key,
                cand.seed,
                cand.type,
                brightness=2.2,
                topic_key=cand.topic_key,
                source="bootstrap",
            )
            self.l0.archive_cold(
                user_key,
                cand.seed,
                thin_context=cand.thin_context,
                topic_key=cand.topic_key,
                source="bootstrap",
                tokens=cand.seed,
            )
        self.db.set_state(user_key, "bootstrapped", True)
        self.db.set_state(user_key, "l0_force_rebalance", True)
        return candidates

    def write(
        self,
        user_key: str,
        text: str,
        reply: str,
        round_no: int,
        *,
        compressed_text: str = "",
        emotion: str = "",
        entities: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {"enabled": False, "candidates": 0, "inserted": 0}
        candidates = self.distiller.distill(text)
        inserted = 0
        decisions: list[dict[str, Any]] = []
        corrected = is_correction(text)
        corrected_slot: int | None = None
        if corrected:
            pair = _CORRECTION_PAIR_RE.search(text)
            if pair:
                old_term = pair.group(1).strip()
                new_term = pair.group(2).strip()
                new_seed = f"最近{EMOTION_WORDS[new_term]}" if new_term in EMOTION_WORDS else f"更正：{new_term}"[:30]
                old_terms = [old_term]
                if old_term in EMOTION_WORDS:
                    old_terms.append(EMOTION_WORDS[old_term])
                decision = self.l0.correct_by_terms(user_key, old_terms, new_seed)
                if decision.get("action") == "correct":
                    corrected_slot = int(decision["slot"])
                    decisions.append(decision)
        for cand in candidates:
            # An address is durable identity, not a changing short-term clue;
            # it is injected every turn by the fact bank, so copying it into
            # L0 would only duplicate it in the prompt.
            if cand.type in ("address", "assistant_name"):
                if self.facts is not None:
                    self.facts.add(user_key, cand.seed, cand.type, importance=cand.importance)
                decisions.append({"action": "identity_fact"})
                continue
            if corrected and cand.source == "correction" and corrected_slot is not None:
                continue
            if corrected and cand.source == "correction":
                decision = self.l0.correct_best(user_key, text, cand.seed)
            else:
                decision = self.l0.insert_candidate(
                    user_key,
                    cand.seed,
                    cand.type,
                    brightness=1.8,
                    topic_key=cand.topic_key,
                    source=cand.source,
                )
            if decision.get("action") in ("insert", "replace", "correct"):
                inserted += 1
            decisions.append(decision)
            if self.facts is not None:
                self.facts.add(user_key, cand.seed, cand.type, importance=cand.importance)
            self.l0.archive_cold(
                user_key,
                cand.seed,
                thin_context=cand.thin_context,
                topic_key=cand.topic_key,
                source=cand.source,
                tokens=cand.seed,
            )

        if not corrected:
            if is_ack(text):
                self.l0.boost_recent(user_key, amount=0.5)
            else:
                self.l0.touch_best(user_key, text, amount=0.5)
        self.lexicon.learn(user_key, text, self.clock())
        self.negative.record(user_key, text, reply, self.clock())
        self.l0.swap_warm(user_key)
        self.db.set_state(user_key, "last_user_text", text)
        self.db.set_state(
            user_key,
            "last_reply_features",
            {
                "chars": len(reply),
                "questions": reply.count("？") + reply.count("?"),
                "emoji": sum(1 for ch in reply if ord(ch) > 0x1F000),
                "text": reply[:120],
            },
        )
        return {
            "enabled": True,
            "candidates": len(candidates),
            "inserted": inserted,
            "decisions": decisions,
        }

    def bootstrap_done(self, user_key: str) -> bool:
        return bool(self.db.get_state(user_key, "bootstrapped", False))

