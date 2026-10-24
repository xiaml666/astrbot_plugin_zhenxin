"""The five-layer pipeline and short-circuit orchestrator."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Sequence

from ._aux.compression import CompressedInbound, compress_inbound
from ._aux.lexicon import Lexicon
from ._aux.negative import NegativePatterns
from ._aux.open_loops import OpenLoops
from ._aux.rhythm import RhythmController
from ._aux.style import StyleLearner
from ._aux.style_learning import HumanStyleLearner
from ._aux.topic_stack import TopicStack
from .config import Config
from .db import Database
from .embeddings import LocalScorer
from .layers.l0 import L0Layer
from .layers.l1 import L1Layer
from .layers.l2 import L2Writer
from .layers.l3 import BuiltContext, ContextBuilder
from .layers.memory import FactBank
from .layers.shortcircuit import ShortCircuit, is_meta_negative
from .llm import LLMCallLedger, LLMGateway, OpenAICompatibleProvider, PolicyViolation
from .metrics import MetricsRecorder
from .models import ContextPlan, TurnResult, TurnTrace
from .normalize import is_correction, normalize
from .proactive import ProactiveScheduler


class ZhenxinEngine:
    def __init__(
        self,
        config: Config | None = None,
        *,
        db: Database | None = None,
        clock: Callable[[], float] = time.time,
        main_provider: Any = None,
        decision_provider: Any = None,
    ) -> None:
        self.cfg = config or Config()
        self.clock = clock
        self.db = db or Database(self.cfg.db_path)
        self.ledger = LLMCallLedger()
        self.scorer = LocalScorer(self.cfg.embedding)
        self.l0 = L0Layer(self.db, self.scorer, self.cfg.l0, clock)
        self.l0.set_warm_limit(self.cfg.l2.warm_limit)
        self.l0.set_cold_limit(self.cfg.l2.cold_max_rows)
        self.shortcircuit = ShortCircuit(self.db, self.scorer, self.cfg.shortcircuit, clock)
        self.l1 = L1Layer(self.db, self.scorer, self.cfg.l1, clock)
        self.l3 = ContextBuilder(self.db, self.cfg.l3)
        self.facts = FactBank(self.db, self.scorer, clock, cfg=self.cfg.memory)
        self.topic = TopicStack(self.db, self.scorer, self.cfg.aux.topic_depth, self.cfg.aux.topic_stack)
        self.rhythm = RhythmController(
            self.db, self.cfg.aux.rhythm, self.cfg.aux.rhythm_postprocess
        )
        self.style = StyleLearner(self.db, self.cfg.aux.style, clock)
        self.style_learning = HumanStyleLearner(
            self.db, self.cfg.style_learning, self.style, clock
        )
        self.proactive = ProactiveScheduler(self.db, self.cfg.proactive, clock)
        self.lexicon = Lexicon(self.db, self.cfg.aux.lexicon_aliases, self.cfg.aux.lexicon)
        self.loops = OpenLoops(
            self.db,
            self.scorer,
            limit=self.cfg.l2.open_loop_limit,
            ttl_rounds=self.cfg.l2.open_loop_ttl_rounds,
            enabled=self.cfg.aux.open_loops and self.cfg.l2.enabled,
            archive=self.l0.archive_cold,
        )
        self.negative = NegativePatterns(self.db, self.cfg.aux.negative_patterns)
        self.l2 = L2Writer(
            self.db,
            self.scorer,
            self.l0,
            self.cfg.l2,
            self.cfg.l0,
            self.lexicon,
            self.loops,
            self.negative,
            clock,
            facts=self.facts,
        )
        self.metrics = MetricsRecorder(self.db, self.cfg.metrics, self.ledger)
        main = main_provider if main_provider is not None else self._build_provider(self.cfg.main_model)
        decision = (
            decision_provider
            if decision_provider is not None
            else self._build_provider(self.cfg.decision_model)
        )
        self.gateway = LLMGateway(main, decision, self.ledger, strict=self.cfg.strict_zero_llm)

    @staticmethod
    def _build_provider(cfg: Any) -> Any:
        if not getattr(cfg, "enabled", False):
            return None
        if not getattr(cfg, "base_url", "") or not getattr(cfg, "name", ""):
            return None
        return OpenAICompatibleProvider(cfg)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ turns
    def begin_turn(
        self,
        user_key: str,
        text: str,
        *,
        round_no: int | None = None,
        host_system_prompt: str = "",
        history: Sequence[dict[str, str]] | None = None,
        shadow: bool | None = None,
    ) -> ContextPlan:
        raw = normalize(text)
        if round_no is None:
            round_no = int(self.db.incr_state(user_key, "round_no", 1))
        else:
            current = int(self.db.get_state(user_key, "round_no", 0) or 0)
            if round_no > current:
                self.db.set_state(user_key, "round_no", round_no)
        shadow = self.cfg.shadow_mode if shadow is None else shadow
        self.note_user_activity(user_key)
        # Upgrade recovery: older versions / a cleared memory bank may have no
        # stored name even though the conversation is still in `turns`.
        try:
            if not self.db.get_state(user_key, "identity_backfill_done", False):
                self.facts.backfill_from_turns(user_key)
                self.db.set_state(user_key, "identity_backfill_done", True)
        except Exception:  # noqa: BLE001 - recovery must never break a turn
            pass
        self._import_history(user_key, history)
        # A retraction ("不用提醒了", "别叫我阿岚") is a strong, immediate
        # signal; the cue check is one regex, the scan only runs on a hit.
        try:
            self.facts.scan_zero_value(user_key, raw)
        except Exception:  # noqa: BLE001 - memory hygiene must never break a turn
            pass
        # Cheap periodic zero-value cleanup: one indexed count every round, a
        # rule-based sweep every 20 rounds and only once memory has some size.
        try:
            if round_no % 20 == 0:
                self.facts.sweep(user_key, now=self.clock())
        except Exception:  # noqa: BLE001 - memory hygiene must never break a turn
            pass

        self.l0.ensure_slots(user_key)
        compressed: CompressedInbound = compress_inbound(
            text,
            threshold=self.cfg.aux.compression_threshold,
            enabled=self.cfg.aux.compression,
        )
        self.l2.bootstrap(user_key, raw)
        self.db.set_state(user_key, "last_inbound_text", raw)
        self.rhythm.observe(user_key, raw)
        self.style.observe(user_key, raw)
        self.topic.observe(user_key, compressed.text)
        loop_state = self.loops.observe(user_key, raw, round_no)
        self.l0.decay(user_key)

        # Feedback is applied to the path used on the previous turn, then the
        # pointer is cleared so one utterance cannot be counted twice.
        previous_sig = self.db.get_state(user_key, "last_path_sig", None)
        previous_round = int(self.db.get_state(user_key, "last_path_round", 0) or 0)
        previous_user = self.db.get_state(user_key, "last_path_user_text", "") or ""
        previous_solid = bool(self.db.get_state(user_key, "last_path_was_solid", False))
        feedback: dict[str, Any] = {"applied": False}
        # Feedback is strongest immediately after a shaped reply.  A delayed
        # correction ("不是…"/"别老说模板") is also attributed to the previous
        # matched path even if that turn used the full path.
        explicit_negative = is_correction(raw) or is_meta_negative(raw)
        if (
            self.cfg.shortcircuit.enabled
            and previous_sig
            and previous_round == round_no - 1
            and (previous_solid or explicit_negative)
        ):
            feedback = self.shortcircuit.feedback(
                user_key,
                raw,
                previous_sig,
                previous_user,
                previous_solid,
                round_no,
            )
        self.db.set_state(user_key, "last_path_sig", None)
        self.db.set_state(user_key, "last_path_round", 0)
        self.db.set_state(user_key, "last_path_was_solid", False)

        l0_block, l0_slots = self.l0.current_block(user_key, round_no)
        injected_seeds = [b.content_seed for b in self.l0.active_balls(user_key) if b.slot in l0_slots]
        l0_best = 0.0
        if l0_slots:
            l0_best, _ = self.l0.injected_relevance(user_key, raw, l0_slots)

        match = None
        hit = False
        if self.cfg.shortcircuit.enabled:
            match = self.shortcircuit.match(
                user_key,
                raw,
                round_no,
                last_path_signature=previous_sig,
            )
            hit = bool(match.hit)
            self.db.set_state(user_key, "last_path_sig", match.signature.key)
            self.db.set_state(user_key, "last_path_round", round_no)
            self.db.set_state(user_key, "last_path_user_text", raw)
            self.db.set_state(user_key, "last_path_was_solid", hit)

        wake = False
        recall = ""
        wake_trigger = ""
        wake_detail: dict[str, Any] = {}
        ephemeral_parts: list[str] = []
        # Long-term fact bank: pull the few facts that actually match this
        # message.  They ride in the tail, so depth is free (no prefix cache hit
        # is lost) and facts that left the recent window are still reachable.
        if self.cfg.memory.enabled:
            recalled_facts = self.facts.recall(user_key, raw, k=self.cfg.memory.top_k)
            # A stated name is not a retrieval lottery: it rides along every
            # turn once the user has said how they want to be addressed.  Keep
            # it out of the generic memory line so the model sees it once.
            address = self.facts.address(user_key)
            if address:
                recalled_facts = [text for text in recalled_facts if text != address]
            if recalled_facts:
                ephemeral_parts.append("〔相关记忆〕" + "；".join(recalled_facts))
            if address:
                name = address.replace("称呼我", "").strip("“”\"' ")
                ephemeral_parts.append(f"〔称呼〕对方希望你称呼TA为「{name}」")
            assistant_name = self.facts.assistant_name(user_key)
            if assistant_name:
                name = assistant_name.replace("你叫", "").strip("“”\"' ")
                ephemeral_parts.append(f"〔你的名字〕对方希望你叫「{name}」")
        if not hit:
            l1_result = self.l1.decide(
                user_key,
                raw,
                round_no,
                l0_best=l0_best,
                injected_seeds=injected_seeds,
                injected_slots=l0_slots,
                loop_touched=loop_state.get("touched", []),
                reranker=self._l1_reranker if (self.cfg.l1.rerank and self.gateway.decision_provider) else None,
            )
            wake = l1_result.wake
            recall = l1_result.recall
            wake_trigger = l1_result.trigger
            wake_detail = l1_result.detail
            if wake:
                ephemeral_parts.append(recall)
            if loop_state.get("hint"):
                ephemeral_parts.append(loop_state["hint"])
            topic_hint = self.topic.hint(user_key)
            if topic_hint:
                ephemeral_parts.append(topic_hint)
            rhythm = self.rhythm.compute(user_key, raw)
            rhythm = self.style.adjust(user_key, rhythm)
            if rhythm.hint:
                ephemeral_parts.append(rhythm.hint)
            style_hint = self.style.hint(user_key, rhythm)
            if style_hint:
                ephemeral_parts.append(style_hint)
            human_hint = self.style_learning.hint(user_key)
            if human_hint:
                ephemeral_parts.append(human_hint)
            lexicon_hint = self.lexicon.hint(user_key)
            if lexicon_hint:
                ephemeral_parts.append(lexicon_hint)
            negative_hint = self.negative.hint(user_key)
            if negative_hint:
                ephemeral_parts.append(negative_hint)
            # A plugin was enabled while the bot was running.  Let the model
            # mention it naturally once, and keep the page banner until ack.
            alert = self.db.get_state("__global__", "conflict_alert", None)
            if isinstance(alert, dict) and not alert.get("injected"):
                items = alert.get("items") or []
                names = "、".join(str(item.get("display") or item.get("name")) for item in items[:2])
                if names:
                    ephemeral_parts.append(
                        "〔插件提醒〕用户刚启用了" + names
                        + "，它可能也会改写上下文或系统提示。若发现记忆重复、回复变慢或变贵，"
                        "可以在真心页面把「由真心接管对话」改成不接管，或先停用其中一个。"
                    )
                    alert = dict(alert)
                    alert["injected"] = True
                    self.db.set_state("__global__", "conflict_alert", alert)
        else:
            rhythm = self.rhythm.compute(user_key, raw)
            rhythm = self.style.adjust(user_key, rhythm)

        host_system = host_system_prompt or self.cfg.system_prompt
        # Cache-safe layout: everything that can change between rounds (the L0
        # clue block) goes after the history, never into the system prompt.
        # A prefix cache would otherwise be invalidated from the clue block
        # onward on every clue change.
        tail_l0 = bool(self.cfg.l0.tail_injection) and bool(l0_block)
        stable_prefix = self.l3.build_prefix(
            host_system, self.cfg.persona, "" if tail_l0 else l0_block
        )
        previous_ephemeral = self.db.get_state(user_key, "last_ephemeral", "") or ""
        built: BuiltContext = self.l3.build(
            user_key=user_key,
            text=compressed.text,
            stable_prefix=stable_prefix,
            l0_block=l0_block,
            l0_slots=l0_slots,
            ephemeral_parts=ephemeral_parts,
            previous_ephemeral=previous_ephemeral,
            l0_tail=l0_block if tail_l0 else "",
        )
        self.db.set_state(user_key, "last_ephemeral", built.ephemeral_tail)

        plan = ContextPlan(
            user_key=user_key,
            round_no=round_no,
            text=raw,
            compressed_text=compressed.text,
            stable_prefix=built.stable_prefix,
            l0_block=built.l0_block,
            l0_chars=built.l0_chars,
            l0_slots=built.l0_slots,
            recent_messages=built.recent_messages,
            prompt=built.prompt,
            ephemeral_tail=built.ephemeral_tail,
            prefix_hash=built.prefix_hash,
            wake=wake,
            wake_trigger=wake_trigger,
            recall=recall,
            mode="solid" if hit else ("l1" if wake else "full"),
            shadow=shadow,
            metadata={
                "ledger_mark": self.ledger.mark(),
                "compressed": compressed.compressed,
                "emotion": compressed.emotion,
                "entities": compressed.entities,
                "l0_best": l0_best,
                "wake_detail": wake_detail,
                "loop_state": loop_state,
                "feedback": feedback,
                "rhythm": {
                    "max_chars": rhythm.max_chars,
                    "tone": rhythm.tone,
                    "warmth": rhythm.warmth,
                    "topic_type": rhythm.topic_type,
                },
                "match": _match_dict(match),
                "injected_seeds": injected_seeds,
                "recent_chars": built.recent_chars,
                "injected_tokens": built.injected_tokens,
            },
        )
        if hit and match is not None:
            plan.solid_signature = match.signature.key
            plan.solid_confidence = match.path.confidence
            small = None
            if self.cfg.shortcircuit.use_small_model and self.gateway.decision_provider is not None:
                def small(payload: dict[str, Any], _text: str = raw) -> str:
                    prompt = f"对方刚说：{_text}\n参考上一句：{payload.get('template', '')}"
                    return self.gateway.generate_solidified(prompt, payload.get("shape", {}))

            plan.solid_reply = self.shortcircuit.render(user_key, match, small_generator=small)
            plan.mode = "solid"
            plan.wake = False
            plan.wake_trigger = ""
            plan.ephemeral_tail = ""
            plan.prompt = compressed.text
        return plan

    def complete_turn(
        self,
        plan: ContextPlan,
        reply: str,
        *,
        main_llm_calls: int = 0,
        decision_llm_calls: int = 0,
        actual_usage: dict[str, int] | None = None,
    ) -> TurnResult:
        if plan.completed:
            raise RuntimeError("turn already completed")
        rhythm = self.style.adjust(
            plan.user_key, self.rhythm.compute(plan.user_key, plan.text)
        )
        if plan.shadow:
            # The host already sent this exact text.  Post-processing it would
            # make the baseline arm look different from what the user saw and
            # silently corrupt every A/B comparison.
            final_reply = reply or ""
        else:
            final_reply = self.rhythm.postprocess(reply or "", rhythm)
            final_reply = self.lexicon.apply(plan.user_key, final_reply, self.cfg.aux.lexicon_postprocess)
        l2_result = self.l2.write(
            plan.user_key,
            plan.text,
            final_reply,
            plan.round_no,
            compressed_text=plan.compressed_text,
            emotion=plan.metadata.get("emotion", ""),
            entities=plan.metadata.get("entities", []),
        )
        self.db.append_turn(plan.user_key, plan.round_no, "user", plan.compressed_text, self.clock())
        self.db.append_turn(plan.user_key, plan.round_no, "assistant", final_reply, self.clock())

        calls = self.ledger.since(int(plan.metadata.get("ledger_mark", 0)))
        l2_calls = sum(1 for c in calls if c.stage.startswith("l2_"))
        main_calls = main_llm_calls + sum(1 for c in calls if c.stage == "main_generate")
        decision_calls = decision_llm_calls + sum(1 for c in calls if c.stage in ("solidify_small", "l1_rerank"))
        if l2_calls:
            raise PolicyViolation("L2 write path issued an LLM call")

        last_hash = self.db.get_state(plan.user_key, "last_prefix_hash", None)
        stable = True if last_hash is None else (last_hash == plan.prefix_hash)
        self.db.set_state(plan.user_key, "last_prefix_hash", plan.prefix_hash)
        mode = plan.mode
        if plan.shadow:
            mode = "shadow_solid" if plan.solid_reply else ("shadow_l1" if plan.wake else "shadow_full")
        usage = actual_usage or {}
        usage_available = bool(usage) and any(
            key in usage for key in ("input_other", "input_cached", "output")
        )
        trace = TurnTrace(
            user_key=plan.user_key,
            round_no=plan.round_no,
            mode=mode,
            wake=bool(plan.wake),
            wake_trigger=plan.wake_trigger,
            prefix_hash=plan.prefix_hash,
            prefix_stable=stable,
            l0_chars=plan.l0_chars,
            l0_balls=len(plan.l0_slots),
            recent_chars=plan.metadata.get("recent_chars", sum(len(m["content"]) for m in plan.recent_messages)),
            ephemeral_chars=len(plan.ephemeral_tail),
            injected_tokens=_plan_tokens(plan),
            shortcircuit_hit=bool(plan.solid_reply),
            solid_signature=plan.solid_signature,
            solid_confidence=plan.solid_confidence,
            l2_llm_calls=l2_calls,
            main_llm_calls=main_calls,
            decision_llm_calls=decision_calls,
            reply_chars=len(final_reply),
            created_at=self.clock(),
            input_tokens_actual=int(usage.get("input_other", 0) or 0),
            cached_tokens_actual=int(usage.get("input_cached", 0) or 0),
            output_tokens_actual=int(usage.get("output", 0) or 0),
            actual_usage_available=usage_available,
            shadow=plan.shadow,
            detail={
                "l2": l2_result,
                "feedback": plan.metadata.get("feedback", {}),
                "wake": plan.metadata.get("wake_detail", {}),
                "loop": plan.metadata.get("loop_state", {}),
                "rhythm": plan.metadata.get("rhythm", {}),
                "match": plan.metadata.get("match", {}),
            },
        )
        plan.completed = True
        self.metrics.record(trace)
        return TurnResult(reply=final_reply, trace=trace, plan=plan)

    def handle_turn(
        self,
        user_key: str,
        text: str,
        generate: Callable[[ContextPlan], str] | None = None,
        *,
        round_no: int | None = None,
        host_system_prompt: str = "",
        history: Sequence[dict[str, str]] | None = None,
        shadow: bool | None = None,
    ) -> TurnResult:
        plan = self.begin_turn(
            user_key,
            text,
            round_no=round_no,
            host_system_prompt=host_system_prompt,
            history=history,
            shadow=shadow,
        )
        if plan.shortcircuit and not plan.shadow:
            return self.complete_turn(plan, plan.solid_reply or "")
        generator = generate or self.default_generate
        target = self._baseline_plan(plan) if plan.shadow else plan
        reply = generator(target)
        external_call = 1 if (generate is not None and not plan.shadow) else 0
        return self.complete_turn(plan, reply, main_llm_calls=external_call)

    def _l1_reranker(self, query: str, candidates: list[Any]) -> list[int]:
        order = self.gateway.rerank(query, [c.seed_tokens for c in candidates])
        mapped: list[int] = []
        for idx in order:
            if 0 <= int(idx) < len(candidates):
                mapped.append(candidates[int(idx)].row_id)
        return mapped

    def default_generate(self, plan: ContextPlan) -> str:
        if self.gateway.main_provider is None:
            raise RuntimeError("no main model configured; pass a generate callable")
        return self.gateway.generate_main(plan.messages, plan.stable_prefix)

    def load_history(self, plan: ContextPlan, history: Sequence[dict[str, str]] | None) -> None:
        """Attach host history when the message-stage plan had none.

        AstrBot's message listener runs before the LLM request, so it cannot
        see ``req.contexts``.  On the first turn after a restart the adapter
        calls this to backfill the recent window without recomputing L0/L1.
        """
        if not history:
            return
        self._import_history(plan.user_key, history)
        rows = self.db.recent_turns(plan.user_key, max(0, self.cfg.l3.recent_turns) * 2)
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        if self.cfg.l3.cache_aware_pruning:
            messages = self.l3._trim_by_chars(messages, self.cfg.l3.recent_max_chars)
        plan.recent_messages = messages
        plan.metadata["recent_chars"] = sum(len(m["content"]) for m in messages)

    def refresh_prefix(self, plan: ContextPlan, host_system_prompt: str) -> None:
        """Rebind the stable prefix to the host system prompt.

        Adapters often only have the host prompt at the LLM-request hook, while
        a short-circuit decision has to happen one stage earlier.  Rebuilding
        the prefix here never changes the L0 block itself, only the host part.
        """
        import hashlib

        l0_in_prefix = plan.l0_block if not self.cfg.l0.tail_injection else ""
        plan.stable_prefix = self.l3.build_prefix(host_system_prompt, self.cfg.persona, l0_in_prefix)
        plan.prefix_hash = hashlib.sha256(plan.stable_prefix.encode("utf-8")).hexdigest()

    def _baseline_plan(self, plan: ContextPlan) -> ContextPlan:
        base = copy.copy(plan)
        base.stable_prefix = self.l3.build_prefix(self.cfg.system_prompt, self.cfg.persona, "")
        base.prompt = plan.compressed_text
        base.ephemeral_tail = ""
        base.recall = ""
        base.wake = False
        return base

    # ---------------------------------------------------------------- helpers
    def _import_history(self, user_key: str, history: Sequence[dict[str, str]] | None) -> None:
        if not history:
            return
        if self.db.get_state(user_key, "history_imported", False):
            return
        count = int(self.db.scalar("SELECT COUNT(*) FROM turns WHERE user_key=?", (user_key,), 0))
        if count > 0:
            self.db.set_state(user_key, "history_imported", True)
            return
        for item in list(history)[-32:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                self.db.append_turn(user_key, 0, role, content, self.clock())
        self.db.set_state(user_key, "history_imported", True)

    def note_user_activity(self, user_key: str) -> None:
        try:
            self.db.set_state(user_key, "last_user_at", self.clock())
        except Exception:  # noqa: BLE001
            pass

    def stats(self, user_key: str | None = None) -> dict[str, Any]:
        report = self.metrics.report(user_key)
        report["solidified"] = self.shortcircuit.stats(user_key) if user_key else {}
        report["memory_pools"] = self.facts.pool_counts(user_key)
        report["proactive"] = self.proactive.stats(user_key, limit=5)
        report["db"] = {
            "users": int(self.db.scalar("SELECT COUNT(DISTINCT user_key) FROM hot_slots", (), 0)),
            "hot_slots": int(self.db.scalar("SELECT COUNT(*) FROM hot_slots WHERE content_seed<>''", (), 0)),
            "warm_slots": int(self.db.scalar("SELECT COUNT(*) FROM warm_slots", (), 0)),
            "cold_seeds": int(self.db.scalar("SELECT COUNT(*) FROM cold_seeds", (), 0)),
            "open_loops": int(self.db.scalar("SELECT COUNT(*) FROM open_loops WHERE status='open'", (), 0)),
            "llm_calls": self.ledger.snapshot(),
        }
        return report

    def record_error(self, where: str, message: str) -> None:
        """Remember the last failure so the page can show a health warning."""
        try:
            self.db.incr_state("__global__", "error_count", 1)
            self.db.set_state("__global__", "last_error", f"{where}: {message}"[:300])
            self.db.set_state("__global__", "last_error_at", self.clock())
        except Exception:
            pass

def _plan_tokens(plan: ContextPlan) -> int:
    from .layers.l3 import estimate_tokens

    total = estimate_tokens(plan.stable_prefix) + estimate_tokens(plan.prompt)
    for m in plan.recent_messages:
        total += estimate_tokens(m["content"]) + 4
    return total


def _match_dict(match: Any) -> dict[str, Any]:
    if match is None:
        return {}
    return {
        "signature": match.signature.key,
        "intent": match.signature.intent,
        "created": match.created,
        "repeated": match.repeated,
        "hit": match.hit,
        "confidence": match.path.confidence,
        "neg_streak": match.path.neg_streak,
        "hits": match.path.hits,
    }

