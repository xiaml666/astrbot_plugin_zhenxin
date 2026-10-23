"""AstrBot adapter (primary target).

Integration points used (all documented AstrBot plugin APIs):

* ``@filter.event_message_type(EventMessageType.ALL)`` - one high-priority
  listener.  On a solidified hit it sends the shaped reply and calls
  ``event.stop_event()``, so the host never invokes the LLM.
* ``@filter.on_llm_request()`` - rewrites ``req.system_prompt`` (stable prefix),
  ``req.contexts`` (recent window) and appends the ephemeral hints through
  ``req.extra_user_content_parts`` as temporary content.
* ``@filter.on_llm_response()`` - feeds the real reply into the zero-LLM write
  path and records metrics.

The module imports safely without AstrBot installed; the root ``main.py``
exports the actual ``Star`` subclass.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from ..config import Config
from ..engine import ZhenxinEngine
from ..models import ContextPlan

log = logging.getLogger(__name__)
PLAN_KEY = "zhenxin_plan"
HOST_KEY = "zhenxin_host_prompt"
MARKERS = ("\n\n【人格锚点】", "\n\n【长期线索·L0】")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def config_from_astrbot(config: Any = None) -> Config:
    path = os.environ.get("ZHENXIN_CONFIG")
    cfg = Config.load(path) if path else Config()
    if config:
        data = cfg.to_dict()
        try:
            deep_merge(data, dict(config))
        except (TypeError, ValueError):
            log.warning("AstrBot config is not mapping-like; using defaults")
        else:
            cfg = Config.from_dict(data)
    cfg.apply_env()
    return cfg


class AstrBotAdapter:
    def __init__(self, engine: ZhenxinEngine) -> None:
        self.engine = engine
        self._fallback_plans: dict[int, ContextPlan] = {}
        self._host_prompts: dict[str, str] = {}

    def shadow_for(self, user_key: str) -> bool:
        cfg = self.engine.cfg
        if not cfg.shadow_mode:
            return False
        return not any(prefix and prefix in user_key for prefix in cfg.live_allowlist)

    # ------------------------------------------------------------- event glue
    @staticmethod
    def text(event: Any) -> str:
        value = getattr(event, "message_str", None)
        if value:
            return str(value)
        obj = getattr(event, "message_obj", None)
        return str(getattr(obj, "message_str", "") or "")

    @staticmethod
    def user_key(event: Any) -> str:
        for attr in ("unified_msg_origin", "session_id"):
            value = getattr(event, attr, None)
            if value:
                return str(value)
        getter = getattr(event, "get_session_id", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                pass
        obj = getattr(event, "message_obj", None)
        for attr in ("session_id", "group_id"):
            value = getattr(obj, attr, None)
            if value:
                return str(value)
        return "default"

    @staticmethod
    def is_command(event: Any) -> bool:
        checker = getattr(event, "is_command", None)
        if callable(checker):
            try:
                if checker():
                    return True
            except Exception:
                pass
        text = AstrBotAdapter.text(event).lstrip()
        return text.startswith(("/", "／", "#"))

    def get_plan(self, event: Any) -> ContextPlan | None:
        getter = getattr(event, "get_extra", None)
        if callable(getter):
            try:
                plan = getter(PLAN_KEY)
                if plan is not None:
                    return plan
            except Exception:
                pass
        return self._fallback_plans.get(id(event))

    def set_plan(self, event: Any, plan: ContextPlan) -> None:
        setter = getattr(event, "set_extra", None)
        if callable(setter):
            try:
                setter(PLAN_KEY, plan)
                return
            except Exception:
                pass
        self._fallback_plans[id(event)] = plan

    # --------------------------------------------------------- message stage
    async def intercept(self, event: Any) -> AsyncIterator[Any]:
        """High-priority message listener.  Yields only on a solid hit."""
        text = self.text(event).strip()
        if not text or self.is_command(event):
            return
        user_key = self.user_key(event)
        try:
            plan = self.engine.begin_turn(
                user_key,
                text,
                host_system_prompt=self._host_prompt(event),
                shadow=self.shadow_for(user_key),
            )
        except Exception:
            log.exception("zhenxin begin_turn failed")
            self.engine.record_error("begin_turn", str(sys.exc_info()[1]))
            return
        self.set_plan(event, plan)
        if plan.shortcircuit and not plan.shadow:
            reply = plan.solid_reply or ""
            try:
                self.engine.complete_turn(plan, reply, main_llm_calls=0)
            except Exception:
                log.exception("zhenxin solidified complete_turn failed")
                self.engine.record_error("solidified_turn", str(sys.exc_info()[1]))
            # Documented order: yield the result, then stop propagation.
            yield event.plain_result(reply)
            stop = getattr(event, "stop_event", None)
            if callable(stop):
                stop()

    # ------------------------------------------------------------ LLM stage
    async def on_llm_request(self, event: Any, req: Any) -> None:
        plan = self.get_plan(event)
        if plan is None:
            try:
                text = str(getattr(req, "prompt", "") or self.text(event))
                plan = self.engine.begin_turn(
                    self.user_key(event),
                    text,
                    host_system_prompt=self._base_system(req),
                    history=self._contexts(req),
                    shadow=self.shadow_for(self.user_key(event)),
                )
                self.set_plan(event, plan)
            except Exception:
                log.exception("zhenxin on_llm_request begin_turn failed")
                self.engine.record_error("llm_request", str(sys.exc_info()[1]))
                return
        if plan.shadow or plan.completed:
            return
        applied = plan.metadata.setdefault("applied_req_ids", [])
        if id(req) in applied:
            return

        contexts = self._contexts(req)
        if not plan.recent_messages and contexts:
            self.engine.load_history(plan, contexts)

        base = self._base_system(req)
        cached_host = self._host_prompt(event)
        if base and base != cached_host:
            self.engine.refresh_prefix(plan, base)
        self._store_host_prompt(event, base)

        try:
            req.system_prompt = plan.stable_prefix
        except Exception:
            log.warning("could not set req.system_prompt")
        try:
            mode = self.engine.cfg.l3.host_context_mode
            if mode == "replace":
                req.contexts = plan.recent_messages
            elif mode == "prepend":
                req.contexts = list(contexts) + list(plan.recent_messages)
            # off: deliberately leave the host's own contexts untouched.
        except Exception:
            log.warning("could not set req.contexts")
        try:
            req.prompt = plan.compressed_text
        except Exception:
            log.warning("could not set req.prompt")
        self._append_ephemeral(req, plan)
        applied.append(id(req))

    async def on_llm_response(self, event: Any, resp: Any) -> None:
        plan = self.get_plan(event)
        if plan is None or plan.completed:
            return
        reply = (
            getattr(resp, "completion_text", None)
            or getattr(resp, "text", None)
            or getattr(resp, "content", None)
            or ""
        )
        try:
            self.engine.complete_turn(
                plan,
                str(reply),
                main_llm_calls=1,
                actual_usage=self._usage(resp),
            )
        except Exception:
            log.exception("zhenxin on_llm_response complete_turn failed")
            self.engine.record_error("llm_response", str(sys.exc_info()[1]))
        finally:
            self._fallback_plans.pop(id(event), None)

    @staticmethod
    def _usage(resp: Any) -> dict[str, int] | None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        out: dict[str, int] = {}
        for name in ("input_other", "input_cached", "output"):
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is not None:
                try:
                    out[name] = int(value)
                except (TypeError, ValueError):
                    pass
        return out or None

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _contexts(req: Any) -> list[dict[str, str]]:
        contexts = getattr(req, "contexts", None) or []
        out: list[dict[str, str]] = []
        for item in contexts:
            if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                out.append({"role": str(item["role"]), "content": str(item["content"])})
        return out

    @staticmethod
    def _base_system(req: Any) -> str:
        value = str(getattr(req, "system_prompt", "") or "")
        for marker in MARKERS:
            idx = value.find(marker)
            if idx >= 0:
                value = value[:idx]
        return value.strip()

    def _host_prompt(self, event: Any) -> str:
        getter = getattr(event, "get_extra", None)
        if callable(getter):
            try:
                value = getter(HOST_KEY)
                if value:
                    return str(value)
            except Exception:
                pass
        return self._host_prompts.get(self.user_key(event), "")

    def _store_host_prompt(self, event: Any, base: str) -> None:
        if not base:
            return
        self._host_prompts[self.user_key(event)] = base
        setter = getattr(event, "set_extra", None)
        if callable(setter):
            try:
                setter(HOST_KEY, base)
            except Exception:
                pass

    @staticmethod
    def _append_ephemeral(req: Any, plan: ContextPlan) -> None:
        if not plan.ephemeral_tail:
            return
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is None:
            parts = []
            try:
                req.extra_user_content_parts = parts
            except Exception:
                return
        try:
            from astrbot.core.agent.message import TextPart  # type: ignore

            part = TextPart(text=plan.ephemeral_tail)
            marker = getattr(part, "mark_as_temp", None)
            if callable(marker):
                marked = marker()
                if marked is not None:
                    part = marked
            parts.append(part)
        except Exception:
            # Older AstrBot versions may not have TextPart.  Appending a bare
            # string to extra_user_content_parts can break the provider, so
            # fall back to the current prompt instead.
            try:
                req.prompt = str(getattr(req, "prompt", "") or "") + "\n\n" + plan.ephemeral_tail
            except Exception:
                log.debug("could not append ephemeral content", exc_info=True)


def make_engine(config: Any = None, data_dir: str | Path | None = None) -> ZhenxinEngine:
    cfg = config_from_astrbot(config)
    # AstrBot's working directory is not guaranteed.  If the user kept the
    # default relative path, put the DB in the plugin data directory instead.
    if data_dir and cfg.db_path == Config().db_path:
        cfg.db_path = str(Path(data_dir) / "zhenxin.sqlite3")
    return ZhenxinEngine(cfg)
