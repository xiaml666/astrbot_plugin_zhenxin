"""Thin NoneBot2 adapter.

NoneBot2 has no built-in LLM pipeline, so this adapter exposes the same
two-phase contract instead of assuming one:

    plan = adapter.begin(event)
    if plan.shortcircuit: ...      # send template, complete, stop matcher
    reply = await my_llm(plan)     # host's own model call
    adapter.finish(plan, reply)

The module imports safely when NoneBot2 is not installed.
"""

from __future__ import annotations

from typing import Any

from ..engine import ZhenxinEngine
from ..models import ContextPlan


class NoneBotAdapter:
    def __init__(self, engine: ZhenxinEngine) -> None:
        self.engine = engine

    @staticmethod
    def user_key(event: Any) -> str:
        getter = getattr(event, "get_session_id", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                pass
        getter = getattr(event, "get_user_id", None)
        if callable(getter):
            try:
                return "private:" + str(getter())
            except Exception:
                pass
        return "default"

    @staticmethod
    def text(event: Any) -> str:
        getter = getattr(event, "get_plaintext", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
        return str(getattr(event, "raw_message", "") or "")

    def begin(self, event: Any) -> ContextPlan:
        plan = self.engine.begin_turn(self.user_key(event), self.text(event))
        try:
            event.state["zhenxin_plan"] = plan
        except Exception:
            pass
        return plan

    def get_plan(self, event: Any) -> ContextPlan | None:
        try:
            return event.state.get("zhenxin_plan")
        except Exception:
            return None

    @staticmethod
    def messages(plan: ContextPlan) -> list[dict[str, str]]:
        """OpenAI-style messages for a host that calls its own model."""
        return [{"role": "system", "content": plan.stable_prefix}, *plan.messages]

    def finish(self, plan: ContextPlan, reply: str, *, main_llm_calls: int = 1):
        return self.engine.complete_turn(plan, reply, main_llm_calls=main_llm_calls)

    def finish_shortcircuit(self, plan: ContextPlan) -> str:
        reply = plan.solid_reply or ""
        self.engine.complete_turn(plan, reply, main_llm_calls=0)
        return reply


def example_nonebot_matcher(adapter: NoneBotAdapter):
    """Return a ready-to-register matcher when NoneBot2 is installed.

    Usage::

        from zhenxin.adapters.nonebot2 import example_nonebot_matcher
        matcher = example_nonebot_matcher(adapter)

        @matcher.handle()
        async def _(event: MessageEvent):
            plan = adapter.get_plan(event)
            if plan.shortcircuit:
                await matcher.finish(adapter.finish_shortcircuit(plan))
    """
    try:
        from nonebot import on_message  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("NoneBot2 is not installed") from exc
    matcher = on_message(priority=1, block=False)
    return matcher
