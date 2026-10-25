"""AstrBot entry point for the Zhenxin plugin.

AstrBot adds the plugin directory to its import path and imports ``main.py``.
The core package lives under ``src/`` so the same repository also works as a
standalone Python library; this shim makes the plugin work without a pip
install.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from astrbot.api.event import AstrMessageEvent, filter  # type: ignore  # noqa: E402
from astrbot.api.star import Context, Star, register  # type: ignore  # noqa: E402

from zhenxin._version import __version__ as VERSION  # noqa: E402
from zhenxin.adapters.astrbot import AstrBotAdapter, make_engine  # noqa: E402
from zhenxin.page_api import PluginPageAPI  # noqa: E402


@register(
    "astrbot_plugin_zhenxin",
    "zhenxin contributors",
    "真心：五层上下文与缓存感知修剪，默认不检索、写入零 LLM 调用。",
    VERSION,
)
class ZhenxinPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        self.engine = make_engine(config, data_dir=self._data_dir())
        self.adapter = AstrBotAdapter(self.engine)
        self.page_api = PluginPageAPI(self, self.engine, config)
        self.page_api.register()
        self._background: asyncio.Task | None = None
        self._ensure_background()

    def _ensure_background(self) -> None:
        """Start the 60s loop, from __init__ or the loaded hook.

        ``Star.__init__`` is not guaranteed to run inside a running event loop
        (AstrBot versions differ), so the loaded hook retries.  Without this,
        the "real-time" scans and periodic learning would never run.
        """
        task = getattr(self, "_background", None)
        if task is not None and not task.done():
            return
        try:
            self._background = asyncio.get_running_loop().create_task(self._background_loop())
        except RuntimeError:
            self._background = None

    @filter.on_astrbot_loaded()
    async def zhenxin_on_loaded(self, event=None):
        self._ensure_background()
        # Record the current enabled-plugin set without alerting; from now on
        # only a *newly* enabled conflict triggers the page/AI reminder.
        try:
            await asyncio.to_thread(self.page_api.poll_conflicts)
        except Exception:
            pass

    @staticmethod
    def _data_dir():
        try:
            from astrbot.api.star import StarTools  # type: ignore

            return StarTools.get_data_dir()
        except Exception:
            return None

    @filter.command("zhenxin")
    async def zhenxin_status(self, event: AstrMessageEvent):
        """Lightweight local status command; never calls the LLM."""
        user_key = self.adapter.user_key(event)
        stats = self.engine.stats(user_key)
        currency = stats.get("actual_currency", "CNY")
        pools = stats.get("memory_pools", {})
        preference = self.engine.style_learning.status(user_key)
        yield event.plain_result(
            "真心状态 v" + VERSION + "\n"
            f"轮次：{stats.get('rounds', 0)}  L1 唤醒：{stats.get('wake_rate', 0):.1%}\n"
            f"前缀末20稳定：{stats.get('prefix_stable_last20', 0):.1%}  L0：{stats.get('l0_chars_max', 0)} 字\n"
            f"记忆：周期 {pools.get('cycle', 0)} / 珍藏 {pools.get('treasure', 0)}"
            f" / 浅忆 {pools.get('shallow', 0)} / 回收 {pools.get('trash', 0)}\n"
            f"说话偏好：{int(preference.get('chars', 0) or 0)} / {int(preference.get('max_chars', 0) or 0)} 字\n"
            f"真实 usage：{stats.get('actual_usage_rounds', 0)}/{stats.get('rounds', 0)} 轮  "
            f"{currency} {stats.get('actual_cost', 0):.4f}"
        )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def zhenxin_on_message(self, event: AstrMessageEvent):
        async for item in self.adapter.intercept(event):
            yield item

    @filter.on_llm_request(priority=-100)
    async def zhenxin_on_llm_request(self, event: AstrMessageEvent, req):
        await self.adapter.on_llm_request(event, req)

    @filter.on_llm_response(priority=100)
    async def zhenxin_on_llm_response(self, event: AstrMessageEvent, resp):
        try:
            setattr(event, "zhenxin_is_llm_reply", True)
        except Exception:
            pass
        await self.adapter.on_llm_response(event, resp)

    @filter.on_decorating_result(priority=-100_000_000_000_000_000)
    async def zhenxin_on_decorating_result(self, event: AstrMessageEvent):
        """Optional reply segmentation ("对话分段").

        AstrBot sends the last component list itself; we send the earlier
        bubbles here and leave the final one in ``result.chain``.  Rules:
        transport only, no model call; only plain-text LLM replies; images,
        voice, @ and reply-quotes are never touched.
        """
        cfg = self.engine.cfg.split
        if not cfg.enabled:
            return
        user_key = self.adapter.user_key(event)
        if self.adapter.shadow_for(user_key):
            return
        try:
            result = event.get_result()
        except Exception:
            return
        if result is None or not getattr(result, "chain", None):
            return
        if getattr(result, "zhenxin_split_done", False):
            return
        checker = getattr(result, "is_model_result", None)
        is_model = False
        if callable(checker):
            try:
                is_model = bool(checker())
            except Exception:
                is_model = False
        if not is_model and not bool(getattr(event, "zhenxin_is_llm_reply", False)):
            return
        try:
            from astrbot.api.message_components import Plain  # type: ignore
        except Exception:
            return
        chain = list(result.chain)
        if not chain:
            return
        # We send the first bubbles ourselves and rewrite the chain for the
        # final one.  If the chain cannot be rewritten, doing nothing is safer
        # than sending the first bubble twice.
        if not hasattr(result.chain, "clear") or not hasattr(result.chain, "extend"):
            return
        setattr(result, "zhenxin_split_done", True)

        from zhenxin._aux.segments import (
            calculate_delay,
            merge_segments,
            segment_text,
            split_chain,
        )

        try:
            segments = split_chain(
                chain,
                plain_factory=Plain,
                max_segments=int(cfg.max_segments),
                min_segment_chars=int(cfg.min_segment_chars),
            )
        except Exception:  # noqa: BLE001 - a broken chain must not break the reply
            return
        segments = [segment for segment in segments if segment]
        if len(segments) <= 1:
            return
        failed_from: int | None = None
        for index, segment in enumerate(segments[:-1]):
            try:
                if not await self._send_chain(user_key, segment):
                    raise RuntimeError("send_message returned false")
            except Exception:
                # Never drop text/media: merge this and every later bubble into
                # the final one, which the framework will still send.
                failed_from = index
                break
            delay = calculate_delay(str(cfg.delay_mode), segment_text(segments[index + 1]))
            if delay:
                await asyncio.sleep(delay)
        if failed_from is not None:
            final = merge_segments(segments[failed_from:], Plain)
        else:
            final = segments[-1]
        try:
            result.chain.clear()
            result.chain.extend(final)
        except Exception:
            # If the result object cannot be rewritten, leave the original
            # chain untouched: a duplicated first bubble is worse.
            return

    async def terminate(self):
        task = getattr(self, "_background", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self.engine.close()
        except Exception:
            pass

    # ------------------------------------------------------------ background
    async def _background_loop(self) -> None:
        """One low-frequency loop for both opt-in background features.

        A 60-second tick is cheap (a few indexed reads) and means a restart
        never sends a burst of overdue messages.
        """
        while True:
            try:
                await asyncio.sleep(60)
                await self._tick_style()
                await self._tick_proactive()
                await self._tick_conflicts()
            except asyncio.CancelledError:
                raise
            except Exception:
                try:
                    from astrbot.api import logger  # type: ignore

                    logger.exception("zhenxin background tick failed")
                except Exception:
                    pass

    async def _tick_style(self) -> None:
        cfg = self.engine.cfg.style_learning
        if not cfg.enabled:
            return
        from zhenxin._aux.style_learning import join_note

        learner = self.engine.style_learning
        for user_key in learner.due_users(limit=3):
            online = bool(cfg.online) and learner.online_due(user_key)
            try:
                gathered = await asyncio.to_thread(learner.gather, user_key, online=online)
                error = str(gathered.get("error") or "")
                pref: list[str] | None = None
                web: list[str] | None = None
                # Re-screen only when there is a *new* preference sentence;
                # otherwise the 24h local tick would call the model for nothing.
                if gathered.get("pref_new"):
                    pref = await self.screen_knowledge("pref", gathered["pref"], user_key)
                    if pref is None:
                        error = join_note(error, "偏好甄别失败，保留上一次结果")
                if gathered.get("web"):
                    web = await self.screen_knowledge("web", gathered["web"], user_key)
                    if web is None:
                        error = join_note(error, "网络知识甄别失败，保留上一次结果")
                await asyncio.to_thread(
                    learner.apply,
                    user_key,
                    pref=pref,
                    web=web,
                    error=error,
                    online=bool(gathered.get("online")),
                )
            except Exception:
                try:
                    from astrbot.api import logger  # type: ignore

                    logger.exception("zhenxin style learning tick failed")
                except Exception:
                    pass
            await asyncio.sleep(0)

    async def _host_provider(self, user_key: str = ""):
        """The provider AstrBot is currently using, or ``None``."""
        getter = getattr(self.context, "get_using_provider_async", None)
        if not callable(getter):
            return None
        try:
            return await getter(user_key)
        except TypeError:
            try:
                return await getter()
            except Exception:  # noqa: BLE001
                return None
        except Exception:  # noqa: BLE001
            return None

    async def screen_knowledge(
        self, kind: str, items: list[str], user_key: str = ""
    ) -> list[str] | None:
        """Screen style-knowledge candidates; ``None`` means "keep old".

        The host provider is preferred (works with AstrBot's own model config
        even when ``main_model.enabled`` is false); the engine gateway is the
        standalone fallback.  No model at all → nothing is written.
        """
        from zhenxin._aux.style_learning import SCREEN_SYSTEM, parse_screened, screen_prompt

        if not items:
            return []
        prompt = screen_prompt(kind, items)
        provider = await self._host_provider(user_key)
        if provider is not None:
            try:
                response = await provider.text_chat(prompt=prompt, system_prompt=SCREEN_SYSTEM)
                raw = str(getattr(response, "completion_text", "") or "")
                parsed = parse_screened(raw)
                if parsed is not None:
                    return parsed
            except Exception:  # noqa: BLE001 - fall back to the gateway
                pass
        try:
            return await asyncio.to_thread(self.engine.screen_knowledge, kind, items)
        except Exception:  # noqa: BLE001
            return None

    async def _tick_conflicts(self) -> None:
        poll = getattr(self.page_api, "poll_conflicts", None)
        if callable(poll):
            await asyncio.to_thread(poll)

    async def _tick_proactive(self) -> None:
        cfg = self.engine.cfg.proactive
        if not cfg.enabled:
            return
        now = self.engine.clock()
        for item in self.engine.proactive.candidates(now=now):
            user_key = item["user"]
            ok, _reason = self.engine.proactive.gate(user_key, item["last_user_at"], now)
            if not ok:
                continue
            text, kind = self.engine.proactive.compose(user_key, now)
            if cfg.use_llm:
                text = await self._polish_proactive(user_key, text) or text
            if not text:
                continue
            sent = await self._send_text(user_key, text)
            if sent:
                self.engine.proactive.mark_sent(user_key, text, kind, now)
            await asyncio.sleep(1)

    async def _polish_proactive(self, user_key: str, draft: str) -> str:
        """Optional: let the host model rewrite one short line.

        Only runs when the user explicitly enables ``proactive.use_llm``.  Any
        failure falls back to the local draft.  (The other optional model call
        in the plugin is the low-frequency style-knowledge screen.)
        """
        try:
            provider = await self._host_provider(user_key)
            if provider is None:
                return ""
            address = self.engine.facts.address(user_key)
            prompt = (
                "你在用微信主动给一个熟人发一条很短的消息。"
                + (f"对方希望你称呼TA为{address}。" if address else "")
                + f"素材：{draft}\n"
                + f"要求：像真人随手发的一句，不超过{self.engine.cfg.proactive.max_chars}字，"
                "不要解释，不要客服腔，不要问太多。只输出这句话。"
            )
            response = await provider.text_chat(prompt=prompt, system_prompt="你很自然地发一条短消息。")
            text = str(getattr(response, "completion_text", "") or "").strip()
            if 1 <= len(text) <= self.engine.cfg.proactive.max_chars * 2:
                return text[: self.engine.cfg.proactive.max_chars]
        except Exception:
            return ""
        return ""

    async def _send_text(self, user_key: str, text: str) -> bool:
        try:
            from astrbot.api.message_components import Plain  # type: ignore

            return await self._send_chain(user_key, [Plain(text)])
        except Exception:
            try:
                from astrbot.api import logger  # type: ignore

                logger.exception("zhenxin proactive send failed")
            except Exception:
                pass
            return False

    async def _send_chain(self, user_key: str, components: list[Any]) -> bool:
        """Send one message chain (used by segmentation).

        Direct ``context.send_message`` bypasses ``on_decorating_result``, so
        the earlier bubbles are not split again.  Component order is kept
        (Reply/images/At included).
        """
        if not components:
            return False
        try:
            from astrbot.api.event import MessageChain  # type: ignore

            chain = MessageChain()
            chain.chain = list(components)
            return bool(await self.context.send_message(user_key, chain))
        except Exception:
            try:
                from astrbot.api import logger  # type: ignore

                logger.exception("zhenxin segmented send failed")
            except Exception:
                pass
            return False

    # ------------------------------------------------------------ LLM tools
    # The only two things the model may call explicitly.  Both stay local:
    # no network, no embedding API, no extra model round.
    @filter.llm_tool(name="zhenxin_recall")
    async def zhenxin_recall(self, event: AstrMessageEvent, query: str):
        """从真心的长期记忆里查找与 query 相关的旧事。

        只在用户明确问起过去（“你还记得…”“上次说的…”）或需要确认偏好、关系时调用；
        普通闲聊不要调用。

        Args:
            query(string): 要查找的关键词或问题
        """
        try:
            user_key = self.adapter.user_key(event)
            facts = self.engine.facts.recall(user_key, str(query), k=3)
        except Exception as exc:  # noqa: BLE001
            self.engine.record_error("tool_recall", str(exc))
            return "读取记忆失败，先不要编造。"
        if not facts:
            return "没有找到相关记忆。"
        return "相关记忆：\n" + "\n".join(f"- {text}" for text in facts)

    @filter.llm_tool(name="zhenxin_remember")
    async def zhenxin_remember(self, event: AstrMessageEvent, text: str):
        """把用户明确要求记住的一句话写进真心长期记忆。

        只有用户说“记住…”“帮我记一下…”时才调用，不要主动把普通闲聊写进去。

        Args:
            text(string): 要记住的简短事实，尽量保留用户原话
        """
        fact = str(text or "").strip()
        if not fact:
            return "没有可记住的内容。"
        try:
            user_key = self.adapter.user_key(event)
            added = self.engine.facts.add(user_key, fact, "fact", importance=0.9)
        except Exception as exc:  # noqa: BLE001
            self.engine.record_error("tool_remember", str(exc))
            return "写入记忆失败。"
        return "已记住。" if added else "这条已经记过了。"
