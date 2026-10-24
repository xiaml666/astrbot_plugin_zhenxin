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
        try:
            self._background = asyncio.get_running_loop().create_task(self._background_loop())
        except RuntimeError:
            # No running loop (standalone import / unit test): the page still
            # works, only scheduled proactive/style refreshes are disabled.
            self._background = None

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
        yield event.plain_result(
            "真心状态 v" + VERSION + "\n"
            f"轮次：{stats.get('rounds', 0)}  L1 唤醒：{stats.get('wake_rate', 0):.1%}\n"
            f"前缀末20稳定：{stats.get('prefix_stable_last20', 0):.1%}  L0：{stats.get('l0_chars_max', 0)} 字\n"
            f"记忆：周期 {pools.get('cycle', 0)} / 珍藏 {pools.get('treasure', 0)}"
            f" / 浅忆 {pools.get('shallow', 0)} / 回收 {pools.get('trash', 0)}\n"
            f"风格：{self.engine.style.describe(user_key)}\n"
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
        await self.adapter.on_llm_response(event, resp)

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
        for user_key in self.engine.style_learning.due_users(limit=3):
            online = bool(cfg.online) and self.engine.style_learning.online_due(user_key)
            await asyncio.to_thread(self.engine.style_learning.refresh, user_key, online=online)
            await asyncio.sleep(0)

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

        This is the only extra model call in the whole plugin and it only runs
        when the user explicitly enables ``proactive.use_llm``.  Any failure
        falls back to the local draft.
        """
        try:
            getter = getattr(self.context, "get_using_provider_async", None)
            if not callable(getter):
                return ""
            provider = await getter(user_key)
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
            from astrbot.api.all import MessageChain  # type: ignore
            from astrbot.api.message_components import Plain  # type: ignore

            return bool(await self.context.send_message(user_key, MessageChain([Plain(text)])))
        except Exception:
            try:
                from astrbot.api import logger  # type: ignore

                logger.exception("zhenxin proactive send failed")
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
