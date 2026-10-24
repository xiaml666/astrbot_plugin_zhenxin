"""AstrBot entry point for the Zhenxin plugin.

AstrBot adds the plugin directory to its import path and imports ``main.py``.
The core package lives under ``src/`` so the same repository also works as a
standalone Python library; this shim makes the plugin work without a pip
install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from astrbot.api.event import AstrMessageEvent, filter  # type: ignore  # noqa: E402
from astrbot.api.star import Context, Star, register  # type: ignore  # noqa: E402

from zhenxin.adapters.astrbot import AstrBotAdapter, make_engine  # noqa: E402
from zhenxin.page_api import PluginPageAPI  # noqa: E402


@register(
    "astrbot_plugin_zhenxin",
    "zhenxin contributors",
    "真心：五层上下文与缓存感知修剪，默认不检索、写入零 LLM 调用。",
    "0.3.2",
)
class ZhenxinPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        self.engine = make_engine(config, data_dir=self._data_dir())
        self.adapter = AstrBotAdapter(self.engine)
        self.page_api = PluginPageAPI(self, self.engine, config)
        self.page_api.register()

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
        stats = self.engine.stats(self.adapter.user_key(event))
        currency = stats.get("actual_currency", "CNY")
        yield event.plain_result(
            "真心状态\n"
            f"轮次：{stats.get('rounds', 0)}  L1 唤醒：{stats.get('wake_rate', 0):.1%}\n"
            f"前缀末20稳定：{stats.get('prefix_stable_last20', 0):.1%}  L0：{stats.get('l0_chars_max', 0)} 字\n"
            f"固化命中：{stats.get('solid_hits', 0)}  L2 LLM：{stats.get('l2_llm_calls', 0)}\n"
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
        try:
            self.engine.close()
        except Exception:
            pass

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
            facts = self.engine.facts.recall(user_key, str(query), k=3, min_score=0.12)
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





