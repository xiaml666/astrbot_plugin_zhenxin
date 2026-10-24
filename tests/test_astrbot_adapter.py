from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers import Config, make_engine
from zhenxin.adapters.astrbot import AstrBotAdapter


class MakeEnginePathsTests(unittest.TestCase):
    def test_default_metrics_jsonl_lives_in_the_plugin_data_dir(self):
        import tempfile
        from zhenxin.adapters.astrbot import make_engine

        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine({}, data_dir=tmp)
            self.assertEqual(str(Path(tmp) / "metrics.jsonl"), engine.cfg.metrics.jsonl_path)
            self.assertEqual(str(Path(tmp) / "zhenxin.sqlite3"), engine.cfg.db_path)
            engine.close()


class FakeEvent:
    def __init__(self, text: str, key: str = "u1") -> None:
        self.message_str = text
        self.unified_msg_origin = key
        self.extras = {}
        self.stopped = False
        self.results = []

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key):
        return self.extras.get(key)

    def plain_result(self, text):
        return ("plain", text)

    def stop_event(self):
        self.stopped = True


class FakeReq:
    def __init__(self) -> None:
        self.system_prompt = "HOST SYSTEM PROMPT"
        self.contexts = []
        self.prompt = "hello"
        self.extra_user_content_parts = []


class FakeUsage:
    def __init__(self) -> None:
        self.input_other = 100
        self.input_cached = 20
        self.output = 8


class FakeResp:
    def __init__(self, text: str, usage=None) -> None:
        self.completion_text = text
        self.usage = usage


class AstrBotAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_injection_and_solidified_stop(self):
        engine, clock = make_engine()
        adapter = AstrBotAdapter(engine)
        for i, text in enumerate(["今天好累", "海盐那本还没画完"], 1):
            clock.advance()
            event = FakeEvent(text)
            async for _ in adapter.intercept(event):
                self.fail("non-hit turns must not yield")
            req = FakeReq()
            req.prompt = text
            req.system_prompt = "HOST SYSTEM PROMPT\n\nOTHER PLUGIN MEMORY"
            if i == 1:
                req.contexts = [{"role": "user", "content": "历史消息"}]
            await adapter.on_llm_request(event, req)
            # Cache-safe layout: the clue block must NOT live in the system
            # prompt (a prefix cache would be invalidated from there on).
            self.assertNotIn("【长期线索·L0】", req.system_prompt)
            self.assertIn("HOST SYSTEM PROMPT", req.system_prompt)
            self.assertIn("OTHER PLUGIN MEMORY", req.system_prompt)
            self.assertIsInstance(req.contexts, list)
            if i == 1:
                self.assertIn("历史消息", [m.get("content") for m in req.contexts])
            plan = event.get_extra("zhenxin_plan")
            tail = plan.ephemeral_tail
            self.assertTrue(req.extra_user_content_parts or tail in req.prompt)
            if plan.l0_block:
                self.assertIn("【长期线索·L0】", tail)
            parts_before = list(req.extra_user_content_parts)
            prompt_before = req.prompt
            await adapter.on_llm_request(event, req)
            self.assertEqual(req.extra_user_content_parts, parts_before)
            self.assertEqual(req.prompt, prompt_before)
            await adapter.on_llm_response(event, FakeResp("回复", FakeUsage()))
            last = engine.metrics.recent("u1", 1)[0]
            self.assertEqual(last["input_tokens_actual"], 100)
            self.assertEqual(last["cached_tokens_actual"], 20)
            self.assertEqual(last["output_tokens_actual"], 8)
            self.assertEqual(last["actual_usage_available"], 1)

        clock.advance()
        event = FakeEvent("今天好累")
        out = [item async for item in adapter.intercept(event)]
        self.assertTrue(event.stopped)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "plain")
        self.assertTrue(out[0][1])
        trace = engine.metrics.recent("u1", 1)[0]
        self.assertEqual(trace["shortcircuit_hit"], 1)
        self.assertEqual(trace["main_llm_calls"], 0)

    async def test_canary_allowlist_bypasses_shadow_only_for_matching_session(self):
        engine, _ = make_engine(**{"shadow_mode": True, "live_allowlist": ["webchat:test"]})
        adapter = AstrBotAdapter(engine)
        self.assertFalse(adapter.shadow_for("webchat:test:1"))
        self.assertTrue(adapter.shadow_for("snowluma:FriendMessage:123"))
        self.assertTrue(adapter.shadow_for("webchat:prod:1"))

    async def test_host_context_mode_off_preserves_other_plugins_contexts(self):
        engine, clock = make_engine(**{"l3.host_context_mode": "off"})
        adapter = AstrBotAdapter(engine)
        clock.advance()
        event = FakeEvent("今天好累")
        async for _ in adapter.intercept(event):
            pass
        req = FakeReq()
        req.contexts = [{"role": "user", "content": "其他插件写入的上下文"}]
        await adapter.on_llm_request(event, req)
        self.assertEqual(req.contexts, [{"role": "user", "content": "其他插件写入的上下文"}])

    async def test_shadow_mode_does_not_touch_request(self):
        cfg = Config(db_path=":memory:")
        cfg.metrics.jsonl_path = ""
        cfg.shadow_mode = True
        engine, clock = make_engine(**{"shadow_mode": True})
        adapter = AstrBotAdapter(engine)
        clock.advance()
        event = FakeEvent("今天好累")
        out = [item async for item in adapter.intercept(event)]
        self.assertEqual(out, [])
        self.assertFalse(event.stopped)
        req = FakeReq()
        before = req.system_prompt
        await adapter.on_llm_request(event, req)
        self.assertEqual(req.system_prompt, before)
        await adapter.on_llm_response(event, FakeResp("真实回复"))
        trace = engine.metrics.recent("u1", 1)[0]
        self.assertTrue(trace["mode"].startswith("shadow"))


if __name__ == "__main__":
    unittest.main()
