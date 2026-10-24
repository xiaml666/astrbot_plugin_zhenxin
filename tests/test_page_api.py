from __future__ import annotations

import unittest

from tests.helpers import ROOT, make_engine, say
from zhenxin.page_api import PluginPageAPI


class FakeContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakePlugin:
    def __init__(self, context):
        self.context = context


class PageApiTests(unittest.TestCase):
    def test_routes_and_coercion(self):
        engine, _ = make_engine()
        context = FakeContext()
        config = {}
        api = PluginPageAPI(FakePlugin(context), engine, config)
        api.register()
        self.assertEqual(len(context.routes), 7)
        self.assertTrue(context.routes[0][0].endswith("/page/status"))
        self.assertTrue(any(r[0].endswith("/page/defaults") for r in context.routes))
        self.assertTrue(any(r[0].endswith("/page/reset") for r in context.routes))
        self.assertTrue(any(r[0].endswith("/page/memory") for r in context.routes))
        self.assertEqual(api._coerce("shadow_mode", "true"), True)
        self.assertEqual(api._coerce("l0.inject", "2"), 2)
        self.assertEqual(api._coerce("l3.host_context_mode", "off"), "off")
        self.assertEqual(api._coerce("live_allowlist", "a\nb\n"), ["a", "b"])

    def test_page_assets_are_beginner_pro_and_jargon_free(self):
        page = (ROOT / "pages" / "zhenxin" / "index.html").read_text(encoding="utf-8")
        # User-facing sections, each with plain-language help.
        for label in ("概览", "记忆", "花费", "行为设置", "高级", "重置"):
            self.assertIn(label, page)
        self.assertIn("长期线索", page)
        self.assertNotIn("小球", page)
        self.assertIn("恢复默认配置", page)
        self.assertIn("data-reset", page)
        self.assertIn("关键词痕迹", page)
        # Pricing knobs moved out of the page; cost is display-only now.
        self.assertNotIn("计价单价", page)
        self.assertNotIn("metrics.input_price_per_million", page)
        # Every control carries a one-line explanation.
        self.assertGreater(page.count('class="row"'), 15)
        # AstrBot injects its bridge at the first literal </body>.  A literal
        # body-close inside the inline script would make it terminate early.
        import re

        script = re.findall(r"<script>(.*?)</script>", page, re.S)[0]
        self.assertNotIn("</body>", script)
        # The bridge arrives after this script, so it must be looked up lazily.
        self.assertNotIn("const bridge = window.AstrBotPluginView", script)
        self.assertIn("waitForBridge", script)
        # window.confirm is blocked inside AstrBot's sandboxed iframe; the
        # clear/reset handlers must use the in-page dialog instead.
        self.assertNotIn("confirm(", script)
        self.assertIn("askConfirm", script)
        self.assertIn("confirm-mask", page)

    def test_memory_list_separates_importance_and_keeps_keyword_traces(self):
        import asyncio

        engine, clock = make_engine()
        engine.facts.add("u1", "今天天气不错", "fact", importance=0.2)
        engine.facts.add("u1", "不吃香菜", "habit", importance=0.9)
        clock.advance(40 * 86400)
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.memory_list())
        self.assertTrue(data["ok"])
        self.assertEqual(1, data["total"])
        self.assertEqual("不吃香菜", data["facts"][0]["text"])
        self.assertIn("importance", data["facts"][0])
        self.assertIn("tier", data["facts"][0])
        self.assertGreaterEqual(data["archived_total"], 1)
        trace = data["traces"][0]
        self.assertTrue(trace["keywords"])
        self.assertNotIn("今天天气不错", trace["keywords"])

    def test_defaults_endpoint_reads_the_shipped_schema(self):
        import asyncio

        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.defaults())
        self.assertTrue(data["ok"])
        defaults = data["defaults"]
        self.assertEqual(defaults["l3.host_context_mode"], "replace")
        self.assertEqual(defaults["shadow_mode"], True)
        self.assertEqual(defaults["l0.inject"], 3)
        self.assertEqual(defaults["metrics.output_price_per_million"], 8.0)

    def test_reset_conversation_keeps_facts_and_metrics(self):
        import asyncio

        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 4):
            clock.advance()
            say(engine, f"第{i}句普通对话", i)
        before = engine.stats(None)
        self.assertGreater(before["rounds"], 0)
        engine.record_error("llm_request", "keep me")

        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._payload_override = {"scope": "conversation"}
        result = asyncio.run(api.reset())
        self.assertTrue(result["ok"])
        for table in ("turns", "hot_slots", "warm_slots", "cold_seeds", "topic_stack"):
            self.assertEqual(
                0, engine.db.scalar(f"SELECT COUNT(*) FROM {table}", (), 0), table
            )
        self.assertEqual(before["rounds"], engine.stats(None)["rounds"])
        # Global health counters survive a conversation reset.
        self.assertEqual(
            1, int(engine.db.get_state("__global__", "error_count", 0) or 0)
        )

    def test_reset_all_clears_metrics_and_restores_config(self):
        import asyncio

        from zhenxin.page_api import schema_defaults

        engine, clock = make_engine()
        clock.advance()
        say(engine, "普通一句", 1)
        engine.cfg.l3.host_context_mode = "replace"
        engine.cfg.l0.max_inject_chars = 299
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._payload_override = {"scope": "all"}
        result = asyncio.run(api.reset())
        self.assertTrue(result["ok"])
        self.assertEqual(0, engine.stats(None)["rounds"])
        defaults = schema_defaults()
        self.assertEqual(engine.cfg.l3.host_context_mode, defaults["l3.host_context_mode"])
        self.assertEqual(engine.cfg.l0.max_inject_chars, defaults["l0.max_inject_chars"])

    def test_status_payload_has_safe_metrics(self):
        import asyncio

        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.status())
        self.assertTrue(data["ok"])
        self.assertIn("metrics", data)
        self.assertIn("actual_cost", data["metrics"])
        self.assertNotIn("password", str(data).lower())

    def test_status_payload_reports_health_and_series(self):
        import asyncio

        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 4):
            clock.advance()
            say(engine, f"第{i}句普通对话", i)
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.status())
        health = data["health"]
        self.assertTrue(health["db_ok"])
        self.assertEqual(0, health["errors"])
        self.assertEqual("", health["last_error"])
        self.assertEqual(3, len(data["series"]))
        self.assertIn("cost", data["series"][0])

    def test_errors_are_recorded_for_the_page(self):
        import asyncio

        engine, _ = make_engine()
        engine.record_error("llm_request", "boom")
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.status())
        self.assertEqual(1, data["health"]["errors"])
        self.assertIn("boom", data["health"]["last_error"])


if __name__ == "__main__":
    unittest.main()
