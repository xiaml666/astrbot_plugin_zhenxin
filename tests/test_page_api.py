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
        self.assertEqual(len(context.routes), 18)
        self.assertTrue(context.routes[0][0].endswith("/page/status"))
        self.assertTrue(any(r[0].endswith("/page/preset") for r in context.routes))
        self.assertTrue(any(r[0].endswith("/page/reset") for r in context.routes))
        self.assertTrue(any(r[0].endswith("/page/memory") for r in context.routes))
        self.assertEqual(api._coerce("shadow_mode", "true"), True)
        self.assertEqual(api._coerce("l0.inject", "2"), 2)
        self.assertEqual(api._coerce("l3.host_context_mode", "off"), "off")
        self.assertEqual(api._coerce("live_allowlist", "a\nb\n"), ["a", "b"])
        self.assertEqual(api._coerce("proactive.only_private", "true"), True)
        self.assertIs(api._coerce("aux.rhythm", False), False)
        self.assertEqual(api._coerce("split.enabled", "true"), True)
        self.assertEqual(api._coerce("split.max_segments", "4"), 4)
        self.assertAlmostEqual(api._coerce("split.delay_s", "0.8"), 0.8)

    def test_page_assets_are_beginner_pro_and_jargon_free(self):
        page = (ROOT / "pages" / "zhenxin" / "index.html").read_text(encoding="utf-8")
        # The overview/cost pages were removed; realtime status now sits above
        # every settings tab and the cost numbers live inside it.
        for label in ("运行总览", "对话设置", "记忆", "说话与主动", "体检与反馈", "高级", "重置"):
            self.assertIn(label, page)
        self.assertNotIn('data-tab="overview"', page)
        self.assertNotIn('data-tab="cost"', page)
        self.assertIn("运行总览", page)
        self.assertIn("长期线索", page)
        self.assertNotIn("小球", page)
        self.assertIn("恢复默认配置", page)
        self.assertIn("data-reset", page)
        self.assertIn("关键词痕迹", page)
        self.assertIn("珍藏池", page)
        self.assertIn("浅忆池", page)
        self.assertIn("回收池", page)
        self.assertIn('data-path="l0.tail_injection"', page)
        self.assertIn('id="refresh"', page)
        self.assertIn('id="kb-details"', page)
        self.assertIn('id="graph"', page)
        self.assertIn('id="memundo"', page)
        # The memory garden is one zoomable/pannable canvas, not a static
        # layout: toolbar + wheel/pinch handlers + a transformed <g>.
        self.assertIn('id="zoom-in"', page)
        self.assertIn('id="zoom-out"', page)
        self.assertIn('id="zoom-fit"', page)
        self.assertIn('id="zoom-level"', page)
        self.assertIn('id="graph-view"', page)
        self.assertIn('data-act="weaken"', page)
        self.assertIn("点一下减轻", page)
        self.assertIn("pointermove", page)
        self.assertIn('addEventListener("wheel"', page)
        # Feedback: the category select sits on the left next to its label,
        # and the "open in new window" control is a button with a real
        # sandbox fallback instead of a dead <a href="#">.
        self.assertIn('class="fb-type"', page)
        self.assertIn('<button class="btn small" id="fb-open-link"', page)
        self.assertNotIn('<a class="btn small" id="fb-open-link"', page)
        self.assertNotIn('id="garden"', page)
        self.assertIn('id="hero"', page)
        self.assertIn("关键链路", page)
        # Pricing knobs moved out of the page; cost is display-only now.
        self.assertNotIn("计价单价", page)
        self.assertNotIn("metrics.input_price_per_million", page)
        # Every control carries a one-line explanation.
        self.assertGreater(page.count('class="row"'), 28)
        # AstrBot injects its bridge at the first literal </body>.  A literal
        # body-close inside the inline script would make it terminate early.
        import re

        script = re.findall(r"<script>(.*?)</script>", page, re.S)[0]
        self.assertNotIn("</body>", script)
        # The bridge arrives after this script, so it must be looked up lazily.
        self.assertNotIn("const bridge = window.AstrBotPluginView", script)
        self.assertIn("waitForBridge", script)
        # The preset card sits above the takeover card and is applied by the
        # backend (conflict-aware), not by a client-side constant.
        self.assertLess(page.index("推荐设置"), page.index("对话接管"))
        self.assertIn("page/preset", script)
        self.assertNotIn("const RECOMMENDED", script)
        # All three preset buttons start neutral; only the clicked one turns
        # green (no premature "already taking over" signal).
        self.assertIn('data-preset="takeover"', page)
        self.assertNotIn('class="btn primary" id="takeover"', page)
        self.assertIn("setPresetActive", script)
        # The health button lives in the maintenance nav, not #nav.
        self.assertIn('button[data-tab="health"]', script)
        self.assertNotIn("#nav button[data-tab=\"health\"]", script)
        self.assertIn('id="conflict-alert"', page)
        # Memory colour is the importance scale (5 blended tiers), and the
        # search box actually filters instead of being a dead input.
        self.assertIn("tone-legend", page)
        self.assertIn("第 5 档 墨绿", page)
        self.assertIn("visibleFacts", script)
        self.assertIn("memoryQuery", script)
        self.assertNotIn("颜色深度", page)
        self.assertNotIn("style-online", script)
        # window.confirm is blocked inside AstrBot's sandboxed iframe; the
        # clear/reset handlers must use the in-page dialog instead.
        self.assertNotIn("confirm(", script)
        self.assertIn("askConfirm", script)
        self.assertIn("confirm-mask", page)
        self.assertIn('data-path="style_learning.enabled"', page)
        self.assertIn('data-path="split.enabled"', page)
        self.assertIn("对话分段", page)
        # The search API configuration was removed: built-in search + model
        # screening only.
        self.assertNotIn('data-path="style_learning.search_endpoint"', page)
        self.assertNotIn('id="search-key"', page)
        # The old statistical imitation switch is gone for good.
        self.assertNotIn('data-path="aux.style"', page)

    def test_memory_list_sorts_by_time_and_exposes_pools(self):
        import asyncio

        engine, clock = make_engine()
        engine.facts.add("u1", "今天天气不错", "fact", importance=0.2)
        engine.facts.add("u1", "不吃香菜", "habit", importance=0.9)
        clock.advance(40 * 86400)
        engine.facts.sweep("u1")
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._query_override = {"view": "all"}
        data = asyncio.run(api.memory_list())
        self.assertTrue(data["ok"])
        self.assertEqual(2, data["total"])
        # Auto sorted by creation time, newest first; no manual sort needed.
        self.assertEqual("不吃香菜", data["facts"][0]["text"])
        self.assertIn("importance", data["facts"][0])
        self.assertIn("pool", data["facts"][0])
        self.assertIn("weight", data["facts"][0])
        self.assertGreaterEqual(data["pools"]["shallow"], 1)
        # Deep-clean is still available as an explicit maintenance operation.
        removed = engine.facts.sweep("u1", min_age_days=30, max_archive=50)
        self.assertGreaterEqual(removed, 1)
        trace = engine.facts.traces("u1")[0]
        self.assertTrue(trace["keywords"])
        self.assertNotIn("今天天气不错", trace["keywords"])

    def test_defaults_preset_uses_the_shipped_schema(self):
        from zhenxin.page_api import schema_defaults

        defaults = schema_defaults()
        self.assertEqual(defaults["l3.host_context_mode"], "replace")
        self.assertEqual(defaults["shadow_mode"], True)
        self.assertEqual(defaults["l0.inject"], 3)
        self.assertEqual(defaults["metrics.output_price_per_million"], 8.0)
        self.assertEqual(defaults["aux.rhythm"], True)
        self.assertNotIn("aux.style", defaults)
        self.assertNotIn("style_learning.search_endpoint", defaults)
        self.assertEqual(defaults["split.enabled"], False)
        self.assertEqual(defaults["split.max_segments"], 3)
        self.assertEqual(defaults["split.delay_s"], 0.6)

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

    @staticmethod
    def _scan(*plugins):
        return {"ok": True, "plugins": list(plugins), "summary": {}}

    def test_takeover_preset_asks_before_conflicts_then_applies(self):
        import asyncio

        engine, _ = make_engine()
        engine.cfg.shadow_mode = True
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._scan_plugins = lambda force=False: self._scan(
            {
                "name": "astrbot_plugin_splitter",
                "display": "分段",
                "severity": "warning",
                "enabled": True,
                "reasons": ["改写 system_prompt"],
            }
        )
        api._payload_override = {"mode": "takeover"}
        first = asyncio.run(api.preset())
        self.assertTrue(first["needs_confirm"])
        self.assertTrue(engine.cfg.shadow_mode)
        self.assertTrue(first["issues"])

        api._payload_override = {"mode": "takeover", "force": True}
        second = asyncio.run(api.preset())
        self.assertTrue(second["applied"])
        self.assertFalse(engine.cfg.shadow_mode)
        self.assertEqual("replace", engine.cfg.l3.host_context_mode)
        # Takeover is the budget-conscious profile, not just a mode switch.
        self.assertEqual(2, engine.cfg.memory.top_k)
        self.assertAlmostEqual(0.32, engine.cfg.memory.min_score)
        self.assertAlmostEqual(0.55, engine.cfg.memory.weight_semantic)
        self.assertEqual(1, engine.cfg.proactive.max_per_day)
        self.assertAlmostEqual(24, engine.cfg.proactive.min_interval_hours)
        self.assertEqual(240, engine.cfg.proactive.min_idle_minutes)

    def test_recommended_preset_adapts_to_conflicts(self):
        import asyncio

        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._scan_plugins = lambda force=False: self._scan(
            {
                "name": "mem0",
                "display": "mem0",
                "severity": "direct",
                "enabled": True,
                "memoryish": True,
                "reasons": ["改写 req.contexts"],
            },
            {
                "name": "astrbot_plugin_proactive_chat",
                "display": "主动消息",
                "severity": "warning",
                "enabled": True,
                "memoryish": False,
                "reasons": ["功能重复"],
            },
        )
        api._payload_override = {"mode": "recommended"}
        result = asyncio.run(api.preset())
        self.assertTrue(result["applied"])
        self.assertTrue(engine.cfg.shadow_mode)
        self.assertEqual("off", engine.cfg.l3.host_context_mode)
        self.assertFalse(engine.cfg.l2.enabled)
        self.assertFalse(engine.cfg.aux.compression)
        self.assertFalse(engine.cfg.proactive.enabled)
        self.assertEqual(2, engine.cfg.memory.top_k)
        self.assertAlmostEqual(0.35, engine.cfg.memory.min_score)
        self.assertAlmostEqual(0.5, engine.cfg.memory.weight_semantic)
        self.assertAlmostEqual(12, engine.cfg.proactive.min_interval_hours)
        self.assertEqual(2, engine.cfg.proactive.max_per_day)

    def test_defaults_preset_restores_fresh_install(self):
        import asyncio

        engine, _ = make_engine()
        engine.cfg.shadow_mode = False
        engine.cfg.l3.host_context_mode = "off"
        engine.cfg.proactive.enabled = True
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._scan_plugins = lambda force=False: self._scan()
        api._payload_override = {"mode": "defaults"}
        result = asyncio.run(api.preset())
        self.assertTrue(result["applied"])
        self.assertTrue(engine.cfg.shadow_mode)
        self.assertEqual("replace", engine.cfg.l3.host_context_mode)
        self.assertFalse(engine.cfg.proactive.enabled)
        self.assertEqual(3, engine.cfg.memory.top_k)
        self.assertEqual(2, engine.cfg.proactive.max_per_day)

    def test_health_scan_alerts_on_a_newly_enabled_plugin(self):
        import asyncio

        engine, _ = make_engine()
        engine.db.set_state("__global__", "health_known_plugins", [])
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._scan_plugins = lambda force=False: self._scan(
            {
                "name": "astrbot_plugin_proactive_chat",
                "display": "主动消息",
                "severity": "warning",
                "enabled": True,
                "reasons": ["功能重复"],
            }
        )
        data = asyncio.run(api.health_scan())
        self.assertEqual(1, len(data["alert"]["items"]))
        self.assertEqual("astrbot_plugin_proactive_chat", data["alert"]["items"][0]["name"])
        self.assertIsNotNone(engine.db.get_state("__global__", "conflict_alert"))

    def test_first_user_initiated_scan_alerts_existing_conflicts(self):
        import asyncio

        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._scan_plugins = lambda force=False: self._scan(
            {
                "name": "astrbot_plugin_splitter",
                "display": "分段",
                "severity": "warning",
                "enabled": True,
                "reasons": ["改写 system_prompt"],
            }
        )
        # Opening the page / health tab is user-initiated: an existing conflict
        # must be visible immediately instead of looking like a silent bug.
        data = asyncio.run(api.health_scan())
        self.assertIsNotNone(data["alert"])
        self.assertEqual(
            ["astrbot_plugin_splitter"],
            engine.db.get_state("__global__", "health_known_plugins", []),
        )

    def test_status_scans_immediately_when_the_plugin_set_changes(self):
        import asyncio

        engine, _ = make_engine()
        engine.db.set_state("__global__", "health_known_plugins", [])
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        scans = []

        def scan(force=False):
            scans.append(force)
            return self._scan(
                {
                    "name": "mem0",
                    "display": "mem0",
                    "severity": "direct",
                    "enabled": True,
                    "reasons": ["改写 req.contexts"],
                }
            )

        api._scan_plugins = scan
        api._plugins_dir = lambda: ROOT
        api._plugins_fingerprint = lambda _path: "v1"
        data = asyncio.run(api.status())
        self.assertEqual(1, len(data["conflict_alert"]["items"]))
        self.assertEqual("mem0", data["conflict_alert"]["items"][0]["name"])
        # Same fingerprint → no second full scan on the next status poll.
        api._alert_checked_at = 0.0
        asyncio.run(api.status())
        self.assertEqual(1, len(scans))

    def test_conflict_alert_fires_again_after_off_and_on(self):
        import asyncio

        engine, _ = make_engine()
        engine.db.set_state("__global__", "health_known_plugins", [])
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        enabled = {
            "name": "astrbot_plugin_proactive_chat",
            "display": "主动消息",
            "severity": "warning",
            "enabled": True,
            "reasons": ["功能重复"],
        }
        disabled = {**enabled, "enabled": False, "severity": "info"}
        api._scan_plugins = lambda force=False: self._scan(enabled)
        self.assertIsNotNone(asyncio.run(api.health_scan())["alert"])
        api._scan_plugins = lambda force=False: self._scan(disabled)
        self.assertIsNone(asyncio.run(api.health_scan())["alert"])
        api._scan_plugins = lambda force=False: self._scan(enabled)
        self.assertIsNotNone(asyncio.run(api.health_scan())["alert"])


if __name__ == "__main__":
    unittest.main()
