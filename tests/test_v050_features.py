from __future__ import annotations

import asyncio
import unittest
from datetime import datetime
from unittest import mock

from tests.helpers import FakeClock, make_engine, say
from zhenxin._aux import style_learning
from zhenxin.page_api import PluginPageAPI


class FakeContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakePlugin:
    def __init__(self, context):
        self.context = context


class VersionSourceTests(unittest.TestCase):
    def test_every_consumer_reads_metadata(self):
        import re
        from pathlib import Path

        from zhenxin import __version__, build_info
        from zhenxin._version import FALLBACK_VERSION
        from tests.helpers import ROOT

        metadata = (Path(ROOT) / "metadata.yaml").read_text(encoding="utf-8-sig")
        match = re.search(r"^version:\s*([^\s]+)", metadata, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), __version__)
        self.assertEqual(match.group(1), FALLBACK_VERSION)
        info = build_info()
        self.assertEqual("metadata.yaml", info["source"])
        self.assertFalse(info["mismatch"])


class MigrationTests(unittest.TestCase):
    def test_old_database_gets_new_columns_before_new_index(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        from zhenxin.db import Database

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite3"
            conn = sqlite3.connect(path)
            conn.executescript(
                "CREATE TABLE facts(id INTEGER PRIMARY KEY, user_key TEXT, text TEXT, kind TEXT,"
                " importance REAL, created_at REAL, last_used REAL, used_count INTEGER);"
                "CREATE TABLE metrics(id INTEGER PRIMARY KEY, user_key TEXT, round_no INTEGER,"
                " mode TEXT, wake INTEGER, wake_trigger TEXT, prefix_hash TEXT, prefix_stable INTEGER,"
                " l0_chars INTEGER, l0_balls INTEGER, recent_chars INTEGER, ephemeral_chars INTEGER,"
                " injected_tokens INTEGER, shortcircuit_hit INTEGER, solid_signature TEXT,"
                " solid_confidence REAL, l2_llm_calls INTEGER, main_llm_calls INTEGER,"
                " decision_llm_calls INTEGER, reply_chars INTEGER, created_at REAL, trace_json TEXT);"
                "INSERT INTO facts(user_key,text,kind,importance,created_at,last_used,used_count)"
                " VALUES('u1','不吃香菜','habit',0.8,1,0,0);"
            )
            conn.commit()
            conn.close()
            db = Database(path)
            cols = {row[1] for row in db.connection.execute("PRAGMA table_info(facts)")}
            for name in ("pool", "mentions", "manual_boost", "trashed_at"):
                self.assertIn(name, cols)
            self.assertEqual("不吃香菜", db.scalar("SELECT text FROM facts", (), ""))
            db.close()


class MemoryV2Tests(unittest.TestCase):
    def test_address_is_injected_every_turn(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        engine.facts.add("u1", "称呼我“阿岚”", "address", importance=0.98)
        plan = engine.begin_turn("u1", "今天天气怎么样", round_no=1)
        self.assertIn("〔称呼〕", plan.ephemeral_tail)
        self.assertIn("阿岚", plan.ephemeral_tail)

    def test_rejected_address_goes_to_trash_and_stops_injection(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        engine.facts.add("u1", "称呼我“阿岚”", "address", importance=0.98)
        engine.begin_turn("u1", "别叫我阿岚了", round_no=1)
        self.assertEqual("", engine.facts.address("u1"))
        self.assertEqual(1, engine.facts.pool_counts("u1")["trash"])

    def test_manual_boost_deepens_weight_and_pool(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "不吃香菜", "habit", importance=0.2)
        before = engine.facts.list_facts(pool="cycle")[0]
        boosted = engine.facts.boost(before["id"], 0.2)
        self.assertGreater(boosted["weight"], before["weight"])
        self.assertGreaterEqual(boosted["manual_boost"], 0.2)

    def test_cycle_separates_treasure_and_shallow(self):
        engine, clock = make_engine()
        engine.facts.add("u1", "很重要的事", "habit", importance=0.99)
        engine.facts.add("u1", "随口一句", "fact", importance=0.1)
        for _ in range(8):
            engine.facts.add("u1", "很重要的事", "habit", importance=0.99)
        clock.advance(20 * 86400)
        engine.facts.sweep("u1")
        pools = engine.facts.pool_counts("u1")
        self.assertGreaterEqual(pools["treasure"], 1)
        self.assertGreaterEqual(pools["shallow"], 1)

    def test_disabled_memory_stops_write_and_recall(self):
        engine, _ = make_engine(**{"memory.enabled": False})
        self.assertFalse(engine.facts.add("u1", "不吃香菜", "habit"))
        self.assertEqual([], engine.facts.recall("u1", "我不吃什么来着"))


class StyleLearningTests(unittest.TestCase):
    def test_local_profile_is_capped_and_injected(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 20):
            say(engine, "嗯，今天好累啊，不想动了哈哈", i)
        result = engine.style_learning.refresh("u1", online=False)
        self.assertTrue(result["ok"])
        status = result["status"]
        self.assertTrue(status["knowledge"])
        self.assertLessEqual(status["chars"], 500)
        self.assertIn("用户习惯", status["knowledge"])
        plan = engine.begin_turn("u1", "在吗", round_no=20)
        self.assertIn("〔像真人一点：", plan.ephemeral_tail)

    def test_web_snippets_are_distilled_without_a_model(self):
        snippets = [
            "真人聊天时更常用短句，先说结论再补充细节。",
            "共情的关键是复述对方的感受，而不是马上给建议。",
            "聊天里适度使用语气词会让表达更自然。",
            "这是一条与聊天无关的广告内容。",
        ]
        with mock.patch.object(style_learning, "search_notes", return_value=(snippets, "")):
            engine, _ = make_engine(**{"shortcircuit.enabled": False})
            for i in range(1, 5):
                say(engine, "好呀，今天还行吧", i)
            result = engine.style_learning.refresh("u1", online=True)
        self.assertTrue(result["ok"])
        self.assertIn("短句", result["status"]["knowledge"] or "")
        self.assertIn("web", result["status"]["sources"])


class ProactiveTests(unittest.TestCase):
    def _noon(self) -> float:
        return datetime(2026, 10, 24, 12, 0, 0).timestamp()

    def test_gate_respects_idle_interval_and_daily_cap(self):
        engine, clock = make_engine(**{"proactive.enabled": True})
        engine.cfg.proactive.min_idle_minutes = 60
        engine.cfg.proactive.min_interval_hours = 2
        engine.cfg.proactive.max_per_day = 1
        now = self._noon()
        clock.now = now
        engine.facts.add("u1", "周五要交稿", "unclosed", importance=0.8)
        engine.db.set_state("u1", "last_user_at", now - 3600)
        ok, reason = engine.proactive.gate("u1", now - 3600, now)
        self.assertTrue(ok, reason)
        text, kind = engine.proactive.compose("u1", now)
        self.assertTrue(text)
        self.assertIn(kind, ("open_loop", "memory", "presence"))
        engine.proactive.mark_sent("u1", text, kind, now)
        ok, _ = engine.proactive.gate("u1", now - 3600, now + 1800)
        self.assertFalse(ok)
        ok, _ = engine.proactive.gate("u1", now - 3600, now + 3 * 3600)
        self.assertFalse(ok)  # daily cap reached

    def test_quiet_hours_and_group_filter(self):
        engine, _ = make_engine(**{"proactive.enabled": True})
        quiet = datetime(2026, 10, 24, 2, 0, 0).timestamp()
        ok, reason = engine.proactive.gate("webchat:FriendMessage:1", quiet - 9999, quiet)
        self.assertFalse(ok)
        self.assertEqual("静默时段", reason)
        ok, reason = engine.proactive.gate("aiocqhttp:GroupMessage:9", self._noon(), self._noon())
        self.assertFalse(ok)
        self.assertEqual("群聊不发", reason)

    def test_daily_cap_is_counted_from_the_log(self):
        engine, clock = make_engine(**{"proactive.enabled": True})
        engine.cfg.proactive.max_per_day = 1
        engine.cfg.proactive.min_interval_hours = 0.1
        now = self._noon()
        engine.proactive.mark_sent("u1", "在吗", "presence", now)
        self.assertEqual(1, engine.proactive.stats("u1")["total"])
        ok, reason = engine.proactive.gate("u1", now - 3 * 3600, now + 400)
        self.assertFalse(ok)
        self.assertEqual("今天已经够了", reason)


class ConfigRoundTripTests(unittest.TestCase):
    def test_tail_injection_survives_save(self):
        engine, _ = make_engine()
        plugin_config = {"l0": {"tail_injection": False}}
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, plugin_config)
        api._payload_override = {
            "values": {"l0.tail_injection": True, "l0.inject": 8, "proactive.enabled": True}
        }
        result = asyncio.run(api.save_config())
        self.assertTrue(result["ok"])
        self.assertTrue(engine.cfg.l0.tail_injection)
        self.assertTrue(plugin_config["l0"]["tail_injection"])
        self.assertEqual(8, engine.cfg.l0.inject)
        self.assertTrue(engine.cfg.proactive.enabled)
        self.assertEqual(True, result["config"]["l0"]["tail_injection"])

    def test_defaults_include_new_sections(self):
        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        data = asyncio.run(api.defaults())
        defaults = data["defaults"]
        for path in (
            "memory.top_k",
            "memory.weight_semantic",
            "style_learning.max_chars",
            "proactive.enabled",
            "l0.tail_injection",
        ):
            self.assertIn(path, defaults)


if __name__ == "__main__":
    unittest.main()
