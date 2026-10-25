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


class ConflictScanTests(unittest.TestCase):
    def test_media_plugins_are_not_conflicts(self):
        import tempfile
        from pathlib import Path

        from zhenxin.page_api import PluginPageAPI

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "astrbot_plugin_bilibili"
            base.mkdir()
            (base / "metadata.yaml").write_text(
                "name: astrbot_plugin_bilibili\ndisplay_name: 哔哩哔哩\ndesc: B站视频链接\n",
                encoding="utf-8",
            )
            (base / "main.py").write_text(
                "from astrbot.api.event import filter\n"
                "@filter.on_llm_request()\n"
                "async def x(event, req):\n"
                "    req.contexts = []\n    req.system_prompt = ''\n",
                encoding="utf-8",
            )
            self.assertIsNone(PluginPageAPI._inspect_plugin(base, set()))

    def test_enabled_memory_plugin_is_direct_from_context_or_prompt(self):
        import tempfile
        from pathlib import Path

        from zhenxin.page_api import PluginPageAPI

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "astrbot_plugin_memory_x"
            base.mkdir()
            (base / "metadata.yaml").write_text(
                "name: astrbot_plugin_memory_x\ndisplay_name: 记忆X\ndesc: 长期记忆与上下文摘要\n",
                encoding="utf-8",
            )
            (base / "main.py").write_text(
                "from astrbot.api.event import filter\n"
                "@filter.on_llm_request()\n"
                "async def x(event, req):\n"
                "    req.contexts = []\n",
                encoding="utf-8",
            )
            item = PluginPageAPI._inspect_plugin(base, set())
            self.assertIsNotNone(item)
            self.assertEqual("direct", item["severity"])

    def test_poll_conflicts_alerts_only_on_newly_enabled_plugin(self):
        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})

        def payload(names):
            return {
                "plugins": [
                    {"name": n, "display": n, "severity": "direct", "reasons": ["x"]}
                    for n in names
                ]
            }

        api._scan_plugins = lambda force=False: payload(["old_memory"])
        self.assertIsNone(api.poll_conflicts())  # baseline, no alert
        self.assertIsNone(api.poll_conflicts())  # unchanged, no alert
        api._scan_plugins = lambda force=False: payload(["old_memory", "new_memory"])
        alert = api.poll_conflicts()
        self.assertIsNotNone(alert)
        self.assertEqual("new_memory", alert["items"][0]["name"])


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

    def test_vague_cancellation_targets_a_single_open_loop(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        engine.facts.add("u1", "海盐绘本", "unclosed", importance=0.8)
        self.assertEqual(1, engine.facts.scan_zero_value("u1", "不想弄这个了"))
        self.assertEqual(1, engine.facts.pool_counts("u1")["trash"])

    def test_vague_address_rejection_uses_the_word_not_an_unrelated_verb(self):
        from zhenxin._aux import zero_value

        self.assertEqual("address_cancel", zero_value.cue("不要这样称呼我了"))
        self.assertTrue(zero_value.matches("称呼我“阿岚”", "address", "不要这样称呼我了", "address_cancel"))
        self.assertEqual("cancelled", zero_value.cue("不要叫外卖了"))
        self.assertFalse(zero_value.matches("称呼我“阿岚”", "address", "不要叫外卖了", "cancelled"))

    def test_backfill_recovers_identity_from_history(self):
        engine, _ = make_engine()
        engine.db.append_turn("u1", 1, "user", "你以后就叫我老大", 1.0)
        engine.db.append_turn("u1", 2, "user", "记住，我不吃香菜", 1.1)
        engine.db.append_turn("u1", 3, "user", "你以后叫小满吧", 1.2)
        added = engine.facts.backfill_from_turns("u1")
        self.assertGreaterEqual(added, 3)
        self.assertIn("老大", engine.facts.address("u1"))
        self.assertIn("小满", engine.facts.assistant_name("u1"))
        self.assertTrue(any("香菜" in row["text"] for row in engine.facts.list_facts(pool="cycle")))

    def test_backfill_is_idempotent(self):
        engine, _ = make_engine()
        engine.db.append_turn("u1", 1, "user", "你以后就叫我老大", 1.0)
        engine.facts.backfill_from_turns("u1")
        before = engine.facts.list_facts()[0]
        engine.facts.backfill_from_turns("u1")
        after = engine.facts.list_facts()[0]
        self.assertEqual(before["mentions"], after["mentions"])
        self.assertEqual(before["importance"], after["importance"])

    def test_backfill_does_not_resurrect_a_rejected_address(self):
        engine, _ = make_engine()
        engine.db.append_turn("u1", 1, "user", "叫我阿岚", 1.0)
        engine.db.append_turn("u1", 2, "user", "别叫我阿岚了", 1.1)
        engine.facts.backfill_from_turns("u1")
        self.assertEqual("", engine.facts.address("u1"))

    def test_a_naming_question_is_not_the_users_address(self):
        # Regression: "那我叫你小肥鱼好不好" used to be parsed as
        # 称呼我“你小肥鱼好不好” because "我叫" matched mid-sentence.
        from zhenxin._aux import zero_value

        self.assertEqual([], zero_value.identity_facts("那我叫你小肥鱼好不好"))
        engine, _ = make_engine()
        engine.db.append_turn("u1", 1, "user", "那我叫你小肥鱼好不好", 1.0)
        engine.facts.backfill_from_turns("u1")
        self.assertEqual("", engine.facts.address("u1"))
        self.assertEqual("", engine.facts.assistant_name("u1"))

    def test_identity_statements_still_work(self):
        from zhenxin._aux import zero_value

        self.assertIn("老大", zero_value.identity_facts("你以后就叫我老大")[0][0])
        self.assertEqual(
            [("你叫“小肥鱼”", "assistant_name")],
            zero_value.identity_facts("以后我就叫你小肥鱼"),
        )
        self.assertEqual([], zero_value.identity_facts("我叫外卖了"))
        self.assertEqual([], zero_value.identity_facts("我叫你不要熬夜"))

    def test_weather_message_does_not_create_an_anger_fact(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        say(engine, "我在巨野，这里明天的天气怎么样？", 1)
        texts = {row["text"] for row in engine.facts.list_facts()}
        self.assertFalse(any("生气" in text for text in texts))

    def test_repair_emotions_removes_weather_anger_but_keeps_real_emotion(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        engine.facts.add("u1", "最近生气（明天）", "emotion", importance=0.73)
        engine.facts.add("u1", "最近疲惫（甲方）", "emotion", importance=0.8)
        engine.db.append_turn("u1", 1, "user", "我在巨野，这里明天的天气怎么样？", 1.0)
        engine.db.append_turn("u1", 2, "user", "今天好累，甲方又改需求", 1.1)
        removed = engine.facts.repair_emotions("u1")
        self.assertEqual(1, removed)
        texts = {row["text"] for row in engine.facts.list_facts()}
        self.assertNotIn("最近生气（明天）", texts)
        self.assertIn("最近疲惫（甲方）", texts)

    def test_repair_identity_removes_legacy_garbage_and_keeps_real_names(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "称呼我“你小肥鱼好不好”", "address", importance=0.98)
        engine.facts.add("u1", "称呼我“阿岚”", "address", importance=0.98)
        engine.db.append_turn("u1", 1, "user", "那我叫你小肥鱼好不好", 1.0)
        removed = engine.facts.repair_identity("u1")
        self.assertEqual(1, removed)
        self.assertEqual("称呼我“阿岚”", engine.facts.address("u1"))
        self.assertEqual("", engine.facts.assistant_name("u1"))

    def test_conflict_alert_is_mentioned_once_to_the_model(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        engine.db.set_state("__global__", "conflict_alert", {
            "at": 1.0,
            "items": [{"name": "memory_x", "display": "记忆X", "severity": "direct", "reasons": ["contexts"]}],
        })
        plan = engine.begin_turn("u1", "在吗", round_no=1)
        self.assertIn("〔插件提醒〕", plan.ephemeral_tail)
        self.assertIn("记忆X", plan.ephemeral_tail)
        alert = engine.db.get_state("__global__", "conflict_alert", {})
        self.assertTrue(alert.get("injected"))
        plan2 = engine.begin_turn("u1", "在吗", round_no=2)
        self.assertNotIn("〔插件提醒〕", plan2.ephemeral_tail)

    def test_manual_boost_deepens_weight_and_pool(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "不吃香菜", "habit", importance=0.2)
        before = engine.facts.list_facts(pool="cycle")[0]
        boosted = engine.facts.boost(before["id"], 0.2)
        self.assertGreater(boosted["weight"], before["weight"])
        self.assertGreaterEqual(boosted["manual_boost"], 0.2)

    def test_boost_can_be_undone_exactly(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "不吃香菜", "habit", importance=0.5)
        before = engine.facts.list_facts()[0]
        boosted = engine.facts.boost(before["id"], 0.12)
        self.assertGreater(boosted["manual_boost"], before["manual_boost"])
        restored = engine.facts.set_boost(
            before["id"], before["manual_boost"], before["importance"], before["pool"]
        )
        self.assertEqual(before["manual_boost"], restored["manual_boost"])
        self.assertEqual(before["importance"], restored["importance"])
        self.assertEqual(before["pool"], restored["pool"])

    def test_color_follows_importance_in_five_tiers(self):
        engine, clock = make_engine()
        for index, importance in enumerate((0.05, 0.3, 0.55, 0.8, 0.98)):
            engine.facts.add("u1", f"记忆{index}", "fact", importance=importance)
        facts = {f["text"]: f for f in engine.facts.list_facts("u1", limit=50)}
        ordered = [facts[f"记忆{i}"] for i in range(5)]
        self.assertEqual([1, 2, 3, 4, 5], [f["tone"] for f in ordered])
        self.assertEqual(5, len({f["color"] for f in ordered}))
        self.assertEqual("淡灰", ordered[0]["tone_label"])
        self.assertEqual("墨绿", ordered[4]["tone_label"])
        # Colour is bound to importance, not to age/retention: an old but
        # important fact keeps the same dark tone as a fresh one.
        engine.facts.add("u1", "老记忆", "fact", importance=0.98)
        clock.advance(200 * 86400)
        old = {f["text"]: f for f in engine.facts.list_facts("u1", limit=50)}["老记忆"]
        self.assertEqual(ordered[4]["color"], old["color"])
        self.assertLess(old["retention"], ordered[4]["retention"])

    def test_weaken_lightens_symmetrically_and_demotes_pools(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "爱喝冰美式", "habit", importance=0.8)
        before = engine.facts.list_facts()[0]
        # Negative delta is a real lighten, not a no-op: manual boost goes
        # below zero and the dot visibly gets lighter.
        weakened = engine.facts.boost(before["id"], -0.12)
        self.assertLess(weakened["manual_boost"], 0)
        self.assertLess(weakened["importance"], before["importance"])
        self.assertLess(weakened["weight"], before["weight"])
        for _ in range(3):
            weakened = engine.facts.boost(before["id"], -0.12)
        self.assertEqual("shallow", weakened["pool"])
        # Deepening returns to the original state and can cross into treasure.
        for _ in range(6):
            deepened = engine.facts.boost(before["id"], 0.12)
        self.assertGreater(deepened["manual_boost"], 0.2)
        self.assertGreaterEqual(deepened["weight"], before["weight"])

    def test_weakened_boost_can_be_undone_exactly(self):
        engine, _ = make_engine()
        engine.facts.add("u1", "养了猫叫团子", "relationship", importance=0.7)
        before = engine.facts.list_facts()[0]
        engine.facts.boost(before["id"], -0.24)
        restored = engine.facts.set_boost(
            before["id"], before["manual_boost"], before["importance"], before["pool"]
        )
        self.assertEqual(before["manual_boost"], restored["manual_boost"])
        self.assertEqual(before["importance"], restored["importance"])
        self.assertEqual(before["pool"], restored["pool"])

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
    def test_a_stray_filler_is_never_a_candidate(self):
        # The reported bug: one "就是" in the middle of ordinary messages was
        # stored as "开口常说就是".  A sentence without an explicit preference
        # cue must never even reach the model.
        texts = [
            "嗯，今天就是有点累",
            "随便聊聊吧",
            "这个就是那个意思",
        ]
        self.assertEqual([], style_learning.preference_candidates(texts))

    def test_explicit_preferences_are_candidates_but_questions_are_not(self):
        texts = [
            "以后别老追问我了",
            "说短一点就行",
            "那我叫你小肥鱼好不好",
        ]
        picked = style_learning.preference_candidates(texts)
        self.assertIn("以后别老追问我了", picked)
        self.assertTrue(any("短一点" in item for item in picked))
        self.assertNotIn("那我叫你小肥鱼好不好", picked)

    def test_without_a_model_nothing_is_written(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 8):
            say(engine, "希望以后说话短一点，别老追问", i)
        result = engine.style_learning.refresh("u1", online=False)
        self.assertTrue(result["ok"])
        self.assertIn("甄别模型", result["error"])
        self.assertNotIn("pref", result["status"]["sources"])
        self.assertNotIn("短一点", result["status"]["knowledge"])

    def test_screened_preferences_are_stored_and_injected(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 8):
            say(engine, "以后说话短一点，别老追问", i)
        seen = []

        def reviewer(kind, items):
            seen.append((kind, list(items)))
            return ["对方希望回复短一点，不要连续追问"]

        result = engine.style_learning.refresh("u1", online=False, reviewer=reviewer)
        self.assertTrue(result["ok"])
        self.assertEqual("pref", seen[0][0])
        self.assertIn("短一点", result["status"]["knowledge"])
        self.assertIn("pref", result["status"]["sources"])
        plan = engine.begin_turn("u1", "在吗", round_no=9)
        self.assertIn("〔说话偏好：", plan.ephemeral_tail)

    def test_web_snippets_are_prefiltered_before_the_model(self):
        snippets = [
            "限时优惠，扫码加微信领取英语课程，名额有限。",
            "论坛里大家聊天更爱用短句，一次只说一件事。",
            "真人聊天时先接住情绪，再给建议。",
        ]
        picked = style_learning.web_candidates(snippets)
        self.assertEqual(2, len(picked))
        self.assertFalse(any("限时" in item for item in picked))

    def test_model_rejection_clears_web_knowledge(self):
        snippets = [
            "真人聊天时更常用短句，先说结论再补充细节。",
            "这是一条与聊天无关的广告内容。",
        ]
        with mock.patch.object(style_learning, "search_notes", return_value=(snippets, "")):
            engine, _ = make_engine(**{"shortcircuit.enabled": False})
            for i in range(1, 5):
                say(engine, "好呀，今天还行吧", i)
            result = engine.style_learning.refresh(
                "u1", online=True, reviewer=lambda kind, items: []
            )
        self.assertTrue(result["ok"])
        self.assertNotIn("web", result["status"]["sources"])
        self.assertNotIn("广告", result["status"]["knowledge"])

    def test_screened_web_knowledge_is_stored(self):
        snippets = ["论坛里大家聊天更爱用短句，一次只说一件事。"]
        with mock.patch.object(style_learning, "search_notes", return_value=(snippets, "")):
            engine, _ = make_engine(**{"shortcircuit.enabled": False})
            for i in range(1, 5):
                say(engine, "好呀，今天还行吧", i)
            result = engine.style_learning.refresh(
                "u1",
                online=True,
                reviewer=lambda kind, items: ["日常聊天多用短句，一次只说一件事"],
            )
        self.assertIn("web", result["status"]["sources"])
        self.assertIn("短句", result["status"]["knowledge"])

    def test_topic_queries_follow_the_conversation_not_generic_pages(self):
        queries, topics = style_learning.topic_queries(
            ["在家写小说", "今天又卡文了", "新书没思路", "卡文的时候你会怎么办"]
        )
        self.assertIn("小说", topics)
        self.assertNotIn("知道", topics)
        self.assertTrue(any("小说" in query for query in queries))

    def test_360_results_are_parsed_and_network_errors_are_hidden(self):
        page = (
            '<h3 class="res-title"><a data-mdurl="https://www.v2ex.com/t/1">x</a></h3>'
            '<div><span class="res-list-summary">论坛里大家聊天更爱用短句，一次只说一件事。</span></div>'
        )
        with mock.patch.object(style_learning, "_fetch", return_value=page):
            snippets, error = style_learning.search_notes(
                queries=("小说 怎么聊",), timeout=1
            )
        self.assertEqual(["论坛里大家聊天更爱用短句，一次只说一件事。"], snippets)
        self.assertEqual("", error)

    def test_web_candidates_accept_topic_knowledge(self):
        self.assertEqual(
            ["小说卡文时可以先放松一下，再回头改大纲"],
            style_learning.web_candidates(
                ["小说卡文时可以先放松一下，再回头改大纲。"], topics=["小说"]
            ),
        )

    def test_search_results_must_match_the_requested_site(self):
        page = (
            '<li class="b_algo"><h2><a href="https://www.yomiclub.com/">x</a></h2>'
            "<p>匿名聊天交友站点，随机匹配陌生人开启对话。</p></li>"
            '<li class="b_algo"><h2><a href="https://www.v2ex.com/t/1">y</a></h2>'
            "<p>论坛里大家聊天更爱用短句，一次只说一件事。</p></li>"
        )
        with mock.patch.object(style_learning, "_fetch", return_value=page):
            snippets, _error = style_learning.search_notes(
                queries=("site:v2ex.com 聊天",), timeout=1
            )
        self.assertEqual(["论坛里大家聊天更爱用短句，一次只说一件事。"], snippets)

    def test_redirect_links_are_unwrapped_and_domains_checked(self):
        encoded = "a1aHR0cHM6Ly9leGFtcGxlLmNvbS9h"
        redirect = "https://www.bing.com/ck/a?x=1&amp;u=" + encoded
        self.assertEqual("https://example.com/a", style_learning._real_url(redirect))
        self.assertTrue(style_learning._url_allowed("https://v2ex.com/t/1", "v2ex.com"))
        self.assertTrue(style_learning._url_allowed("https://www.v2ex.com/t/1", "v2ex.com"))
        self.assertFalse(style_learning._url_allowed("https://evil-v2ex.com/t/1", "v2ex.com"))
        self.assertTrue(style_learning._url_allowed("", ""))

    def test_parse_screened_accepts_json_and_rejects_ads(self):
        self.assertEqual(
            ["对方希望被叫小肥鱼"],
            style_learning.parse_screened('["对方希望被叫小肥鱼"]'),
        )
        self.assertEqual(
            ["日常聊天多用短句"],
            style_learning.parse_screened('```json\n["日常聊天多用短句"]\n```'),
        )
        self.assertEqual([], style_learning.parse_screened('["限时优惠，扫码报名"]'))
        self.assertIsNone(style_learning.parse_screened("完全不是 JSON"))
        self.assertIsNone(style_learning.parse_screened(""))

    def test_legacy_imitation_and_unfiltered_web_rows_are_purged(self):
        import tempfile
        from pathlib import Path

        from zhenxin.db import Database

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zhenxin.sqlite3"
            db = Database(path)
            db.execute(
                "INSERT INTO style_knowledge(user_key,source,content,chars,created_at,updated_at)"
                " VALUES('u1','local','用户习惯：开口常说“就是”',16,1,1),"
                "('u1','web','限时优惠，扫码报名',9,1,1),"
                "('u1','seed','内置常识',4,1,1)"
            )
            # Pretend this is a 0.6.5 database: version 1 means "the old learner
            # ran here", so the next open must purge its rows once.
            db.connection.execute("PRAGMA user_version=1")
            db.connection.commit()
            db.close()
            db = Database(path)
            sources = {
                str(row["source"])
                for row in db.fetchall("SELECT source FROM style_knowledge WHERE user_key='u1'")
            }
            self.assertEqual({"seed"}, sources)
            db.close()


class StyleLearnApiTests(unittest.TestCase):
    def test_manual_learn_screens_with_the_host_model(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False, "style_learning.min_turns": 1})
        say(engine, "以后说话短一点，别老追问", 1)
        plugin = FakePlugin(FakeContext())
        calls = []

        async def screener(kind, items, user=""):
            calls.append((kind, list(items)))
            return ["对方希望回复短一点，不要连续追问"]

        plugin.screen_knowledge = screener
        api = PluginPageAPI(plugin, engine, {})
        api._payload_override = {"user": "u1", "online": False}
        data = asyncio.run(api.style_learn())
        self.assertTrue(data["ok"])
        self.assertEqual("pref", calls[0][0])
        self.assertIn("短一点", data["status"]["knowledge"])
        self.assertIn("pref", data["status"]["sources"])

    def test_manual_learn_explains_when_there_is_no_preference_sentence(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False, "style_learning.min_turns": 1})
        say(engine, "你好，今天在家写小说", 1)
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._payload_override = {"user": "u1", "online": False}
        data = asyncio.run(api.style_learn())
        self.assertTrue(data["ok"])
        self.assertIn("没有找到明确的说话偏好", data["message"])
        self.assertIn("称呼名字不算", data["message"])

    def test_split_settings_round_trip_through_the_page(self):
        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._payload_override = {
            "values": {
                "split.enabled": True,
                "split.max_segments": 6,
                "split.min_segment_chars": 12,
                "split.delay_mode": "slow",
            }
        }
        result = asyncio.run(api.save_config())
        self.assertTrue(result["ok"])
        self.assertTrue(engine.cfg.split.enabled)
        self.assertEqual(6, engine.cfg.split.max_segments)
        self.assertEqual(12, engine.cfg.split.min_segment_chars)
        self.assertEqual("slow", engine.cfg.split.delay_mode)
        saved = result["config"]["split"]
        self.assertEqual(6, saved["max_segments"])
        self.assertEqual(12, saved["min_segment_chars"])
        self.assertEqual("slow", saved["delay_mode"])


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

    def test_greeting_bank_covers_casual_check_ins(self):
        from zhenxin.proactive import GENERIC_GREETINGS, GREETINGS

        every = [line for lines in GREETINGS.values() for line in lines]
        every.extend(GENERIC_GREETINGS)
        for phrase in ("吃饭了没？", "在干嘛呀？", "还没睡？"):
            self.assertIn(phrase, every)
        engine, _ = make_engine(**{"proactive.enabled": True})
        now = self._noon()
        text, kind = engine.proactive.compose("u1", now)
        self.assertEqual("presence", kind)
        self.assertIn(text, list(GREETINGS["noon"]) + list(GENERIC_GREETINGS))
        # Different hours on the same day should not repeat one fixed line.
        lines = {
            engine.proactive.compose("u1", datetime(2026, 10, 24, hour, 5, 0).timestamp())[0]
            for hour in (6, 9, 12, 15, 18, 21)
        }
        self.assertGreater(len(lines), 1)

    def test_every_third_message_is_plain_small_talk(self):
        engine, _ = make_engine(**{"proactive.enabled": True})
        now = self._noon()
        engine.facts.add("u1", "周五要交稿", "unclosed", importance=0.9)
        kinds = []
        for index in range(6):
            text, kind = engine.proactive.compose("u1", now + index * 60)
            kinds.append(kind)
            engine.proactive.mark_sent("u1", text, kind, now + index * 60)
        # memory, memory, greeting — repeat: greetings are not excluded when
        # memory material happens to be available.
        self.assertEqual(["memory", "memory", "presence"] * 2, kinds)

    def test_custom_template_can_use_the_greeting_slot(self):
        engine, _ = make_engine(**{"proactive.enabled": True})
        engine.cfg.proactive.template = "{greeting}"
        text, kind = engine.proactive.compose("u1", self._noon())
        self.assertEqual("presence", kind)
        self.assertTrue(text)
        engine.cfg.proactive.template = "在吗？{loop}{fact}"
        text, kind = engine.proactive.compose("u1", self._noon())
        # Unfilled placeholders must not leak to the user.
        self.assertNotIn("{loop}", text)
        self.assertNotIn("{fact}", text)


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

    def test_removed_imitation_switch_cannot_be_saved(self):
        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        api._payload_override = {"values": {"aux.style": False, "aux.rhythm": False}}
        result = asyncio.run(api.save_config())
        self.assertTrue(result["ok"])
        self.assertNotIn("aux.style", result["saved"])
        self.assertFalse(engine.rhythm.enabled)
        self.assertTrue(engine.style_learning.cfg.enabled)

    def test_prepend_is_no_longer_accepted(self):
        engine, _ = make_engine()
        api = PluginPageAPI(FakePlugin(FakeContext()), engine, {})
        with self.assertRaises(ValueError):
            api._coerce("l3.host_context_mode", "prepend")
        self.assertEqual("off", api._coerce("l3.host_context_mode", "off"))

    def test_defaults_include_new_sections(self):
        from zhenxin.page_api import schema_defaults

        defaults = schema_defaults()
        for path in (
            "memory.top_k",
            "memory.weight_semantic",
            "style_learning.max_chars",
            "split.enabled",
            "proactive.enabled",
            "l0.tail_injection",
        ):
            self.assertIn(path, defaults)


if __name__ == "__main__":
    unittest.main()
