from __future__ import annotations

import unittest

from tests.helpers import make_engine
from zhenxin._aux import zero_value
from zhenxin.layers.l3 import BOUNDARY


def add_fact(engine, text, *, kind="fact", pool="cycle", importance=0.9):
    engine.db.execute(
        "INSERT INTO facts(user_key,text,kind,importance,created_at,pool)"
        " VALUES(?,?,?,?,?,?)",
        ("u1", text, kind, importance, engine.clock(), pool),
    )


class SessionFreshnessTests(unittest.TestCase):
    def test_old_session_turns_are_not_injected_after_a_long_silence(self):
        engine, clock = make_engine()
        old = clock.now - 4 * 3600
        engine.db.append_turn("u1", 1, "user", "昨晚聊过的旧话题", old)
        engine.db.append_turn("u1", 1, "assistant", "昨晚的回复", old)

        plan = engine.begin_turn("u1", "早")
        contents = [m["content"] for m in plan.recent_messages]
        self.assertNotIn("昨晚聊过的旧话题", contents)
        self.assertNotIn("昨晚的回复", contents)

    def test_proactive_opener_survives_the_new_session_boundary(self):
        engine, clock = make_engine()
        old = clock.now - 6 * 3600
        engine.db.append_turn("u1", 1, "user", "昨晚聊过的旧话题", old)
        engine.record_proactive_turn("u1", "在干嘛呀？")
        engine.db.execute(
            "UPDATE turns SET created_at=? WHERE content='在干嘛呀？'", (old + 3600,)
        )
        plan = engine.begin_turn("u1", "准备睡了")
        contents = [m["content"] for m in plan.recent_messages]
        self.assertIn("在干嘛呀？", contents)
        self.assertNotIn("昨晚聊过的旧话题", contents)

    def test_same_session_turns_are_still_injected(self):
        engine, clock = make_engine()
        engine.begin_turn("u1", "在吗")
        engine.db.append_turn("u1", 2, "user", "刚才说的", clock.now)
        engine.db.append_turn("u1", 2, "assistant", "刚才回的", clock.now)

        plan = engine.begin_turn("u1", "然后呢")
        contents = [m["content"] for m in plan.recent_messages]
        self.assertIn("刚才说的", contents)
        self.assertIn("刚才回的", contents)


class NameQualityTests(unittest.TestCase):
    def test_fragment_names_are_rejected(self):
        self.assertTrue(zero_value.name_is_valid("蓝心心"))
        self.assertTrue(zero_value.name_is_valid("老大"))
        self.assertFalse(zero_value.name_is_valid("笨鱼就叫"))
        self.assertFalse(zero_value.name_is_valid("你小肥鱼好不好"))

    def test_invalid_identity_facts_are_skipped(self):
        engine, _ = make_engine()
        add_fact(engine, "你叫“笨鱼就叫”", kind="assistant_name", importance=1.0)
        add_fact(engine, "你叫“蓝心心”", kind="assistant_name", importance=0.9)
        self.assertEqual("你叫“蓝心心”", engine.facts.assistant_name("u1"))

    def test_nickname_rule_follows_the_assistant_name(self):
        self.assertIn("先直接回答", BOUNDARY)
        self.assertIn("自己说过的话要算数", BOUNDARY)
        engine, _ = make_engine()
        add_fact(engine, "你叫“蓝心心”", kind="assistant_name", importance=1.0)
        plan = engine.begin_turn("u1", "晚安")
        self.assertIn("别自创昵称", plan.ephemeral_tail)
        self.assertIn("别用你自己的名字去叫对方", plan.ephemeral_tail)


class MemoryCleanupTests(unittest.TestCase):
    def test_low_value_facts_are_rejected_at_write_time(self):
        engine, _ = make_engine()
        self.assertFalse(engine.facts.add("u1", "提到“有啊”"))
        self.assertFalse(engine.facts.add("u1", "爱你", "habit"))
        self.assertFalse(engine.facts.add("u1", "你叫“笨鱼就叫”", "assistant_name"))
        self.assertTrue(engine.facts.add("u1", "周五要交稿"))
        self.assertEqual(
            1,
            int(
                engine.db.scalar(
                    "SELECT COUNT(*) FROM facts WHERE user_key='u1'", (), 0
                )
            ),
        )

    def test_fragment_and_noise_facts_are_purged(self):
        engine, _ = make_engine()
        add_fact(engine, "提到“有啊”")
        add_fact(engine, "相关：reset")
        add_fact(engine, "爱你", kind="habit")
        add_fact(engine, "你叫“笨鱼就叫”", kind="assistant_name")
        add_fact(engine, "周五要交稿", importance=1.0)
        add_fact(engine, "称呼我“老大”", kind="address")

        removed = engine.facts.repair_low_quality("u1")
        self.assertEqual(4, removed)
        remaining = {
            str(row["text"])
            for row in engine.db.fetchall(
                "SELECT text FROM facts WHERE user_key=?", ("u1",)
            )
        }
        self.assertEqual({"周五要交稿", "称呼我“老大”"}, remaining)


if __name__ == "__main__":
    unittest.main()
