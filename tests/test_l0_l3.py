from __future__ import annotations

import hashlib
import unittest

from tests.helpers import make_engine, say


class L0Tests(unittest.TestCase):
    def test_ten_slots_and_injection_limit(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        facts = [
            "海盐那本还没画完",
            "团子今天一直趴我键盘上",
            "最近在喝中药调理",
            "我不吃香菜",
            "阿哲是摄影师",
            "冰美式救不了我",
            "周五要交稿",
            "昨晚没睡好",
            "甲方又改需求",
            "周末想看电影",
            "妈妈周末过来",
            "线稿还要重画",
        ]
        for i, text in enumerate(facts, 1):
            clock.advance()
            say(engine, text, i)
        rows = engine.db.fetchall("SELECT * FROM hot_slots WHERE user_key='u1'")
        self.assertEqual(len(rows), 10)
        block, slots = engine.l0.current_block("u1", 20)
        self.assertLessEqual(len(block), 150)
        self.assertGreaterEqual(len(slots), 1)
        self.assertLessEqual(len(slots), 3)
        lines = [line for line in block.splitlines() if line.startswith("- ")]
        self.assertLessEqual(len(lines), 3)
        # The model must never see all ten balls.
        self.assertLess(len(lines), 10)

    def test_prefix_byte_stable_under_brightness_only(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        say(engine, "今天好累", 1)
        first = engine.begin_turn("u1", "嗯", round_no=2)
        engine.complete_turn(first, "好。")
        hashes = [first.prefix_hash]
        for i in range(3, 13):
            clock.advance()
            plan = engine.begin_turn("u1", "嗯", round_no=i)
            self.assertEqual(plan.l0_block, first.l0_block)
            self.assertEqual(plan.prefix_hash, first.prefix_hash)
            hashes.append(plan.prefix_hash)
            engine.complete_turn(plan, "好。")
        self.assertEqual(len(set(hashes)), 1)

    def test_new_fact_does_not_evict_injected_ball(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False})
        facts = [f"记住代号 C{i}" for i in range(12)]
        for fact in facts:
            engine.l0.insert_candidate("u1", fact, "fact")
        block1, members1 = engine.l0.current_block("u1", 1)
        self.assertTrue(block1)
        engine.l0.insert_candidate("u1", "记住代号 NEW", "fact")
        block2, members2 = engine.l0.current_block("u1", 2)
        self.assertEqual(members1, members2)
        self.assertEqual(block1, block2)

    def test_decay_reduces_brightness(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        say(engine, "今天好累", 1)
        before = engine.l0.active_balls("u1")[0].brightness
        clock.advance()
        say(engine, "窗外有只鸟", 2)
        after = engine.l0.active_balls("u1")[0].brightness
        self.assertLess(after, before)


class L3Tests(unittest.TestCase):
    def test_prefix_is_untouched_and_ephemeral_not_persisted(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        say(engine, "今天好累", 1)
        for i in range(2, 22):
            clock.advance()
            say(engine, f"这是第{i}轮普通对话", i)
        rows = engine.db.fetchall("SELECT role,content FROM turns WHERE user_key='u1'")
        joined = "\n".join(r["content"] for r in rows)
        self.assertNotIn("本轮提示", joined)
        self.assertNotIn("〔节奏：", joined)
        prefix_hash = engine.begin_turn("u1", "继续", round_no=22).prefix_hash
        self.assertEqual(prefix_hash, engine.db.get_state("u1", "last_prefix_hash"))

    def test_recent_window_char_budget(self):
        engine, clock = make_engine(
            **{"shortcircuit.enabled": False, "l3.recent_max_chars": 120, "l3.recent_turns": 10}
        )
        for i in range(1, 11):
            clock.advance()
            say(engine, "很长的一段话" * 8, i)
        plan = engine.begin_turn("u1", "现在呢", round_no=11)
        self.assertLessEqual(sum(len(m["content"]) for m in plan.recent_messages), 120)
        self.assertTrue(plan.recent_messages)

    def test_plain_mode_keeps_previous_ephemeral(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False, "l3.cache_aware_pruning": False})
        say(engine, "今天好累", 1)
        engine.db.set_state("u1", "last_ephemeral", "PREV_HINT")
        plan = engine.begin_turn("u1", "嗯", round_no=2)
        self.assertIn("PREV_HINT", plan.prompt)
        self.assertIn("PREV_HINT", plan.prompt)
        # With cache-aware pruning on, previous ephemeral is stripped.
        engine2, clock2 = make_engine(**{"shortcircuit.enabled": False})
        say(engine2, "今天好累", 1)
        engine2.db.set_state("u1", "last_ephemeral", "PREV_HINT")
        plan2 = engine2.begin_turn("u1", "嗯", round_no=2)
        self.assertNotIn("PREV_HINT", plan2.prompt)


if __name__ == "__main__":
    unittest.main()
