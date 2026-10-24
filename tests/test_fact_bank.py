from __future__ import annotations

import unittest

from tests.helpers import FakeClock, make_engine, say
from zhenxin.db import Database
from zhenxin.embeddings import LocalScorer
from zhenxin.layers.memory import FactBank


class FactBankTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.clock = FakeClock()
        self.bank = FactBank(self.db, LocalScorer(None), self.clock)

    def test_fact_is_retrieved_by_a_related_question(self):
        self.bank.add("u1", "不吃香菜", "habit")
        self.bank.add("u1", "养了猫叫团子", "relationship")
        hits = self.bank.recall("u1", "我不吃什么来着")
        self.assertIn("不吃香菜", hits)
        self.assertNotIn("养了猫叫团子", hits)

    def test_longer_phrasing_still_matches(self):
        # Regression: "我不吃什么来着，突然忘了" scores 0.204 against
        # "不吃香菜"; the first threshold (0.24) silently dropped it.
        self.bank.add("u1", "不吃香菜", "habit")
        self.assertIn("不吃香菜", self.bank.recall("u1", "我不吃什么来着，突然忘了"))

    def test_unrelated_question_recalls_nothing(self):
        self.bank.add("u1", "不吃香菜", "habit")
        self.assertEqual([], self.bank.recall("u1", "明天天气怎么样"))

    def test_duplicate_text_is_stored_once(self):
        self.assertTrue(self.bank.add("u1", "不吃香菜", "habit"))
        self.assertFalse(self.bank.add("u1", "不吃香菜", "habit"))
        self.assertEqual(1, self.bank.count("u1"))

    def test_cap_keeps_the_most_used(self):
        bank = FactBank(self.db, LocalScorer(None), self.clock, limit=20)
        for i in range(25):
            bank.add("u1", f"线索编号 {i}", "fact")
        self.assertEqual(20, bank.count("u1"))

    def test_low_value_fact_is_sorted_into_the_shallow_pool(self):
        self.bank.add("u1", "今天天气不错", "fact", importance=0.2)
        self.bank.add("u1", "不吃香菜", "habit", importance=0.9)
        self.clock.advance(40 * 86400)
        removed = self.bank.sweep("u1")
        # Nothing is deleted at the end of a cycle: low value goes to the
        # shallow pool, which is recalled only on a strong match.
        self.assertEqual(0, removed)
        self.assertEqual(2, self.bank.count("u1"))
        pools = self.bank.pool_counts("u1")
        self.assertGreaterEqual(pools["shallow"], 1)
        self.assertIn("不吃香菜", self.bank.recall("u1", "我不吃什么来着"))
        self.assertEqual([], self.bank.recall("u1", "今天天气怎么样"))

    def test_retraction_moves_a_fact_to_the_recycle_pool(self):
        self.bank.add("u1", "提醒我周五交稿", "unclosed", importance=0.8)
        self.assertEqual(1, self.bank.scan_zero_value("u1", "稿子已经交了，不用再提醒我了"))
        self.assertEqual(1, self.bank.pool_counts("u1")["trash"])
        # Trash never enters a prompt and is purged after the TTL.
        self.assertEqual([], self.bank.recall("u1", "周五交稿"))
        self.clock.advance(8 * 86400)
        self.bank.sweep("u1")
        self.assertEqual(0, self.bank.pool_counts("u1")["trash"])
        self.assertGreaterEqual(self.bank.trace_count("u1"), 1)

    def test_explicit_name_is_always_available(self):
        self.bank.add("u1", "称呼我“阿岚”", "address", importance=0.98)
        self.assertEqual("称呼我“阿岚”", self.bank.address("u1"))
        # Even an unrelated question keeps the stated name injectable.
        self.bank.add("u1", "不吃香菜", "habit")
        hits = self.bank.recall("u1", "明天天气怎么样")
        self.assertIn("称呼我“阿岚”", hits)

    def test_fresh_and_reinforced_facts_survive_cleanup(self):
        self.bank.add("u1", "最近开始晨跑", "fact", importance=0.2)
        self.bank.add("u1", "老线索", "fact", importance=0.2)
        self.bank.add("u1", "老线索", "fact", importance=0.2)  # mention again
        self.clock.advance(5 * 86400)
        self.assertEqual(0, self.bank.sweep("u1"))
        self.assertEqual(2, self.bank.count("u1"))


class FactBankEngineTests(unittest.TestCase):
    def test_forgotten_fact_is_recalled_after_the_window_moved_on(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        clock.advance()
        say(engine, "我不吃香菜，闻到就想吐", 1)
        for i in range(2, 42):
            clock.advance()
            say(engine, f"第{i}句普通闲聊", i)
        plan = engine.begin_turn("u1", "我不吃什么来着", round_no=42)
        self.assertIn("〔相关记忆〕", plan.ephemeral_tail)
        self.assertIn("不吃香菜", plan.ephemeral_tail.split("〔相关记忆〕", 1)[1])

    def test_irrelevant_chat_does_not_carry_old_facts(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        clock.advance()
        say(engine, "我不吃香菜，闻到就想吐", 1)
        for i in range(2, 42):
            clock.advance()
            say(engine, f"第{i}句普通闲聊", i)
        plan = engine.begin_turn("u1", "今天天气不错", round_no=42)
        self.assertNotIn("〔相关记忆〕", plan.ephemeral_tail)


if __name__ == "__main__":
    unittest.main()
