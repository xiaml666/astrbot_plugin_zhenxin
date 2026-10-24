from __future__ import annotations

import unittest

from tests.helpers import make_engine, say


class L1Tests(unittest.TestCase):
    def _seed(self, engine, clock):
        for i, text in enumerate(
            ["今天好累", "海盐那本还没画完", "团子今天一直趴我键盘上", "最近还在喝中药，苦死了"],
            1,
        ):
            clock.advance()
            say(engine, text, i)

    def test_wake_on_strong_reference_and_cooldown(self):
        engine, clock = make_engine()
        self._seed(engine, clock)
        clock.advance()
        plan = engine.begin_turn("u1", "上次说的那个中药，还要喝多久？", round_no=5)
        self.assertTrue(plan.wake)
        self.assertIn("想起", plan.ephemeral_tail)
        self.assertLessEqual(len(plan.recall), 100)
        engine.complete_turn(plan, "再喝一周吧。")
        # Same seed inside the cooldown must not wake again.
        clock.advance()
        plan2 = engine.begin_turn("u1", "上次说的那个中药，还要喝多久？", round_no=6)
        self.assertFalse(plan2.wake)
        engine.complete_turn(plan2, "嗯。")

    def test_generic_question_does_not_wake(self):
        engine, clock = make_engine()
        self._seed(engine, clock)
        clock.advance()
        plan = engine.begin_turn("u1", "有什么推荐吗？", round_no=5)
        self.assertFalse(plan.wake)
        engine.complete_turn(plan, "看你想看什么。")

    def test_open_loop_trigger_and_optional_rerank(self):
        engine, _ = make_engine(**{"l1.rerank": True})
        engine.l0.archive_cold("u1", "看医生", thin_context="未闭合：看医生")
        engine.l0.archive_cold("u1", "医生复诊", thin_context="未闭合：医生复诊")
        result = engine.l1.decide(
            "u1",
            "上次说的看医生，后来呢",
            1,
            l0_best=0.0,
            loop_touched=["看医生"],
            reranker=lambda query, candidates: [c.row_id for c in reversed(candidates)],
        )
        self.assertTrue(result.wake)
        self.assertEqual(result.trigger, "open_loop")
        self.assertLessEqual(len(result.recall), 100)

    def test_l0_sufficient_skips_wake(self):
        engine, clock = make_engine()
        self._seed(engine, clock)
        # 团子 is in the current injected set or represented in L0.
        clock.advance()
        plan = engine.begin_turn("u1", "你还记得团子叫什么吗？", round_no=5)
        self.assertFalse(plan.wake)
        engine.complete_turn(plan, "团子。")


class L2Tests(unittest.TestCase):
    def test_write_path_zero_llm_and_no_raw_cold_text(self):
        engine, clock = make_engine()
        raw = "我今天想认真说一件事：海盐绘本的第五页改到凌晨三点，甲方还是不满意，我真的有点撑不住了。"
        clock.advance()
        r = say(engine, raw, 1)
        self.assertEqual(r.trace.l2_llm_calls, 0)
        cold = engine.db.fetchall("SELECT seed_tokens,thin_context FROM cold_seeds WHERE user_key='u1'")
        self.assertTrue(cold)
        for row in cold:
            self.assertNotIn(raw, row["seed_tokens"])
            self.assertNotIn(raw, row["thin_context"])

    def test_warm_limit_physical_eviction(self):
        engine, clock = make_engine(**{"l2.warm_limit": 2, "shortcircuit.enabled": False})
        for i in range(1, 26):
            clock.advance()
            say(engine, f"记住这个代号 CODE{i}", i)
        warm = int(engine.db.scalar("SELECT COUNT(*) FROM warm_slots WHERE user_key='u1'", (), 0))
        self.assertLessEqual(warm, 2)
        cold = int(engine.db.scalar("SELECT COUNT(*) FROM cold_seeds WHERE user_key='u1'", (), 0))
        self.assertGreater(cold, 0)

    def test_open_loop_cap_and_completion(self):
        engine, clock = make_engine()
        actions = [
            "明天要交稿",
            "后天要面试",
            "周五要看医生",
            "月底要搬家",
            "下周要出差",
            "周六要开会",
            "明早要体检",
        ]
        for i, text in enumerate(actions, 1):
            clock.advance()
            engine.loops.observe("u1", text, i)
        count = int(engine.db.scalar("SELECT COUNT(*) FROM open_loops WHERE user_key='u1'", (), 0))
        self.assertLessEqual(count, 5)
        engine.loops.observe("u1", "明天交稿搞定了", 20)
        self.assertLessEqual(
            int(engine.db.scalar("SELECT COUNT(*) FROM open_loops WHERE user_key='u1'", (), 0)), 5
        )

    def test_correction_updates_the_right_ball(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        clock.advance()
        say(engine, "今天好累", 1)
        clock.advance()
        say(engine, "不是累，是烦", 2)
        seeds = [b.content_seed for b in engine.l0.active_balls("u1")]
        self.assertTrue(any("烦躁" in s for s in seeds), seeds)


if __name__ == "__main__":
    unittest.main()
