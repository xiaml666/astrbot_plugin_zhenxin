from __future__ import annotations

import unittest

from tests.helpers import make_engine, say


class ShortCircuitTests(unittest.TestCase):
    def test_repeat_hits_then_negative_degrades(self):
        engine, clock = make_engine()
        results = []
        for i, text in enumerate(
            [
                "今天好累",          # create path, conf 2.0
                "海盐那本还没画完",  # neutral
                "今天好累",          # repeat +1 -> 3.0, hit
                "不是累，是烦",      # negative -> 2.0, neg 1
                "今天好累",          # repeat -> 3.0 but neg 1, no hit
                "别老说这种模板话",  # delayed meta negative -> neg 2, degrade
                "今天好累",          # degraded, no hit
            ],
            1,
        ):
            clock.advance()
            results.append(say(engine, text, i))
        self.assertTrue(results[2].trace.shortcircuit_hit)
        self.assertEqual(results[2].trace.main_llm_calls, 0)
        self.assertGreaterEqual(engine.shortcircuit.stats("u1")["hits"], 1)
        self.assertGreaterEqual(engine.shortcircuit.stats("u1")["degrade_events"], 1)
        self.assertFalse(results[6].trace.shortcircuit_hit)
        path = engine.shortcircuit.get_path("u1", "re:tired")
        self.assertIsNotNone(path)
        self.assertGreaterEqual(path.neg_streak, 2)
        self.assertGreater(path.degraded_until, 0)

    def test_long_planning_statement_never_shortcircuits(self):
        engine, clock = make_engine()
        for i in range(1, 8):
            clock.advance()
            r = say(engine, "周五交稿，周六想睡一天", i)
            self.assertFalse(r.trace.shortcircuit_hit)

    def test_template_has_no_llm_call(self):
        engine, clock = make_engine()
        for i, text in enumerate(["今天好累", "海盐那本还没画完", "今天好累"], 1):
            clock.advance()
            r = say(engine, text, i)
        self.assertTrue(r.trace.shortcircuit_hit)
        self.assertEqual(r.trace.main_llm_calls, 0)
        self.assertEqual(r.trace.decision_llm_calls, 0)
        self.assertTrue(r.reply)
        self.assertLessEqual(len(r.reply), 80)


if __name__ == "__main__":
    unittest.main()
