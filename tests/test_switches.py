from __future__ import annotations

import unittest

from tests.helpers import make_engine, say


class SwitchTests(unittest.TestCase):
    def test_all_modules_off_still_runs(self):
        engine, clock = make_engine(
            **{
                "l0.enabled": False,
                "shortcircuit.enabled": False,
                "l1.enabled": False,
                "l2.enabled": False,
                "l3.enabled": False,
                "aux.topic_stack": False,
                "aux.compression": False,
                "aux.rhythm": False,
                "aux.lexicon": False,
                "aux.open_loops": False,
                "aux.negative_patterns": False,
                "metrics.enabled": False,
            }
        )
        for i, text in enumerate(["今天好累", "上周说的那个事", "又加班"], 1):
            clock.advance()
            r = say(engine, text, i)
            self.assertFalse(r.trace.wake)
            self.assertFalse(r.trace.shortcircuit_hit)
            self.assertEqual(r.trace.l0_chars, 0)
            self.assertEqual(r.trace.l2_llm_calls, 0)
            self.assertTrue(r.reply)

    def test_l2_off_means_no_memory_rows(self):
        engine, clock = make_engine(**{"l2.enabled": False, "metrics.enabled": False})
        clock.advance()
        say(engine, "今天好累", 1)
        hot = engine.l0.active_balls("u1")
        self.assertEqual(hot, [])
        self.assertEqual(int(engine.db.scalar("SELECT COUNT(*) FROM cold_seeds", (), 0)), 0)


if __name__ == "__main__":
    unittest.main()
