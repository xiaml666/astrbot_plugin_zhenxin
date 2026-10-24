from __future__ import annotations

import unittest

from tests.helpers import make_engine, say


class CacheLayoutTests(unittest.TestCase):
    """The two properties the prompt cache depends on.

    Provider prompt caches are prefix caches: the first differing token makes
    everything after it uncached.  So (a) a clue change must not touch the
    system prompt, and (b) the recent window must stay byte-identical for
    several rounds at a time.
    """

    def test_l0_change_keeps_system_prefix_byte_identical(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        say(engine, "今天好累", 1)
        first = engine.begin_turn("u1", "嗯", round_no=2)
        engine.complete_turn(first, "好。")
        self.assertTrue(first.l0_block)
        self.assertIn("【长期线索·L0】", first.ephemeral_tail)
        self.assertNotIn("【长期线索·L0】", first.stable_prefix)

        members = engine.db.get_state("u1", "l0_members", []) or []
        self.assertTrue(members)
        engine.l0.update_ball("u1", int(members[0]), content_seed="换成了一条全新的线索")
        clock.advance()
        second = engine.begin_turn("u1", "嗯", round_no=3)
        self.assertNotEqual(second.l0_block, first.l0_block)
        self.assertIn("换成了一条全新的线索", second.ephemeral_tail)
        # The whole point: the prefix cache survives a clue change.
        self.assertEqual(second.stable_prefix, first.stable_prefix)
        self.assertEqual(second.prefix_hash, first.prefix_hash)

    def test_l0_can_be_kept_in_prefix_for_comparison(self):
        engine, _ = make_engine(**{"shortcircuit.enabled": False, "l0.tail_injection": False})
        say(engine, "今天好累", 1)
        plan = engine.begin_turn("u1", "嗯", round_no=2)
        self.assertTrue(plan.l0_block)
        self.assertIn("【长期线索·L0】", plan.stable_prefix)
        self.assertNotIn("【长期线索·L0】", plan.ephemeral_tail)

    def test_window_grows_append_only_between_cuts(self):
        engine, clock = make_engine(
            **{"shortcircuit.enabled": False, "l3.recent_turns": 4, "l3.window_grow_turns": 4}
        )
        windows: list[list[str]] = []
        for i in range(1, 17):
            clock.advance()
            plan = engine.begin_turn("u1", f"第{i}句话", round_no=i)
            windows.append([m["content"] for m in plan.recent_messages])
            engine.complete_turn(plan, f"回{i}")

        stable_steps = 0
        for prev, cur in zip(windows, windows[1:]):
            if cur[: len(prev)] == prev:
                stable_steps += 1
            else:
                # A cut must be a real cut, not a one-message slide.
                self.assertLess(len(cur), len(prev))
        # Grow 4 turns then cut back: most rounds leave the prefix untouched.
        self.assertGreaterEqual(stable_steps, len(windows) // 2)
        self.assertGreater(max(len(w) for w in windows), 8)

    def test_char_budget_cuts_below_high_water(self):
        engine, clock = make_engine(
            **{
                "shortcircuit.enabled": False,
                "l3.recent_turns": 10,
                "l3.recent_max_chars": 200,
                "l3.recent_char_low_water": 0.5,
            }
        )
        sizes: list[int] = []
        for i in range(1, 26):
            clock.advance()
            say(engine, "这是一句四十个字符左右的话用来撑满预算的测试文本内容", i)
            plan = engine.begin_turn("u1", "现在呢", round_no=100 + i)
            chars = sum(len(m["content"]) for m in plan.recent_messages)
            sizes.append(chars)
            self.assertLessEqual(chars, 200)
            engine.complete_turn(plan, "好。")
        cut_at = [i for i, (prev, cur) in enumerate(zip(sizes, sizes[1:])) if cur < prev]
        self.assertTrue(cut_at, "budget should have been enforced at least once")
        # Coarse eviction: a cut must buy several rounds of a stable prefix.
        gaps = [b - a for a, b in zip(cut_at, cut_at[1:])]
        if gaps:
            self.assertGreaterEqual(min(gaps), 3)

    def test_tail_is_not_persisted_into_history(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 6):
            clock.advance()
            say(engine, f"第{i}句普通对话", i)
        rows = engine.db.fetchall("SELECT content FROM turns WHERE user_key='u1'")
        joined = "\n".join(r["content"] for r in rows)
        self.assertNotIn("【长期线索·L0】", joined)


if __name__ == "__main__":
    unittest.main()
