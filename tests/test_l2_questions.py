from __future__ import annotations

import unittest

from tests.helpers import make_engine, say
from zhenxin.layers.l2 import Distiller, _is_question
from zhenxin.layers.l3 import BOUNDARY, ContextBuilder
from zhenxin.config import L3Config


def seeds(text: str) -> list[str]:
    return [c.seed for c in Distiller().distill(text)]


class QuestionIsNotAFactTests(unittest.TestCase):
    """A question may carry emotion, but it must never become a stored fact."""

    def test_question_about_a_person_creates_no_role_fact(self):
        self.assertTrue(_is_question("上次说的阿哲，是不是搞摄影的"))
        self.assertFalse([s for s in seeds("上次说的阿哲，是不是搞摄影的") if "阿哲" in s])

    def test_interrogative_preference_is_dropped(self):
        self.assertFalse([s for s in seeds("你还记得我不吃什么吗") if "不吃什么" in s])

    def test_plain_statement_still_creates_the_preference(self):
        self.assertIn("不吃香菜", seeds("我不吃香菜，闻到就想吐"))

    def test_missing_entity_question_leaves_no_clue(self):
        self.assertEqual([], [s for s in seeds("阿哲是谁来着？") if "阿哲" in s])


class PrefixBoundaryTests(unittest.TestCase):
    def test_boundary_line_is_part_of_the_stable_prefix(self):
        builder = ContextBuilder(None, L3Config())
        prefix = builder.build_prefix("SYS", "人格", "")
        self.assertIn(BOUNDARY, prefix)
        self.assertEqual(prefix, builder.build_prefix("SYS", "人格", ""))

    def test_facts_outrank_transient_emotion(self):
        from zhenxin.config import L0Config

        bonus = L0Config().type_bonus
        self.assertGreater(bonus["habit"], bonus["emotion"])
        self.assertGreaterEqual(bonus["fact"], bonus["emotion"])


class ShadowArmIsNotRewrittenTests(unittest.TestCase):
    def test_shadow_reply_is_stored_verbatim(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False, "shadow_mode": True})
        clock.advance()
        long_reply = "这是一段明显超过节奏字数控件的回复，用来确认影子模式不会改写宿主已经发出的文本。" * 2
        result = say(engine, "今天好累", 1, reply=long_reply, shadow=True)
        self.assertEqual(result.reply, long_reply)


if __name__ == "__main__":
    unittest.main()
