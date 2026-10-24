from __future__ import annotations

import unittest

from tests.helpers import FakeClock, make_engine, say
from zhenxin._aux.rhythm import Rhythm
from zhenxin._aux.style import MIN_SAMPLES, StyleLearner
from zhenxin.db import Database


def rhythm() -> Rhythm:
    return Rhythm(
        max_chars=72,
        followup_allowed=True,
        emoji_allowed=True,
        tone="neutral",
        warmth=0.5,
        topic_type="general",
        hint="",
    )


class StyleLearnerTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.clock = FakeClock()
        self.learner = StyleLearner(self.db, clock=self.clock)

    def feed(self, *texts: str) -> None:
        for text in texts:
            self.learner.observe("u1", text)

    def test_short_messages_tighten_reply_length_and_hint(self):
        self.feed("嗯", "好", "行", "哈哈", "好呀", "可以")
        adjusted = self.learner.adjust("u1", rhythm())
        self.assertLessEqual(adjusted.max_chars, 46)
        self.assertIn("短句", self.learner.hint("u1", adjusted))
        self.assertIn("短句", self.learner.describe("u1"))

    def test_formal_user_gets_formal_hint(self):
        self.feed(
            "请问您明天有空吗",
            "麻烦您帮我看一下",
            "谢谢您",
            "您觉得呢",
            "请问这个怎么弄",
            "您好",
        )
        self.assertIn("您", self.learner.hint("u1", self.learner.adjust("u1", rhythm())))

    def test_colloquial_user_gets_casual_hint(self):
        self.feed("好呀", "今天太累啦", "哈哈", "是嘛", "行哦", "好嘞")
        self.assertIn("口语化", self.learner.hint("u1", self.learner.adjust("u1", rhythm())))

    def test_explicit_correction_is_immediate(self):
        self.feed("今天工作还行", "晚上吃过了", "在路上", "快到了", "好的", "嗯嗯")
        self.feed("说短点，别用表情，别老问")
        adjusted = self.learner.adjust("u1", rhythm())
        self.assertLessEqual(adjusted.max_chars, 46)
        self.assertFalse(adjusted.emoji_allowed)
        self.assertFalse(adjusted.followup_allowed)
        hint = self.learner.hint("u1", adjusted)
        self.assertIn("短句", hint)
        self.assertIn("少追问", hint)

    def test_no_hint_before_enough_samples(self):
        self.feed("嗯", "好", "行", "哈哈")
        self.assertEqual("", self.learner.hint("u1", rhythm()))
        self.assertEqual(72, self.learner.adjust("u1", rhythm()).max_chars)

    def test_style_never_loosens_the_cap(self):
        self.feed(
            "今天想跟你详细说一下这件事，其实还挺复杂的，慢慢讲",
            "我更喜欢你把话说得细一点，多讲一点背景",
            "刚刚那段可以展开说说吗，我还想听更多",
            "对，多说一点，不用太短",
            "展开讲讲吧，我有时间",
            "继续多说一些细节",
        )
        adjusted = self.learner.adjust("u1", rhythm())
        self.assertLessEqual(adjusted.max_chars, 72)
        self.assertGreaterEqual(MIN_SAMPLES, 5)

    def test_switched_off_does_nothing(self):
        learner = StyleLearner(self.db, enabled=False, clock=self.clock)
        for text in ("嗯", "好", "行", "哈哈", "好呀", "可以"):
            learner.observe("u1", text)
        self.assertEqual("", learner.hint("u1", rhythm()))
        self.assertEqual(72, learner.adjust("u1", rhythm()).max_chars)


class StyleEngineTests(unittest.TestCase):
    def test_style_hint_reaches_the_ephemeral_tail(self):
        engine, clock = make_engine(**{"shortcircuit.enabled": False})
        for i in range(1, 9):
            clock.advance()
            say(engine, ["嗯", "好", "行", "哈哈", "好呀", "可以", "在吗", "睡了吗"][i - 1], i)
        plan = engine.begin_turn("u1", "在吗", round_no=9)
        self.assertIn("他的习惯", plan.ephemeral_tail)
        self.assertIn("短句", plan.ephemeral_tail)


if __name__ == "__main__":
    unittest.main()
