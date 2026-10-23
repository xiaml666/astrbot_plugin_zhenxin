from __future__ import annotations

import unittest

from tests.helpers import make_engine


class AuxTests(unittest.TestCase):
    def test_compression_keeps_emotion_and_instruction(self):
        text = (
            "今天发生了很多事情，早上开会，中午改图，下午甲方又提了新的要求，"
            "我真的很烦，感觉做不完了。你帮我看看先改哪一页比较好？"
        )
        from zhenxin._aux.compression import compress_inbound

        out = compress_inbound(text, threshold=50)
        self.assertTrue(out.compressed)
        self.assertIn("烦", out.text)
        self.assertIn("你帮我", out.text)
        self.assertLess(len(out.text), len(text))

    def test_topic_stack_depth(self):
        from zhenxin._aux.topic_stack import TopicStack

        engine, _ = make_engine()
        stack = TopicStack(engine.db, engine.scorer, max_depth=3)
        topics = ["海盐绘本交稿", "团子生病", "阿哲摄影", "周末电影", "香菜和冰美式"]
        for i, topic in enumerate(topics, 1):
            stack.observe("u1", topic)
        self.assertLessEqual(len(stack.rows("u1")), 3)

    def test_lexicon_learn_and_apply(self):
        from zhenxin._aux.lexicon import Lexicon

        engine, _ = make_engine()
        lex = Lexicon(engine.db)
        lex.learn("u1", "不叫海盐，叫盐海", 100.0)
        maps = lex.mappings("u1")
        self.assertTrue(maps)
        reply = lex.apply("u1", "盐海那本快画完了")
        self.assertIn("海盐", reply)

    def test_rhythm_postprocess(self):
        from zhenxin._aux.rhythm import Rhythm, RhythmController

        engine, _ = make_engine()
        rhythm = RhythmController(engine.db)
        r = Rhythm(max_chars=10, followup_allowed=False, emoji_allowed=False, tone="steady", warmth=0.4, topic_type="work", hint="")
        out = rhythm.postprocess("这是一句很长很长的话，还有一个问题？🙂", r)
        self.assertLessEqual(len(out), 11)
        self.assertNotIn("🙂", out)

    def test_negative_pattern_counts(self):
        from zhenxin._aux.negative import NegativePatterns

        engine, _ = make_engine()
        neg = NegativePatterns(engine.db)
        neg.record("u1", "算了，随便吧", "这是一段很长很长的回复？还有问题吗？", 100.0)
        self.assertTrue(neg.hint("u1"))


if __name__ == "__main__":
    unittest.main()

