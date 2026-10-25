from __future__ import annotations

import unittest

from tests.helpers import make_engine
from zhenxin._aux.segments import split_reply


class SplitReplyTests(unittest.TestCase):
    def test_short_reply_stays_one_bubble(self):
        self.assertEqual(["好，我在。"], split_reply("好，我在。", min_chars=40))

    def test_sentences_become_separate_bubbles(self):
        text = "今天确实挺累的，先别硬撑。喝口水缓一缓吧。晚点再说也行。"
        parts = split_reply(text, max_segments=3, min_chars=10, min_segment_chars=4)
        self.assertEqual(3, len(parts))
        self.assertEqual(text, "".join(parts))
        self.assertTrue(parts[0].endswith("。"))

    def test_tiny_fragments_are_merged(self):
        parts = split_reply(
            "嗯。今天真的挺累的，先歇会儿吧。好。",
            max_segments=4,
            min_chars=5,
            min_segment_chars=8,
        )
        self.assertLessEqual(len(parts), 2)
        self.assertIn("今天真的挺累的", "".join(parts))

    def test_max_segments_caps_without_losing_text(self):
        text = "第一句。第二句。第三句。第四句。第五句。"
        parts = split_reply(text, max_segments=2, min_chars=5, min_segment_chars=2)
        self.assertEqual(2, len(parts))
        self.assertEqual(text, "".join(parts))

    def test_quotes_and_brackets_are_never_cut(self):
        text = "他说：“好。明天见。”然后就走了。我也该走了。"
        parts = split_reply(text, max_segments=4, min_chars=5, min_segment_chars=2)
        joined = "".join(parts)
        self.assertEqual(text, joined)
        self.assertIn("“好。明天见。”", joined)
        # The quoted sentence must stay in one bubble.
        self.assertTrue(any("“好。明天见。”" in part for part in parts))

    def test_code_fence_is_not_split(self):
        text = "看这段代码。\n```\na = 1\nb = 2\n```\n就这些。"
        parts = split_reply(text, max_segments=4, min_chars=5, min_segment_chars=2)
        self.assertTrue(any("```\na = 1\nb = 2\n```" in part for part in parts))

    def test_engine_default_is_off(self):
        engine, _ = make_engine()
        self.assertFalse(engine.cfg.split.enabled)
        self.assertEqual(3, engine.cfg.split.max_segments)
        self.assertEqual(40, engine.cfg.split.min_chars)
        self.assertAlmostEqual(0.6, engine.cfg.split.delay_s)


if __name__ == "__main__":
    unittest.main()
