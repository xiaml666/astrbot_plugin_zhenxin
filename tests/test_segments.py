from __future__ import annotations

import unittest

from tests.helpers import make_engine
from zhenxin._aux.segments import (
    calculate_delay,
    merge_segments,
    segment_text,
    split_chain,
)


class Plain:
    def __init__(self, text: str) -> None:
        self.text = text


class Reply:
    def __init__(self, text: str = "") -> None:
        self.text = text


class Image:
    def __init__(self, file: str = "x.png") -> None:
        self.file = file


class At:
    def __init__(self, qq: str = "1") -> None:
        self.qq = qq


def _join(segments) -> str:
    return "".join(
        part.text for segment in segments for part in segment if isinstance(part, Plain)
    )


class SplitChainTests(unittest.TestCase):
    def test_long_text_splits_and_keeps_all_text(self):
        text = "今天确实挺累的，先别硬撑。喝口水缓一缓吧。晚点再说也行。"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertGreaterEqual(len(segments), 2)
        self.assertLessEqual(len(segments), 5)
        self.assertEqual(text, _join(segments))

    def test_line_break_keeps_a_short_second_line_as_its_own_bubble(self):
        # The reported case: two lines, second one is 9 chars (<10).  A line
        # break is a deliberate bubble boundary, so it must not be merged back.
        text = "还没呢，一直躺着等你说句话\n哥哥你呢，吃了没呀"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertEqual(2, len(segments))
        self.assertEqual(text.replace("\n", ""), _join(segments).replace("\n", ""))

    def test_short_reply_stays_one_bubble(self):
        segments = split_chain(
            [Plain("好。我在。")], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertEqual(1, len(segments))

    def test_reply_quote_stays_on_the_first_bubble(self):
        # This is the old bug: any Reply component made the whole chain skip
        # segmentation.  AstrBot normally quotes the user message.
        chain = [Reply("1"), Plain("第一句。第二句。第三句。")]
        segments = split_chain(
            chain, plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertGreaterEqual(len(segments), 2)
        self.assertIsInstance(segments[0][0], Reply)
        self.assertEqual("第一句。第二句。第三句。", _join(segments))

    def test_image_is_sent_as_its_own_bubble(self):
        chain = [Plain("第一句。第二句。"), Image()]
        segments = split_chain(
            chain, plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertTrue(any(len(seg) == 1 and isinstance(seg[0], Image) for seg in segments))
        self.assertEqual("第一句。第二句。", _join(segments))

    def test_at_follows_next_text(self):
        chain = [Plain("第一句。"), At("2"), Plain("第二句。")]
        segments = split_chain(
            chain, plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertEqual("第一句。第二句。", _join(segments))
        self.assertTrue(any(isinstance(part, At) for segment in segments for part in segment))

    def test_code_fence_is_not_cut(self):
        text = "看这段代码。\n```\na = 1\nb = 2\n```\n就这些。"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertTrue(
            any("```\na = 1\nb = 2\n```" in _join([segment]) for segment in segments)
        )
        self.assertEqual(text.replace("\n", ""), _join(segments).replace("\n", ""))

    def test_quoted_sentence_is_not_cut(self):
        text = "他说：“好。明天见。”然后就走了。我也该走了。"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertEqual(text, _join(segments))
        self.assertTrue(
            any("“好。明天见。”" in _join([segment]) for segment in segments)
        )

    def test_max_segments_merges_the_tail_without_losing_text(self):
        text = "第一句。第二句。第三句。第四句。第五句。"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=2, min_segment_chars=2
        )
        self.assertEqual(2, len(segments))
        self.assertEqual(text, _join(segments))

    def test_markdown_table_is_not_cut(self):
        text = "看表。\n| a | b |\n| - | - |\n| 1 | 2 |\n完。"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertEqual(text.replace("\n", ""), _join(segments).replace("\n", ""))
        self.assertTrue(any("| 1 | 2 |" in _join([segment]) for segment in segments))

    def test_merge_segments_flattens_and_keeps_order(self):
        merged = merge_segments(
            [[Plain("a"), Reply()], [Plain("b")]], Plain
        )
        self.assertEqual(3, len(merged))
        self.assertEqual("a", merged[0].text)
        self.assertIsInstance(merged[1], Reply)
        self.assertEqual("b", merged[2].text)

    def test_delay_modes(self):
        self.assertAlmostEqual(0.5 + 0.1 * 10, calculate_delay("natural", "x" * 10))
        self.assertAlmostEqual(0.3, calculate_delay("fast", "x"))
        self.assertAlmostEqual(2.5, calculate_delay("slow", "x"))
        self.assertLessEqual(calculate_delay("natural", "x" * 200), 5.0)

    def test_engine_defaults_enable_segmentation(self):
        engine, _ = make_engine()
        self.assertTrue(engine.cfg.split.enabled)
        self.assertEqual(5, engine.cfg.split.max_segments)
        self.assertEqual(10, engine.cfg.split.min_segment_chars)
        self.assertEqual("natural", engine.cfg.split.delay_mode)


if __name__ == "__main__":
    unittest.main()
