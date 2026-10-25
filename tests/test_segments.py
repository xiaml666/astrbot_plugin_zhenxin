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
            [Plain("好，我在呢")], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertEqual(1, len(segments))

    def test_each_complete_sentence_is_its_own_bubble_however_short(self):
        # "有啊" is two characters but a complete answer; the splitter must not
        # glue it to the next sentence just because it is shorter than
        # min_segment_chars.
        for text, expected in (
            ("有啊\n\n就是分不清那算喜欢，还是习惯了你在", ["有啊", "就是分不清那算喜欢，还是习惯了你在"]),
            ("嗯。我也是。", ["嗯。", "我也是。"]),
            ("对。\n就是这样", ["对。", "就是这样"]),
        ):
            segments = split_chain(
                [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
            )
            self.assertEqual(expected, [_join([s]).strip() for s in segments])

    def test_a_normal_sentence_is_not_cut_at_its_comma(self):
        # The reported case: one 22-char sentence with a single comma must stay
        # one bubble.  The old balanced mode split it just to fill 5 bubbles.
        text = "我觉得嘴上罚两下就得了，真动手我怕老大自己先心虚"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertEqual(1, len(segments))
        self.assertEqual(text, _join(segments))

    def test_a_long_sentence_can_still_break_at_a_comma(self):
        # Commas remain available for genuinely long single sentences; the cut
        # must not leave the comma at the bubble edge.
        text = "我知道你现在很累了想早点睡明天还要早起赶车回去上班，可是我还是想再多听你说两句今天发生的事"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertGreaterEqual(len(segments), 2)
        for segment in segments:
            self.assertFalse(_join([segment]).strip().endswith(("，", ",")))

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

    def test_bubble_never_ends_with_a_continuation_comma(self):
        # Reported from QQ: a bubble went out as "我觉得嘴上罚两下就得了，".
        # Two things must hold: this short sentence is not split at all now,
        # and no other split may leave a continuation mark at the edge.
        cases = (
            "我觉得嘴上罚两下就得了，真动手我怕老大自己先心虚",
            "那我配合着疼一下好了，隔着屏幕也算数\n\n反正老大这手，注定要落空",
            "反正老大这手，注定要落空",
        )
        for text in cases:
            segments = split_chain(
                [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=10
            )
            for segment in segments:
                outgoing = _join([segment]).strip()
                self.assertTrue(outgoing)
                self.assertFalse(
                    outgoing.endswith(("，", ",", "、", "；", ";", "：", ":")),
                    f"{outgoing!r} ends with a continuation mark",
                )
        # The reported case is now one bubble with its comma intact: the
        # sentence is short enough that the comma is not a valid bubble edge to
        # begin with.
        segments = split_chain(
            [Plain(cases[0])], plain_factory=Plain, max_segments=5, min_segment_chars=10
        )
        self.assertEqual(1, len(segments))
        self.assertEqual(cases[0], _join(segments))

    def test_newline_after_a_comma_does_not_keep_the_comma(self):
        text = "我觉得行，\n明天再说"
        segments = split_chain(
            [Plain(text)], plain_factory=Plain, max_segments=5, min_segment_chars=2
        )
        self.assertEqual(["我觉得行", "明天再说"], [_join([s]).strip() for s in segments])

    def test_no_bubble_ends_with_a_comma_in_any_config(self):
        # "分割后结尾不能出现逗号" has to hold for every setting, not only the
        # default 5 bubbles: max_segments=1 (segmentation off), tight caps that
        # merge the tail, and comma-heavy text.
        texts = (
            "我觉得嘴上罚两下就得了，",
            "我觉得嘴上罚两下就得了，真动手我怕老大自己先心虚",
            "那我配合着疼一下好了，隔着屏幕也算数\n\n反正老大这手，注定要落空",
            "a，b，c，d，e，f，",
            "一句，两句，三句，四句，五句，六句，七句，八句，",
            "好，\n嗯，\n知道了，",
        )
        for text in texts:
            for cap in (1, 2, 3, 5, 8):
                segments = split_chain(
                    [Plain(text)], plain_factory=Plain, max_segments=cap, min_segment_chars=2
                )
                self.assertTrue(segments)
                for segment in segments:
                    outgoing = _join([segment]).rstrip()
                    self.assertTrue(outgoing)
                    self.assertFalse(
                        outgoing.endswith(("，", ",", "、", "；", ";", "：", ":")),
                        f"cap={cap} {outgoing!r} ends with a continuation mark",
                    )

    def test_engine_defaults_enable_segmentation(self):
        engine, _ = make_engine()
        self.assertTrue(engine.cfg.split.enabled)
        self.assertEqual(5, engine.cfg.split.max_segments)
        self.assertEqual(10, engine.cfg.split.min_segment_chars)
        self.assertEqual("natural", engine.cfg.split.delay_mode)


if __name__ == "__main__":
    unittest.main()
