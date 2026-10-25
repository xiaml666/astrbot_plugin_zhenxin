from __future__ import annotations

import unittest

from tests.helpers import make_engine
from zhenxin._aux.directives import (
    capability_block,
    is_redo_request,
    is_voice_request,
)


class RedoDetectTests(unittest.TestCase):
    def test_detects_short_redo_instructions(self):
        for text in (
            "重来",
            "这句话就是多余的，重来",
            "重来，你有没有喜欢的人",
            "再说一遍",
            "重新说一次",
            "说完就停。重来",
        ):
            self.assertTrue(is_redo_request(text), text)

    def test_normal_messages_are_not_redo(self):
        for text in (
            "今天好累",
            "你说什么",
            "重来是什么呢",
            "a" * 80 + "重来",
            "",
        ):
            self.assertFalse(is_redo_request(text), text)


class VoiceDetectTests(unittest.TestCase):
    def test_detects_voice_requests(self):
        for text in (
            "你用语音说句我爱你",
            "语音",
            "语音语音",
            "用语音啊",
            "不像你的声音啊，再说一遍",
            "说给我听",
            "唱一句",
        ):
            self.assertTrue(is_voice_request(text), text)

    def test_plain_messages_are_not_voice_requests(self):
        for text in ("今天好累", "你在干嘛", "晚安"):
            self.assertFalse(is_voice_request(text), text)


class VoiceWindowTests(unittest.TestCase):
    def test_voice_request_opens_a_window_and_followups_stay_in_it(self):
        engine, clock = make_engine()
        engine.begin_turn("u1", "用语音说句我爱你")
        self.assertTrue(engine.voice_pending("u1"))
        # "说慢一点" has no voice keyword but is a follow-up in the window.
        plan = engine.begin_turn("u1", "说慢一点")
        self.assertIn("要的是语音回复", plan.ephemeral_tail)
        clock.advance(engine.VOICE_TTL_SECONDS + 1)
        self.assertFalse(engine.voice_pending("u1"))
        plan = engine.begin_turn("u1", "说慢一点")
        self.assertNotIn("要的是语音回复", plan.ephemeral_tail)

    def test_voice_stop_closes_the_window(self):
        engine, _ = make_engine()
        engine.begin_turn("u1", "用语音说句我爱你")
        self.assertTrue(engine.voice_pending("u1"))
        plan = engine.begin_turn("u1", "别发语音了，打字就行")
        self.assertFalse(engine.voice_pending("u1"))
        self.assertNotIn("要的是语音回复", plan.ephemeral_tail)
        plan = engine.begin_turn("u1", "在吗")
        self.assertNotIn("要的是语音回复", plan.ephemeral_tail)


class CapabilityBlockTests(unittest.TestCase):
    def test_voice_line_only_when_enabled(self):
        text = capability_block({"voice": True})
        self.assertIn("可以发语音", text)
        self.assertIn("嗓子", text)
        self.assertEqual("", capability_block({"voice": False}))
        self.assertEqual("", capability_block(None))


class EngineDirectiveTests(unittest.TestCase):
    def test_redo_hint_rides_in_the_tail(self):
        engine, _ = make_engine()
        plan = engine.begin_turn("u1", "这句话就是多余的，重来")
        self.assertIn("重做上一条回复", plan.ephemeral_tail)
        self.assertIn("只输出重做后的内容", plan.ephemeral_tail)

    def test_redo_hint_is_absent_for_normal_turns(self):
        engine, _ = make_engine()
        plan = engine.begin_turn("u1", "今天有点累")
        self.assertNotIn("重做上一条回复", plan.ephemeral_tail)

    def test_voice_capability_is_in_the_stable_prefix(self):
        engine, _ = make_engine()
        engine.set_capabilities({"voice": True})
        plan = engine.begin_turn("u1", "在吗")
        self.assertIn("可以发语音", plan.stable_prefix)
        engine.set_capabilities({"voice": False})
        plan = engine.begin_turn("u1", "在吗")
        self.assertNotIn("可以发语音", plan.stable_prefix)


if __name__ == "__main__":
    unittest.main()
