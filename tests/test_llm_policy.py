from __future__ import annotations

import unittest

from tests.helpers import make_engine
from zhenxin.llm import CallableProvider, LLMCallLedger, LLMGateway, PolicyViolation


class PolicyTests(unittest.TestCase):
    def test_forbidden_stage_raises(self):
        ledger = LLMCallLedger()
        with self.assertRaises(PolicyViolation):
            ledger.record("l2_write", "main")
        self.assertEqual(ledger.count(), 0)

    def test_allowed_main_and_small_model(self):
        ledger = LLMCallLedger()
        provider = CallableProvider(lambda messages, kwargs: "ok", "fake")
        gateway = LLMGateway(provider, provider, ledger)
        self.assertEqual(gateway.generate_main([{"role": "user", "content": "hi"}], "sys"), "ok")
        self.assertEqual(gateway.generate_solidified("hi", {"min_chars": 2}), "ok")
        self.assertEqual(ledger.count("main_generate"), 1)
        self.assertEqual(ledger.count("solidify_small"), 1)

    def test_l2_never_calls_gateway(self):
        engine, _ = make_engine()
        engine.l2.write("u1", "今天好累", "好。", 1)
        self.assertEqual(engine.ledger.count("l2_write"), 0)
        self.assertEqual(engine.ledger.count(), 0)

    def test_style_screen_stage_is_allowed(self):
        ledger = LLMCallLedger()
        main = CallableProvider(lambda messages, kwargs: "unused", "main")
        decision = CallableProvider(lambda messages, kwargs: '["对方希望回复短一点"]', "small")
        gateway = LLMGateway(main, decision, ledger)
        raw = gateway.screen("待审核：以后说话短一点")
        self.assertIn("短一点", raw)
        self.assertEqual(ledger.count("style_screen"), 1)
        self.assertEqual(ledger.count("main_generate"), 0)

    def test_engine_screen_returns_none_without_a_model(self):
        engine, _ = make_engine()
        self.assertIsNone(engine.screen_knowledge("pref", ["以后说话短一点"]))

    def test_engine_screen_parses_the_model_output(self):
        engine, _ = make_engine()
        provider = CallableProvider(
            lambda messages, kwargs: '["对方希望回复短一点"]', "small"
        )
        engine.gateway = LLMGateway(provider, provider, engine.ledger)
        self.assertEqual(
            ["对方希望回复短一点"],
            engine.screen_knowledge("pref", ["以后说话短一点"]),
        )


if __name__ == "__main__":
    unittest.main()
