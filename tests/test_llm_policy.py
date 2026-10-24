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


if __name__ == "__main__":
    unittest.main()
