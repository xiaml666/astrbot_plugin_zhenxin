from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers import ROOT
from scripts.run_shadow_sim import acceptance, load_dialogue, run


class PipelineTests(unittest.TestCase):
    def test_60_round_acceptance(self):
        dialogue = load_dialogue(Path(ROOT) / "data" / "sim_dialogue.jsonl")
        _, live, _ = run("live", dialogue)
        _, shadow, _ = run("shadow", dialogue)
        checks = acceptance(live, shadow)
        failed = [name for name, ok in checks.items() if not ok]
        self.assertEqual(failed, [], f"failed: {failed}; live={live}; shadow={shadow}")
        self.assertGreaterEqual(shadow["turns_alive"], 50)
        self.assertLessEqual(shadow["wake_rate_at_50"], 0.20)
        self.assertGreaterEqual(shadow["prefix_stable_last20"], 0.95)
        self.assertEqual(shadow["l2_llm_calls"], 0)
        self.assertLessEqual(shadow["l0_chars_max"], 150)
        self.assertGreater(live["solid_hits"], 0)
        self.assertGreater(live["solidified"]["degrade_events"], 0)


if __name__ == "__main__":
    unittest.main()
