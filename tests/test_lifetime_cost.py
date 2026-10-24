from __future__ import annotations

import unittest

from scripts.lifetime_cost import estimate


class LifetimeCostTests(unittest.TestCase):
    def test_finite_baseline_and_bounded_plugin(self):
        row = estimate(
            context=8192,
            daily_turns=50,
            turn_tokens=60,
            system_tokens=50,
            plugin_tokens=700,
            output_tokens=80,
            reserve_output=4000,
            solid_rate=0.05,
            input_price=2.0,
            output_price=8.0,
        )
        self.assertEqual(row["baseline_turns"], 69)
        self.assertTrue(row["plugin_fits"])
        self.assertGreater(row["saved_ratio"], 0.5)
        self.assertLess(row["plugin_cost"], row["baseline_cost"])

    def test_larger_window_has_more_savings(self):
        kwargs = dict(
            daily_turns=50,
            turn_tokens=60,
            system_tokens=50,
            plugin_tokens=700,
            output_tokens=80,
            reserve_output=4000,
            solid_rate=0.05,
            input_price=2.0,
            output_price=8.0,
        )
        small = estimate(context=8192, **kwargs)
        large = estimate(context=131072, **kwargs)
        self.assertGreater(large["saved_ratio"], small["saved_ratio"])
        self.assertGreater(large["baseline_days"], small["baseline_days"])


if __name__ == "__main__":
    unittest.main()
