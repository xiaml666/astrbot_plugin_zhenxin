from __future__ import annotations

import unittest

from scripts.quality_sheet import render_sheet


class QualitySheetTests(unittest.TestCase):
    def test_blind_sheet_hides_plugin_side_and_escapes(self):
        html = render_sheet(
            [
                {
                    "id": "HE01",
                    "category": "short_ack",
                    "user": "嗯<script>",
                    "baseline_reply": "嗯，我在。",
                    "plugin_reply": "嗯。",
                }
            ],
            seed=1,
        )
        self.assertIn("盲测评分表", html)
        self.assertIn("data-plugin-side=", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("导出", html)
        # Click-only scoring: visible numbers, one click records, state survives
        # a refresh, and export never depends on a file download.
        self.assertIn('class="num"', html)
        self.assertIn("localStorage", html)
        self.assertIn('id="modal"', html)
        self.assertIn("score_a", html)


if __name__ == "__main__":
    unittest.main()
