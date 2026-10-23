from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT
from zhenxin.compare import render_html, render_markdown, run_comparison, write_comparison


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(ROOT) / "data" / "sim_dialogue.jsonl"
        cls.dialogue = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cls.result = run_comparison(cls.dialogue, mode="fixed", baseline_turns=20)

    def test_plugin_is_cheaper_than_full_and_sliding_over_60_rounds(self):
        t = self.result.totals
        self.assertEqual(t["rounds"], 60)
        self.assertLess(t["plugin_total"], t["full_total"])
        self.assertLess(t["plugin_total"], t["sliding_total"])
        self.assertLess(t["last_plugin"], t["last_full"])
        self.assertGreaterEqual(t["solid_turns"], 1)
        self.assertLessEqual(t["l0_chars_max"], 150)

    def test_html_and_markdown_are_self_contained(self):
        html_text = render_html(self.result)
        self.assertIn("装插件 vs 不装插件", html_text)
        self.assertIn("上下文 token 曲线", html_text)
        self.assertIn("固化直答，本轮未调用模型", html_text)
        self.assertNotIn("http://", html_text.split("<body>")[0])  # no external CSS/JS deps
        md = render_markdown(self.result)
        self.assertIn("相对全量历史节省", md)

    def test_writer_outputs_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_comparison(self.result, tmp)
            for key in ("html", "markdown", "json"):
                self.assertTrue(paths[key].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["totals"]["rounds"], 60)
            self.assertIn("turns", payload)


if __name__ == "__main__":
    unittest.main()
