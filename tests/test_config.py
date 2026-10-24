from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import Config  # noqa: F401
from zhenxin import _mini_yaml


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = Config()
        self.assertTrue(cfg.l0.enabled)
        self.assertTrue(cfg.l3.cache_aware_pruning)
        self.assertFalse(cfg.shadow_mode)
        self.assertEqual(cfg.l0.inject, 3)
        self.assertEqual(cfg.l1.wake_cooldown_rounds, 8)

    def test_yaml_and_nested_env(self):
        text = """
db_path: ":memory:"
shadow_mode: true
l0:
  enabled: true
  inject: 2
  hysteresis: 1.5
l1:
  enabled: false
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            path.write_text(text, encoding="utf-8")
            old = os.environ.get("ZHENXIN__L1__ENABLED")
            os.environ["ZHENXIN__L1__ENABLED"] = "true"
            os.environ["ZHENXIN__DB_PATH"] = ":memory:"
            try:
                cfg = Config.load(path)
            finally:
                if old is None:
                    os.environ.pop("ZHENXIN__L1__ENABLED", None)
                else:
                    os.environ["ZHENXIN__L1__ENABLED"] = old
                os.environ.pop("ZHENXIN__DB_PATH", None)
        self.assertTrue(cfg.shadow_mode)
        self.assertEqual(cfg.l0.inject, 2)
        self.assertTrue(cfg.l1.enabled)
        self.assertEqual(cfg.db_path, ":memory:")

    def test_mini_yaml_lists_and_objects(self):
        text = """
items:
  - name: a
    count: 2
  - name: b
    count: 3
flags: [true, false]
persona: "你好"
"""
        data = _mini_yaml.loads(text)
        self.assertEqual(data["items"][1]["count"], 3)
        self.assertEqual(data["flags"], [True, False])
        self.assertEqual(data["persona"], "你好")


if __name__ == "__main__":
    unittest.main()
