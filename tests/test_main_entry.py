from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tests.helpers import ROOT


class _Filter:
    class EventMessageType:
        ALL = "all"

    @staticmethod
    def command(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    @staticmethod
    def event_message_type(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    @staticmethod
    def on_llm_request(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    @staticmethod
    def on_llm_response(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    @staticmethod
    def llm_tool(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    @staticmethod
    def on_astrbot_loaded(*args, **kwargs):
        def deco(fn):
            return fn

        return deco


class MainEntryTests(unittest.TestCase):
    def test_version_matches_metadata(self):
        import re

        from zhenxin import __version__

        metadata = (Path(ROOT) / "metadata.yaml").read_text(encoding="utf-8")
        match = re.search(r"^version:\s*([^\s]+)", metadata, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), __version__)

    def test_main_module_registers_star_without_astrbot(self):
        created = {}

        class Star:
            def __init__(self, context):
                self.context = context

        def register(*args, **kwargs):
            created["args"] = args

            def deco(cls):
                cls._registered = True
                return cls

            return deco

        modules = {
            "astrbot": types.ModuleType("astrbot"),
            "astrbot.api": types.ModuleType("astrbot.api"),
            "astrbot.api.event": types.ModuleType("astrbot.api.event"),
            "astrbot.api.star": types.ModuleType("astrbot.api.star"),
        }
        modules["astrbot.api.event"].filter = _Filter
        modules["astrbot.api.event"].AstrMessageEvent = object
        modules["astrbot.api.star"].Star = Star
        modules["astrbot.api.star"].Context = object
        modules["astrbot.api.star"].register = register
        old = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            spec = importlib.util.spec_from_file_location("zhenxin_main_entry", Path(ROOT) / "main.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            plugin = module.ZhenxinPlugin(
                object(), {"db_path": ":memory:", "metrics": {"jsonl_path": ""}}
            )
            self.assertTrue(created["args"])
            self.assertTrue(hasattr(plugin, "adapter"))
            self.assertTrue(hasattr(plugin, "zhenxin_status"))
            self.assertTrue(hasattr(plugin, "zhenxin_on_message"))
            self.assertTrue(hasattr(plugin, "zhenxin_on_llm_request"))
            self.assertTrue(hasattr(plugin, "zhenxin_on_llm_response"))
            self.assertTrue(hasattr(plugin, "zhenxin_recall"))
            self.assertTrue(hasattr(plugin, "zhenxin_remember"))

            class Event:
                unified_msg_origin = "webchat:test:user"

            import asyncio

            self.assertEqual(
                "已记住。", asyncio.run(plugin.zhenxin_remember(Event(), "不吃香菜"))
            )
            recalled = asyncio.run(plugin.zhenxin_recall(Event(), "我不吃什么来着"))
            self.assertIn("不吃香菜", recalled)
            plugin.engine.close()
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
