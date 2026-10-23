#!/usr/bin/env python
"""Run this inside the AstrBot virtualenv BEFORE enabling Zhenxin live.

It checks the concrete API surface the adapter relies on, imports the plugin
entry point, and runs a local no-network smoke test.  It does not call an LLM.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _has_fields(obj, names):
    if is_dataclass(obj):
        present = {f.name for f in fields(obj)}
    else:
        present = set(getattr(obj, "__annotations__", {}) or {})
        for name in names:
            if hasattr(obj, name):
                present.add(name)
    return {name: name in present for name in names}


def main() -> int:
    print(f"plugin: {ROOT}")
    try:
        import astrbot  # type: ignore
    except Exception as exc:
        print(f"❌ AstrBot is not importable in this interpreter: {exc}")
        print("   Run this script with the same Python that starts AstrBot.")
        return 2

    results: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool, detail: str = "", critical: bool = True) -> None:
        results.append((name, "✅" if ok else ("❌" if critical else "⚠️"), detail))

    try:
        from astrbot.api.event import filter  # type: ignore

        for name in ("event_message_type", "on_llm_request", "on_llm_response"):
            add(f"filter.{name}", hasattr(filter, name), "adapter hook")
        add(
            "EventMessageType.ALL",
            hasattr(getattr(filter, "EventMessageType", object), "ALL"),
            "message listener coverage",
        )
    except Exception as exc:
        add("astrbot.api.event.filter", False, str(exc))

    try:
        from astrbot.core.agent.message import TextPart  # type: ignore

        add("TextPart", True, "ephemeral content part")
        add("TextPart.mark_as_temp", hasattr(TextPart, "mark_as_temp"), "4.24+; fallback exists", critical=False)
    except Exception as exc:
        add("TextPart", False, f"{exc} (fallback appends to req.prompt)", critical=False)

    try:
        from astrbot.api.provider import ProviderRequest  # type: ignore

        fields_check = _has_fields(
            ProviderRequest,
            ["prompt", "system_prompt", "contexts", "extra_user_content_parts"],
        )
        for name, ok in fields_check.items():
            add(f"ProviderRequest.{name}", ok)
    except Exception as exc:
        add("ProviderRequest", False, str(exc))

    try:
        from astrbot.api.provider import LLMResponse  # type: ignore

        add("LLMResponse", True)
        add("LLMResponse.usage", hasattr(LLMResponse, "usage") or "usage" in getattr(LLMResponse, "__annotations__", {}), "real token usage", critical=False)
        try:
            from astrbot.core.provider.entities import TokenUsage  # type: ignore

            usage_fields = _has_fields(TokenUsage, ["input_other", "input_cached", "output"])
            for name, ok in usage_fields.items():
                add(f"TokenUsage.{name}", ok, "actual usage capture", critical=False)
        except Exception as exc:
            add("TokenUsage", False, str(exc), critical=False)
    except Exception as exc:
        add("LLMResponse", False, str(exc))

    try:
        meta = ROOT / "metadata.yaml"
        schema = ROOT / "_conf_schema.json"
        add("metadata.yaml", meta.exists())
        if schema.exists():
            json.loads(schema.read_text(encoding="utf-8"))
            add("_conf_schema.json", True)
        else:
            add("_conf_schema.json", False)
    except Exception as exc:
        add("metadata/schema", False, str(exc))

    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import main as plugin_main  # type: ignore

        add("main.ZhenxinPlugin", hasattr(plugin_main, "ZhenxinPlugin"))
    except Exception as exc:
        add("main import", False, str(exc))

    # Local no-network pipeline smoke test (not AstrBot's real request path).
    try:
        from zhenxin import Config, ZhenxinEngine

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.load()
            cfg.db_path = str(Path(tmp) / "preflight.sqlite3")
            cfg.metrics.jsonl_path = ""
            engine = ZhenxinEngine(cfg)
            for i, text in enumerate(["今天好累", "嗯", "又加班"], 1):
                engine.handle_turn("preflight", text, lambda plan: "好。", round_no=i)
            stats = engine.stats("preflight")
            add("local pipeline", stats.get("rounds") == 3)
            add("L2 zero LLM", stats.get("l2_llm_calls") == 0)
            add("L0 <= 150 chars", (stats.get("l0_chars_max") or 0) <= 150)
            engine.close()
    except Exception as exc:
        add("local pipeline", False, str(exc))

    print()
    for name, state, detail in results:
        print(f"{state}  {name:<34} {detail}")
    critical_failed = [r for r in results if r[1] == "❌"]
    print()
    if critical_failed:
        print(f"FAILED: {len(critical_failed)} critical checks")
        return 1
    print("PASS: AstrBot adapter surface and local pipeline are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
