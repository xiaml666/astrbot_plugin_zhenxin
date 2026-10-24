from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from zhenxin import Config, ZhenxinEngine  # noqa: E402


class FakeClock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float = 3600.0) -> None:
        self.now += seconds


def make_engine(**overrides):
    cfg = Config(db_path=":memory:")
    cfg.metrics.jsonl_path = ""
    for dotted, value in overrides.items():
        target = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    clock = FakeClock()
    return ZhenxinEngine(cfg, clock=clock), clock


def say(engine, text: str, round_no: int, reply: str = "好。", shadow: bool | None = None):
    return engine.handle_turn("u1", text, lambda plan: reply, round_no=round_no, shadow=shadow)
