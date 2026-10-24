"""Prefix-cache simulation: how much of each request survives a cache hit.

Real providers cache *prefixes*: the longest shared prefix is billed at the
cached rate, the rest at the full input rate.  This script replays the bundled
60-round dialogue through four layout variants and reports, per variant:

* total request tokens (estimate),
* tokens served from the cached prefix,
* tokens billed as fresh input,
* price at the configured input / cached-input prices,
* how many rounds changed the system prefix or moved the window start.

It is a simulation of the *layout*, not a bill: token counts are char-based
estimates and the provider's real chunking is approximated by 64-token blocks.

    python scripts/cache_prefix_sim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from zhenxin import Config, ZhenxinEngine  # noqa: E402
from zhenxin.layers.l3 import estimate_tokens  # noqa: E402

DIALOGUE = ROOT / "data" / "sim_dialogue.jsonl"
CHUNK = 64

# AstrBot appends its own text to req.system_prompt *after* the plugin hooks
# (skills prompt, tool definitions, safety notices...).  Live measurement on
# 4.29.1: the system prompt was ~2.8k tokens up to the clue block and ~7.9k
# tokens after it.  That suffix is what a clue change used to invalidate.
REGIMES = {
    "plain host": 0,
    "AstrBot +7.9k system suffix": 7900,
}


def variants() -> dict[str, dict]:
    return {
        "l0_in_prefix + slide": {"l0.tail_injection": False, "l3.window_grow_turns": 0},
        "l0_in_prefix + grow": {"l0.tail_injection": False, "l3.window_grow_turns": 4},
        "l0_in_tail   + slide": {"l0.tail_injection": True, "l3.window_grow_turns": 0},
        "l0_in_tail   + grow": {"l0.tail_injection": True, "l3.window_grow_turns": 4},
    }


def host_suffix(tokens: int) -> str:
    # ASCII filler: estimate_tokens() counts 4 ASCII chars per token.
    return ("tool definition line for the host agent. " * (tokens // 9 + 1))[: tokens * 4] if tokens else ""


def wire(plan, suffix: str = "") -> str:
    """Approximate the request as the provider sees it, in wire order."""
    parts = [plan.stable_prefix]
    if suffix:
        parts.append(suffix)
    for msg in plan.recent_messages:
        parts.append(f"[{msg['role']}] {msg['content']}")
    parts.append(f"[user] {plan.prompt}")
    if plan.ephemeral_tail:
        parts.append(plan.ephemeral_tail)
    return "\n".join(parts)


def common_prefix_chars(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def run(name: str, overrides: dict, suffix: str = "") -> dict:
    cfg = Config(db_path=":memory:")
    cfg.metrics.jsonl_path = ""
    for dotted, value in overrides.items():
        target = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    engine = ZhenxinEngine(cfg)

    rounds = [json.loads(line) for line in DIALOGUE.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = ""
    total = cached = 0
    prefix_changes = window_moves = 0
    last_prefix = None
    last_window: list[str] = []

    for item in rounds:
        plan = engine.begin_turn("sim", item["text"], round_no=int(item["round"]))
        text = wire(plan, suffix)
        tokens = estimate_tokens(text)
        total += tokens
        shared = common_prefix_chars(previous, text) if previous else 0
        shared_tokens = estimate_tokens(text[:shared]) if shared else 0
        shared_tokens -= shared_tokens % CHUNK
        cached += shared_tokens
        if last_prefix is not None and plan.stable_prefix != last_prefix:
            prefix_changes += 1
        if last_window and [m["content"] for m in plan.recent_messages][: len(last_window)] != last_window:
            window_moves += 1
        last_prefix = plan.stable_prefix
        last_window = [m["content"] for m in plan.recent_messages]
        previous = text
        engine.complete_turn(plan, "好。")

    fresh = total - cached
    price = fresh / 1e6 * cfg.metrics.input_price_per_million + cached / 1e6 * cfg.metrics.cached_input_price_per_million
    return {
        "variant": name,
        "total_tokens": total,
        "cached_tokens": cached,
        "fresh_tokens": fresh,
        "cached_ratio": cached / total if total else 0.0,
        "price_cny": price,
        "prefix_changes": prefix_changes,
        "window_moves": window_moves,
    }


def table(rows: list[dict], title: str) -> None:
    print(f"\n== {title} ==")
    header = f"{'variant':<24}{'total':>9}{'cached':>9}{'fresh':>9}{'hit%':>7}{'price':>9}{'prefΔ':>7}{'winΔ':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant']:<24}{row['total_tokens']:>9,}{row['cached_tokens']:>9,}"
            f"{row['fresh_tokens']:>9,}{row['cached_ratio']:>7.1%}{row['price_cny']:>9.4f}"
            f"{row['prefix_changes']:>7}{row['window_moves']:>6}"
        )
    baseline = rows[0]
    print()
    for row in rows[1:]:
        print(
            f"{row['variant']}: fresh input {row['fresh_tokens'] / baseline['fresh_tokens'] - 1:+.1%}, "
            f"price {row['price_cny'] / baseline['price_cny'] - 1:+.1%}"
        )


def main() -> int:
    out = ROOT / "reports" / "cache_prefix_sim.json"
    results: dict[str, list[dict]] = {}
    for regime, tokens in REGIMES.items():
        rows = [run(name, overrides, host_suffix(tokens)) for name, overrides in variants().items()]
        results[regime] = rows
        table(rows, f"{regime} (host suffix {tokens} tokens)")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
