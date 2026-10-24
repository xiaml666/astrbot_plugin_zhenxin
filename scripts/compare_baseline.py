#!/usr/bin/env python
"""Generate a visual A/B report: with Zhenxin vs without Zhenxin.

Usage:
    python scripts/compare_baseline.py
    python scripts/compare_baseline.py --mode adaptive
    python scripts/compare_baseline.py --limit 20 --baseline-turns 8

Outputs:
    reports/comparison.html   <- open this in a browser
    reports/comparison.md
    reports/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhenxin.compare import run_comparison, write_comparison  # noqa: E402


def load_dialogue(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["round"]))
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "sim_dialogue.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    parser.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--baseline-turns", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dialogue = load_dialogue(Path(args.data), args.limit)
    result = run_comparison(dialogue, mode=args.mode, baseline_turns=args.baseline_turns)
    paths = write_comparison(result, args.out)
    t = result.totals
    print(f"rounds: {t['rounds']}  mode: {args.mode}")
    print(f"plugin total:  {t['plugin_total']:>8,} tok  avg {t['plugin_avg']:>7,.1f}")
    print(f"full total:    {t['full_total']:>8,} tok  avg {t['full_avg']:>7,.1f}")
    print(f"sliding total: {t['sliding_total']:>8,} tok  avg {t['sliding_avg']:>7,.1f}")
    print(f"saved vs full: {t['saved_vs_full']:.1%}   vs sliding: {t['saved_vs_sliding']:.1%}")
    print(f"solid turns: {t['solid_turns']}  wake turns: {t['wake_turns']}  L0 max: {t['l0_chars_max']}")
    print(f"open: {paths['html'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
