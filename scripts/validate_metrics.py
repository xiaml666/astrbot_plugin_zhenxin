#!/usr/bin/env python
"""Validate a real Zhenxin SQLite database after a shadow/live run.

Usage:
    python scripts/validate_metrics.py --db data/zhenxin.sqlite3
    python scripts/validate_metrics.py --db data/zhenxin.sqlite3 --json

It never calls a model.  "actual" usage fields come from AstrBot's
LLMResponse.usage; when the host/provider does not expose them, estimated
injection metrics are still available but real cost cannot be computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhenxin import Config  # noqa: E402
from zhenxin.db import Database  # noqa: E402
from zhenxin.metrics import MetricsRecorder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "zhenxin.sqlite3"))
    parser.add_argument("--user", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on critical failures")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.exists():
        print(f"database not found: {path}")
        return 2
    cfg = Config()
    cfg.db_path = str(path)
    db = Database(path)
    report = MetricsRecorder(db, cfg.metrics).report(args.user)
    db.close()
    if not report.get("rounds"):
        print("no metrics rows yet")
        return 2

    checks = {
        "L2 LLM calls == 0": report.get("l2_llm_calls", 0) == 0,
        "L0 <= 150 chars": (report.get("l0_chars_max") or 0) <= 150,
        "prefix_last20 >= 95%": (report.get("prefix_stable_last20") or 0) >= 0.95,
        "wake_rate <= 20%": (report.get("wake_rate") or 0) <= 0.20,
        "actual usage captured": (report.get("actual_usage_rounds") or 0) > 0,
    }
    critical = {
        "L2 LLM calls == 0": checks["L2 LLM calls == 0"],
        "L0 <= 150 chars": checks["L0 <= 150 chars"],
        "prefix_last20 >= 95%": checks["prefix_last20 >= 95%"] if report.get("rounds", 0) >= 20 else True,
    }

    if args.json:
        print(json.dumps({"report": report, "checks": checks}, ensure_ascii=False, indent=2))
    else:
        print(f"rounds={report['rounds']}  user={args.user or '*'}  db={path}")
        print(f"wake_rate={report.get('wake_rate', 0):.1%}  last10={report.get('last10_wake_rate', 0):.1%}")
        print(f"prefix_last20={report.get('prefix_stable_last20', 0):.1%}  prefix_overall={report.get('prefix_stable_ratio', 0):.1%}")
        print(
            f"L0_max={report.get('l0_chars_max', 0)}  solid_hits={report.get('solid_hits', 0)}"
            f" ({report.get('solid_hit_rate', 0):.1%})  L2_llm={report.get('l2_llm_calls', 0)}"
        )
        if (report.get("solid_hit_rate") or 0) > 0.20:
            print("   ⚠️ 固化命中率 >20%，模板化风险偏高；请先做盲测或关掉 shortcircuit。")
        print(
            "actual_usage_rounds="
            f"{report.get('actual_usage_rounds', 0)}/{report.get('rounds', 0)}"
            f"  input={report.get('actual_total_input_tokens', 0):,}"
            f" (noncached={report.get('actual_input_tokens', 0):,}, cached={report.get('actual_cached_tokens', 0):,})"
            f"  output={report.get('actual_output_tokens', 0):,}"
            f"  cost={report.get('actual_currency', 'CNY')} {report.get('actual_cost', 0):.4f}"
        )
        print()
        for name, ok in checks.items():
            print(f"{'✅' if ok else '❌'}  {name}")
        if not checks["actual usage captured"]:
            print("   ⚠️  provider did not expose usage; real cost unavailable (estimates still logged).")

    if args.strict and not all(critical.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
