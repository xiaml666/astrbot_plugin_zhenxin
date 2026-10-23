"""Turn the exported human scores into an A/B verdict.

Usage:
    python scripts/quality_scores_report.py --scores reports/quality_scores.json

Accepts the JSON exported by reports/quality_sheet_*.html:
    [{"id":"AB01","plugin_side":"B","score_a":4,"score_b":3,"prefer":"A"}]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arm_of(side: str) -> str:
    """plugin_side == 'A' means A is the plugin reply."""
    return "plugin" if side == "A" else "baseline"


def other(side: str) -> str:
    return "baseline" if side == "A" else "plugin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(ROOT / "reports" / "quality_scores.json"))
    args = ap.parse_args()

    rows = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    scores = {"plugin": [], "baseline": []}
    wins = {"plugin": 0, "baseline": 0, "tie": 0, "skip": 0}
    per_case = []
    for row in rows:
        side = row.get("plugin_side") or "A"
        a, b = row.get("score_a"), row.get("score_b")
        if a is not None and b is not None:
            scores["plugin"].append(a if side == "A" else b)
            scores["baseline"].append(b if side == "A" else a)
        prefer = row.get("prefer")
        if prefer in ("A", "B"):
            if (prefer == "A") == (side == "A"):
                wins["plugin"] += 1
                verdict = "plugin"
            else:
                wins["baseline"] += 1
                verdict = "baseline"
        elif prefer == "tie":
            wins["tie"] += 1
            verdict = "tie"
        else:
            wins["skip"] += 1
            verdict = "-"
        per_case.append(
            {
                "id": row.get("id"),
                "score_plugin": a if side == "A" else b,
                "score_baseline": b if side == "A" else a,
                "verdict": verdict,
            }
        )

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    p, s = mean(scores["plugin"]), mean(scores["baseline"])
    print(f"cases: {len(rows)}  scored: {len(scores['plugin'])}")
    print(f"average score   plugin {p:.2f}   baseline {s:.2f}   delta {p - s:+.2f}")
    print(f"preference      plugin {wins['plugin']}   baseline {wins['baseline']}   tie {wins['tie']}   skip {wins['skip']}")
    decided = wins["plugin"] + wins["baseline"]
    if decided:
        print(f"win rate        plugin {wins['plugin'] / decided:.0%}  (of {decided} decisive)")
    out = ROOT / "reports" / "quality_human_summary.json"
    out.write_text(json.dumps(per_case, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
