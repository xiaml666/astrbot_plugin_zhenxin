"""Aggregate the blind judge output into per-arm scores and a verdict."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "reports" / "quality_judge.json").read_text(encoding="utf-8"))
PAIRS = {
    json.loads(x)["id"]: json.loads(x)
    for x in (ROOT / "data" / "quality_pairs_local.jsonl").read_text(encoding="utf-8").splitlines()
    if x.strip()
}
DIMS = ("fact", "ref", "halluc", "human", "fit")


def arm_scores(entry: dict, best: dict) -> dict[str, dict]:
    """Return {baseline: scores, plugin: scores} for one judging call."""
    if best is None:
        return {}
    a, b = best.get("A") or {}, best.get("B") or {}
    if entry["swapped"]:
        return {"plugin": a, "baseline": b}
    return {"baseline": a, "plugin": b}


def main() -> None:
    # Only judge pairs that survived the data hygiene filter (round 1 excluded).
    keep = set(PAIRS)
    rows = [r for r in ROWS if r["id"] in keep]
    totals = {"baseline": {d: [] for d in DIMS}, "plugin": {d: [] for d in DIMS}}
    decisive = {"baseline": 0, "plugin": 0, "tie": 0}
    b_wins = 0
    calls = 0
    for row in rows:
        per_pair = {"baseline": [], "plugin": []}
        for order in row["orders"]:
            s = arm_scores(order, order["scores"])
            if not s:
                continue
            calls += 1
            for arm in ("baseline", "plugin"):
                vals = [s[arm].get(d) for d in DIMS]
                if any(v is None for v in vals):
                    continue
                for d, v in zip(DIMS, vals):
                    totals[arm][d].append(float(v))
            per_pair["baseline"].append(sum(s["baseline"].get(d, 0) for d in DIMS) / len(DIMS))
            per_pair["plugin"].append(sum(s["plugin"].get(d, 0) for d in DIMS) / len(DIMS))
            winner = (order["scores"] or {}).get("winner", "tie")
            # translate A/B into arms
            if winner == "tie":
                pass
            elif (winner == "A") != bool(order["swapped"]):
                b_wins += 1  # winner is baseline
        if len(per_pair["baseline"]) == 2:
            bs, ps = per_pair["baseline"]
            if abs(bs - ps) < 0.2:
                decisive["tie"] += 1
            elif bs > ps:
                decisive["baseline"] += 1
            else:
                decisive["plugin"] += 1

    print(f"judged calls: {calls}  (position-B wins {b_wins}/{calls} = {b_wins / max(1, calls):.0%})")
    print("\narm        " + "  ".join(f"{d:>6}" for d in DIMS) + "    mean")
    for arm in ("baseline", "plugin"):
        means = [sum(totals[arm][d]) / max(1, len(totals[arm][d])) for d in DIMS]
        print(f"{arm:<10} " + "  ".join(f"{m:>6.2f}" for m in means) + f"   {sum(means)/len(means):.2f}")
    print(f"\ndecisive pairs (both orders agree): {decisive}")
    n = sum(decisive.values())
    if n:
        print(
            f"baseline {decisive['baseline']}/{n}, plugin {decisive['plugin']}/{n}, "
            f"tie {decisive['tie']}/{n}"
        )


if __name__ == "__main__":
    main()
