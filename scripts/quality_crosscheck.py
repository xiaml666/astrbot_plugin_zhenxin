"""Cross-check human scores against the machine judge and inspect the losses."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN = json.loads((ROOT / "reports" / "quality_scores.json").read_text(encoding="utf-8"))
JUDGE = {r["id"]: r for r in json.loads((ROOT / "reports" / "quality_judge.json").read_text(encoding="utf-8"))}
PAIRS = {
    json.loads(x)["id"]: json.loads(x)
    for x in (ROOT / "data" / "quality_pairs_local.jsonl").read_text(encoding="utf-8").splitlines()
    if x.strip()
}


def human_verdict(row):
    pref = row.get("prefer")
    side = row["plugin_side"]
    if pref == "tie":
        return "tie"
    if pref in ("A", "B"):
        return "plugin" if pref == side else "baseline"
    return None


def machine_verdict(row):
    votes = []
    for order in JUDGE.get(row["id"], {}).get("orders", []):
        s = order.get("scores") or {}
        w = s.get("winner")
        if w not in ("A", "B"):
            continue
        votes.append("plugin" if (w == "A") == (row["plugin_side"] == "A") else "baseline")
    if len(votes) == 2 and votes[0] == votes[1]:
        return votes[0]
    return "tie"


def arm_scores(row):
    a, b = row.get("score_a"), row.get("score_b")
    if a is None or b is None:
        return None, None
    return (a, b) if row["plugin_side"] == "A" else (b, a)


agree = {"same": 0, "diff": 0}
rows = []
for row in HUMAN:
    h = human_verdict(row)
    m = machine_verdict(row)
    p, s = arm_scores(row)
    rows.append({"id": row["id"], "human": h, "machine": m, "plugin": p, "baseline": s})
    if h and m:
        agree["same" if h == m else "diff"] += 1

print(f"human-vs-judge agreement: {agree['same']}/{agree['same'] + agree['diff']} "
      f"({agree['same'] / max(1, agree['same'] + agree['diff']):.0%})")

print("\n== human verdicts ==")
counts = {}
for r in rows:
    counts[r["human"]] = counts.get(r["human"], 0) + 1
print(counts)

lens = {"plugin": [], "baseline": []}
for r in rows:
    pair = PAIRS[r["id"]]
    lens["plugin"].append(len(pair["plugin_reply"]))
    lens["baseline"].append(len(pair["baseline_reply"]))
print(f"\navg reply chars: plugin {sum(lens['plugin']) / len(lens['plugin']):.1f} "
      f"baseline {sum(lens['baseline']) / len(lens['baseline']):.1f}")

print("\n== baseline wins (human) ==")
for r in rows:
    if r["human"] != "baseline":
        continue
    pair = PAIRS[r["id"]]
    print(f"\n[{r['id']}] score plugin={r['plugin']} baseline={r['baseline']}  Q: {pair['user']}")
    print(f"  插件: {pair['plugin_reply'][:90]}")
    print(f"  基线: {pair['baseline_reply'][:90]}")

print("\n== plugin wins where judge said otherwise ==")
for r in rows:
    if r["human"] == "plugin" and r["machine"] == "baseline":
        pair = PAIRS[r["id"]]
        print(f"\n[{r['id']}]  Q: {pair['user']}")
        print(f"  插件: {pair['plugin_reply'][:90]}")
        print(f"  基线: {pair['baseline_reply'][:90]}")
