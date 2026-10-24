"""E4: what each arm answers when asked to recall something from round 5-20."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\plugin_data\astrbot_plugin_zhenxin\zhenxin.sqlite3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "reports" / "ab_e4.json"))
    ap.add_argument("--script", default=str(ROOT / "data" / "validation_probe.jsonl"))
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    script = [
        json.loads(x)
        for x in Path(args.script).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    probes = {int(item["round"]): item["text"] for item in script if item.get("probe")}

    import sqlite3

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def turns(session: str) -> dict[int, dict[str, str]]:
        key = f"webchat:FriendMessage:webchat!astrbot!{session}"
        out: dict[int, dict[str, str]] = {}
        for row in conn.execute(
            "SELECT turn_no, role, content FROM turns WHERE user_key=? ORDER BY id", (key,)
        ):
            out.setdefault(int(row["turn_no"]), {})[row["role"]] = row["content"]
        return out

    shadow = turns(run["shadow_session"])
    live = turns(run["live_session"])

    print(f"rounds: {run['rounds']}; probes: {sorted(probes)}\n")
    for n, question in sorted(probes.items()):
        # probes were inserted, so their turn numbers are not exactly n; match by text
        def find(data):
            for rnd, roles in data.items():
                if roles.get("user", "").startswith(question[:8]):
                    return rnd, roles.get("assistant", "")
            return None, ""

        sr, sa = find(shadow)
        lr, la = find(live)
        print(f"== 第{n}轮提问：{question}")
        print(f"   基线（第{sr}轮）: {sa.strip()[:160]}")
        print(f"   真心（第{lr}轮）: {la.strip()[:160]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
