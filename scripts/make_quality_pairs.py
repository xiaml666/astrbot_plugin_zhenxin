"""Build blind quality pairs from a shadow-vs-live A/B stored by the plugin.

The plugin's own `turns` table keeps every round for both sessions, which is
cleaner than AstrBot's conversation record (replace mode trims it).

    python scripts/make_quality_pairs.py \
        --shadow-session <uuid> --live-session <uuid> --prefix E1 --out data/pairs_e1.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = glob.glob(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\plugin_data\*\zhenxin.sqlite3")
DEFAULT_DB = DEFAULT_DB[0] if DEFAULT_DB else str(ROOT / "data" / "zhenxin.sqlite3")


def key_for(session: str) -> str:
    return f"webchat:FriendMessage:webchat!astrbot!{session}"


def rounds(conn, user_key: str):
    out: dict[int, dict[str, str]] = {}
    history: dict[int, list[dict]] = {}
    for row in conn.execute(
        "SELECT turn_no, role, content FROM turns WHERE user_key=? ORDER BY id", (user_key,)
    ):
        n = int(row["turn_no"])
        out.setdefault(n, {})[row["role"]] = row["content"]
        history.setdefault(n, []).append({"role": row["role"], "content": row["content"]})
    return out, history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--shadow-session", required=True)
    ap.add_argument("--live-session", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--min-round", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "data" / "quality_pairs_local.jsonl"))
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    s, s_hist = rounds(conn, key_for(args.shadow_session))
    l, _ = rounds(conn, key_for(args.live_session))

    pairs = []
    for n in sorted(set(s) & set(l)):
        if n < args.min_round:
            continue
        if "user" not in s[n] or "assistant" not in s[n] or "assistant" not in l[n]:
            continue
        context = []
        for rnd in range(max(1, n - 2), n):
            for item in s_hist.get(rnd, []):
                who = "用户" if item["role"] == "user" else "朋友"
                context.append(f"{who}：{item['content']}")
        pairs.append(
            {
                "id": f"{args.prefix}{n:02d}",
                "category": "dialogue",
                "context": "\n".join(context[-4:]),
                "user": s[n]["user"],
                "baseline_reply": s[n]["assistant"],
                "plugin_reply": l[n]["assistant"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n", encoding="utf-8")
    avg_s = sum(len(p["baseline_reply"]) for p in pairs) / max(1, len(pairs))
    avg_l = sum(len(p["plugin_reply"]) for p in pairs) / max(1, len(pairs))
    print(f"wrote {out}: {len(pairs)} pairs, avg chars shadow {avg_s:.1f} live {avg_l:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
