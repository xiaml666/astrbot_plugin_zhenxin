"""E3 analysis: repetition and drift over 100 rounds."""

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(text: str) -> str:
    return "".join(ch for ch in text if ch.strip() and ch not in "。，！？…、~～ ")


def duplicates(replies: list[str]) -> tuple[int, list[str]]:
    seen = Counter(norm(r) for r in replies if r.strip())
    dups = [text for text, n in seen.items() if n > 1]
    return sum(n - 1 for n in seen.values() if n > 1), dups


def near_duplicates(replies: list[str], threshold: float = 0.9) -> int:
    count = 0
    normed = [norm(r) for r in replies]
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            if not normed[i] or not normed[j]:
                continue
            if difflib.SequenceMatcher(None, normed[i], normed[j]).ratio() >= threshold:
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "reports" / "ab_e3.json"))
    args = ap.parse_args()
    data = json.loads(Path(args.run).read_text(encoding="utf-8"))
    print("sessions:", data["shadow_session"], data["live_session"])

    rows = {}
    import sqlite3

    db = Path(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\plugin_data\astrbot_plugin_zhenxin\zhenxin.sqlite3")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    for label, session in (("shadow", data["shadow_session"]), ("live", data["live_session"])):
        key = f"webchat:FriendMessage:webchat!astrbot!{session}"
        turns = list(
            conn.execute(
                "SELECT turn_no, role, content FROM turns WHERE user_key=? ORDER BY id", (key,)
            )
        )
        assistant = [(int(r["turn_no"]), r["content"]) for r in turns if r["role"] == "assistant"]
        first = [c for n, c in assistant if n <= 50]
        second = [c for n, c in assistant if n > 50]
        metric = list(
            conn.execute(
                "SELECT SUM(input_tokens_actual) ic, SUM(cached_tokens_actual) cc,"
                " SUM(output_tokens_actual) oc, COUNT(*) n FROM metrics WHERE user_key=?",
                (key,),
            )
        )[0]
        ic, cc, oc = metric["ic"] or 0, metric["cc"] or 0, metric["oc"] or 0
        price = ic * 2 / 1e6 + cc * 0.5 / 1e6 + oc * 8 / 1e6
        rows[label] = dict(
            n=len(assistant),
            first=first,
            second=second,
            ic=ic,
            cc=cc,
            oc=oc,
            price=price,
        )

    for label in ("shadow", "live"):
        r = rows[label]
        d_all, _ = duplicates(r["first"] + r["second"])
        d_late, _ = duplicates(r["second"])
        near = near_duplicates(r["second"])
        avg1 = sum(len(x) for x in r["first"]) / max(1, len(r["first"]))
        avg2 = sum(len(x) for x in r["second"]) / max(1, len(r["second"]))
        print(
            f"{label:<7} rounds={r['n']:<3} fresh={r['ic']:>7,} cached={r['cc']:>7,} out={r['oc']:>5,} "
            f"price=¥{r['price']:.4f}"
        )
        print(
            f"        重复回复 全程={d_all} 后半={d_late} 近似重复(后半)={near} "
            f"平均长度 前半={avg1:.1f} 后半={avg2:.1f}"
        )

    s, l = rows["shadow"], rows["live"]
    print(
        f"\n总输入 {(l['ic'] + l['cc']):,} vs {(s['ic'] + s['cc']):,} "
        f"({(l['ic'] + l['cc']) / (s['ic'] + s['cc']) - 1:+.1%})，"
        f"价格 {l['price'] / s['price'] - 1:+.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
