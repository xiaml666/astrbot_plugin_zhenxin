"""Summarise a shadow-vs-live A/B from the plugin DB.

Reads the same SQLite the plugin writes (``metrics`` table with real usage) and
compares the arm recorded in shadow mode against the arm the plugin took over.

    python scripts/ab_report.py --db data/plugin_data/astrbot_plugin_zhenxin/zhenxin.sqlite3
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
import statistics
from pathlib import Path

PRICE = {"in": 2.0, "cached": 0.5, "out": 8.0}
LIVE_MODES = ("full", "l1", "solid")
ROOT = Path(__file__).resolve().parents[1]


def default_db() -> str:
    patterns = [
        "data/plugin_data/*/zhenxin.sqlite3",
        "data/zhenxin.sqlite3",
        str(ROOT.parents[1] / "bot-local" / "AstrBot-4.29.1" / "data" / "plugin_data" / "*" / "zhenxin.sqlite3"),
    ]
    for pattern in patterns:
        found = glob.glob(pattern)
        if found:
            return found[0]
    return "data/zhenxin.sqlite3"


def load(conn, live: bool):
    if live:
        marks = ",".join("?" for _ in LIVE_MODES)
        sql = (
            "SELECT round_no, mode, wake, input_tokens_actual ic, cached_tokens_actual cc,"
            " output_tokens_actual oc, l0_chars, recent_chars, reply_chars, prefix_stable"
            f" FROM metrics WHERE mode IN ({marks}) ORDER BY id"
        )
        return list(conn.execute(sql, LIVE_MODES))
    sql = (
        "SELECT round_no, mode, wake, input_tokens_actual ic, cached_tokens_actual cc,"
        " output_tokens_actual oc, l0_chars, recent_chars, reply_chars, prefix_stable"
        " FROM metrics WHERE mode LIKE 'shadow%' ORDER BY id"
    )
    return list(conn.execute(sql))


def describe(name: str, rows) -> dict:
    if not rows:
        return {"arm": name, "rounds": 0}
    ic = sum(r["ic"] for r in rows)
    cc = sum(r["cc"] for r in rows)
    oc = sum(r["oc"] for r in rows)
    price = ic * PRICE["in"] / 1e6 + cc * PRICE["cached"] / 1e6 + oc * PRICE["out"] / 1e6
    drops, prev = [], None
    for r in rows:
        if prev is not None and r["cc"] < prev:
            drops.append((r["round_no"], prev - r["cc"], r["l0_chars"], r["recent_chars"]))
        prev = r["cc"]
    return {
        "arm": name,
        "rounds": len(rows),
        "input": ic,
        "cached": cc,
        "output": oc,
        "hit": cc / (cc + ic) if (cc + ic) else 0,
        "price": price,
        "wake": sum(1 for r in rows if r["wake"]) / len(rows),
        "l0_changes": sum(1 for a, b in zip(rows, rows[1:]) if a["l0_chars"] != b["l0_chars"]),
        "drops": drops,
        "drop_tokens": sum(d[1] for d in drops),
        "reply_chars": statistics.mean([r["reply_chars"] for r in rows]),
        "recent_chars_max": max(r["recent_chars"] for r in rows),
        "prefix_stable": sum(1 for r in rows if r["prefix_stable"]) / len(rows),
    }


def show(a: dict) -> None:
    if not a["rounds"]:
        print(f"\n== {a['arm']} == 无数据")
        return
    print(f"\n== {a['arm']} ==")
    print(f"轮次 {a['rounds']}  唤醒 {a['wake']:.0%}  前缀稳定 {a['prefix_stable']:.0%}")
    print(f"非缓存输入 {a['input']:,}  缓存输入 {a['cached']:,}  输出 {a['output']:,}  命中 {a['hit']:.1%}")
    print(f"价格 ¥{a['price']:.5f}  平均回复 {a['reply_chars']:.1f} 字  窗口峰值 {a['recent_chars_max']} 字")
    print(f"线索块变化 {a['l0_changes']} 次；缓存回落 {len(a['drops'])} 次，合计 {a['drop_tokens']:,} token")
    for d in a["drops"]:
        print(f"   r{d[0]}: -{d[1]:,} token (l0={d[2]}字, rec={d[3]}字)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=default_db())
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    shadow = describe("shadow 基线", load(conn, False))
    live = describe("live 真心", load(conn, True))
    show(shadow)
    show(live)
    if shadow["rounds"] and live["rounds"]:
        print("\n== 对照 ==")
        print(f"非缓存输入 {live['input'] / shadow['input'] - 1:+.1%}")
        print(f"缓存输入   {live['cached'] / shadow['cached'] - 1:+.1%}")
        print(f"总输入     {(live['input'] + live['cached']) / (shadow['input'] + shadow['cached']) - 1:+.1%}")
        print(f"价格       {live['price'] / shadow['price'] - 1:+.1%}")
        print(f"平均回复   {live['reply_chars'] / shadow['reply_chars'] - 1:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
