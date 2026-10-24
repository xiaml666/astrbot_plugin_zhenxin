"""One-line cost summary for a session pair (E1/E2 experiments)."""

import argparse
import glob
import sqlite3

DEFAULT_DB = glob.glob(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\plugin_data\*\zhenxin.sqlite3")[0]
P = {"in": 2.0, "cache": 0.5, "out": 8.0}


def summary(db, session, label):
    key = f"webchat:FriendMessage:webchat!astrbot!{session}"
    rows = list(
        db.execute(
            "SELECT COUNT(*) n, SUM(input_tokens_actual) ic, SUM(cached_tokens_actual) cc,"
            " SUM(output_tokens_actual) oc, AVG(reply_chars) rc FROM metrics WHERE user_key=?",
            (key,),
        )
    )[0]
    ic, cc, oc = rows[1] or 0, rows[2] or 0, rows[3] or 0
    price = ic * P["in"] / 1e6 + cc * P["cache"] / 1e6 + oc * P["out"] / 1e6
    print(
        f"{label:<10} rounds={rows[0]:<3} fresh={ic:>7,} cached={cc:>7,} out={oc:>5,} "
        f"avg_reply={rows[4] or 0:5.1f} price=¥{price:.4f}"
    )
    return {"n": rows[0], "ic": ic, "cc": cc, "oc": oc, "price": price, "reply": rows[4] or 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--shadow", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--label", default="exp")
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    s = summary(db, args.shadow, f"{args.label}-base")
    l = summary(db, args.live, f"{args.label}-live")
    print(f"total input {l['ic'] + l['cc']:,} vs {s['ic'] + s['cc']:,} "
          f"({(l['ic'] + l['cc']) / (s['ic'] + s['cc']) - 1:+.1%}), price {l['price'] / s['price'] - 1:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
