"""Fit the measured per-round context growth and project a 1M-token window.

Model (documented in the answer):
  baseline: context(n) = b + k*n           (linear fit on the 50-round A/B)
            fresh(n)   = measured mean     (one new turn per round)
            cached(n)  = context(n) - fresh(n)
  plugin:   context(n) = measured mean, nearly flat
            fresh(n)   = measured mean
Price: input 2.0 / cached 0.5 / output 8.0 CNY per million (plugin config).
"""

import glob
import sqlite3
import statistics

import numpy as np

PRICE_IN, PRICE_CACHE, PRICE_OUT = 2.0, 0.5, 8.0
OUT_RESERVE = 4096
DAILY_ROUNDS = 50

DB = glob.glob(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\plugin_data\*\zhenxin.sqlite3")[0]
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row


def load(where, args=()):
    return list(
        conn.execute(
            "SELECT round_no, input_tokens_actual ic, cached_tokens_actual cc,"
            f" output_tokens_actual oc FROM metrics WHERE {where} ORDER BY id",
            args,
        )
    )


def fit(rows):
    x = np.array([r["round_no"] for r in rows], float)
    y = np.array([(r["ic"] or 0) + (r["cc"] or 0) for r in rows], float)
    k, b = np.polyfit(x, y, 1)
    return float(k), float(b)


shadow = load("mode LIKE 'shadow%'")
live = load("mode IN ('full','l1','solid')")

k_s, b_s = fit(shadow)
k_l, b_l = fit(live)
fresh_s = statistics.mean(r["ic"] for r in shadow)
fresh_l = statistics.mean(r["ic"] for r in live)
out_s = statistics.mean(r["oc"] for r in shadow)
out_l = statistics.mean(r["oc"] for r in live)

print(f"baseline  context(n) = {b_s:,.0f} + {k_s:,.1f} n   fresh/round {fresh_s:,.0f}  out {out_s:.0f}")
print(f"plugin    context(n) = {b_l:,.0f} + {k_l:,.1f} n   fresh/round {fresh_l:,.0f}  out {out_l:.0f}")


def cost_per_turn(context, fresh, out):
    cached = max(0.0, context - fresh)
    return (fresh * PRICE_IN + cached * PRICE_CACHE + out * PRICE_OUT) / 1e6


def project(window: int, limit_rounds: int = 100_000):
    cap = window - OUT_RESERVE
    if k_s > 0:
        n_s = int((cap - b_s) / k_s)
    else:
        n_s = limit_rounds
    n_s = max(1, min(n_s, limit_rounds))

    # baseline: sum over rounds of context(n); closed form for the linear model
    total_s = 0.0
    for n in range(1, n_s + 1):
        total_s += cost_per_turn(b_s + k_s * n, fresh_s, out_s)

    # plugin: cost stays flat, run it the same number of rounds (and 3x as a margin)
    for label, rounds in (("same rounds", n_s), ("3x rounds", n_s * 3)):
        total_l = sum(cost_per_turn(b_l + k_l * n, fresh_l, out_l) for n in range(1, rounds + 1))
        print(
            f"  {window//1000:>4}k  baseline dies at ~{n_s:,} rounds ({n_s/DAILY_ROUNDS:,.0f} days @50/day), "
            f"cost ¥{total_s:,.2f} | plugin {label} ¥{total_l:,.2f} "
            f"({(total_l / total_s - 1):+.1%})"
        )
    return n_s, total_s


print("\n== projection (history growth only; excludes tool/extra injections) ==")
for w in (32_768, 128_000, 200_000, 1_000_000):
    project(w)

print("\n== cost per 1,000 rounds of steady state ==")
print(f"  baseline at n=1000: context {b_s + k_s * 1000:,.0f}, "
      f"¥{cost_per_turn(b_s + k_s * 1000, fresh_s, out_s) * 1000:,.2f}")
print(f"  plugin  at n=1000: context {b_l + k_l * 1000:,.0f}, "
      f"¥{cost_per_turn(b_l + k_l * 1000, fresh_l, out_l) * 1000:,.2f}")
