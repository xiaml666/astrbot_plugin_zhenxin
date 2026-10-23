#!/usr/bin/env python
"""Estimate conversation lifespan and API cost until the context window fills.

This is a deliberately simple model:

  baseline per-turn input  ~= system + t * (user + assistant)
  plugin per-turn input    ~= stable prefix + recent window + ephemeral

The plugin's input is bounded, so once it fits it does not "fill up" with age.
The baseline grows until the context window is exhausted.

Example:
    python scripts/lifetime_cost.py
    python scripts/lifetime_cost.py --context 32768 131072 --turn-tokens 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(
    *,
    context: int,
    daily_turns: int,
    turn_tokens: int,
    system_tokens: int,
    plugin_tokens: int,
    output_tokens: int,
    reserve_output: int,
    solid_rate: float,
    input_price: float,
    output_price: float,
) -> dict:
    if turn_tokens <= 0:
        raise ValueError("turn_tokens must be > 0")
    capacity = max(0, context - reserve_output - system_tokens)
    baseline_turns = capacity // turn_tokens
    baseline_days = baseline_turns / daily_turns if daily_turns else 0.0
    plugin_fits = (plugin_tokens + reserve_output) <= context
    plugin_days = None if plugin_fits else 0.0

    # Sum over t=1..L of (system + t * turn_tokens)
    baseline_input = baseline_turns * system_tokens + turn_tokens * baseline_turns * (baseline_turns + 1) // 2
    # Plugin input is bounded; solid turns do not call the model at all.
    plugin_input = int(plugin_tokens * baseline_turns * (1.0 - solid_rate))
    baseline_output = baseline_turns * output_tokens
    plugin_output = baseline_output

    baseline_cost = baseline_input / 1_000_000 * input_price + baseline_output / 1_000_000 * output_price
    plugin_cost = plugin_input / 1_000_000 * input_price + plugin_output / 1_000_000 * output_price
    saved = (1 - plugin_cost / baseline_cost) if baseline_cost else 0.0
    return {
        "context": context,
        "baseline_turns": baseline_turns,
        "baseline_days": baseline_days,
        "plugin_fits": plugin_fits,
        "plugin_days": plugin_days,
        "baseline_input_tokens": baseline_input,
        "plugin_input_tokens": plugin_input,
        "baseline_output_tokens": baseline_output,
        "plugin_output_tokens": plugin_output,
        "baseline_cost": baseline_cost,
        "plugin_cost": plugin_cost,
        "saved_ratio": saved,
    }


def render_markdown(rows: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# 对话寿命与价格估算",
        "",
        f"- 每天轮数：{args.daily_turns}（一轮 = 一次用户输入 + 一次助手回复）",
        f"- 每轮新增 token（user + assistant）：{args.turn_tokens}",
        f"- 插件上下文预算：{args.plugin_tokens} token（稳定前缀 + 最近窗口 + 临时尾部）",
        f"- 输出预留：{args.reserve_output}；每轮输出：{args.output_tokens}",
        f"- 价格：输入 ¥{args.input_price}/百万 token，输出 ¥{args.output_price}/百万 token",
        f"- 固化免调用比例：{args.solid_rate:.1%}（按该比例把插件输入置 0）",
        "- 不含缓存折扣、工具调用、重试与平台加价。",
        "",
        "| 上下文窗口 | 不装寿命 | 装插件寿命 | 不装总价 | 装插件总价 | 价格降低 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        life = f"{r['baseline_days']:.1f} 天"
        plug_life = "不因历史增长耗尽" if r["plugin_fits"] else "0"
        lines.append(
            f"| {r['context']:,} | {life} | {plug_life} | ¥{r['baseline_cost']:.2f} | ¥{r['plugin_cost']:.2f} | {r['saved_ratio']:.1%} |"
        )
    lines += [
        "",
        "> 说明：装插件寿命显示为“不因历史增长耗尽”，表示它的输入上下文有上界；单条超长消息、宿主额外注入、工具结果等仍可能触发窗口上限。",
        "> 价格对比是在“不装方案耗尽”的同一轮数上计算；之后不装方案无法继续，而插件仍可继续。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, nargs="+", default=[8192, 32768, 131072])
    parser.add_argument("--daily-turns", type=int, default=50)
    parser.add_argument("--turn-tokens", type=int, default=60, help="每轮新增 user+assistant token")
    parser.add_argument("--system-tokens", type=int, default=50)
    parser.add_argument("--plugin-tokens", type=int, default=700, help="插件每轮有界上下文预算")
    parser.add_argument("--output-tokens", type=int, default=80)
    parser.add_argument("--reserve-output", type=int, default=4000)
    parser.add_argument("--solid-rate", type=float, default=0.05)
    parser.add_argument("--input-price", type=float, default=2.0, help="¥/百万 input token")
    parser.add_argument("--output-price", type=float, default=8.0, help="¥/百万 output token")
    parser.add_argument("--out", default="reports/lifetime_cost.md")
    args = parser.parse_args()

    rows = [
        estimate(
            context=c,
            daily_turns=args.daily_turns,
            turn_tokens=args.turn_tokens,
            system_tokens=args.system_tokens,
            plugin_tokens=args.plugin_tokens,
            output_tokens=args.output_tokens,
            reserve_output=args.reserve_output,
            solid_rate=args.solid_rate,
            input_price=args.input_price,
            output_price=args.output_price,
        )
        for c in args.context
    ]
    md = render_markdown(rows, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
