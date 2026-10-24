"""A/B context-cost comparison: with Zhenxin vs without it.

The default generator is local and deterministic, so the report costs nothing
and isolates the variable we care about: what the model would see.  Set
``mode="adaptive"`` to let the Zhenxin branch use L1 recall in its reply; the
baseline branch still answers from the raw conversation only.

Token counts are estimates from the same character-based estimator used by the
plugin.  They are useful for shape/comparison, not for predicting a bill.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .engine import ZhenxinEngine
from .layers.l3 import estimate_tokens

PLUGIN_COLOR = "#0f766e"
FULL_COLOR = "#dc2626"
SLIDING_COLOR = "#d97706"


@dataclass
class TurnComparison:
    round_no: int
    user: str
    plugin_reply: str
    baseline_reply: str
    plugin_tokens: int
    full_tokens: int
    sliding_tokens: int
    plugin_chars: int
    full_chars: int
    sliding_chars: int
    mode: str
    l0_chars: int
    wake: bool
    solid: bool
    prefix_stable: bool
    plugin_context: str
    full_context: str
    sliding_context: str


@dataclass
class ComparisonResult:
    mode: str
    baseline_turns: int
    turns: list[TurnComparison] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def _fixed_reply(text: str) -> str:
    if "累" in text or "困" in text:
        return "那先歇一会儿，别硬撑。"
    if "团子" in text or "猫" in text:
        return "团子今天怎么样？"
    if "海盐" in text or "交稿" in text:
        return "海盐那边还顺利吗？"
    if "电影" in text:
        return "看个轻松的，别选太费脑的。"
    if "晚安" in text:
        return "晚安，早点睡。"
    if "早" == text.strip():
        return "早。"
    return "嗯，我在听。"


def _adaptive_reply(plan: Any) -> str:
    if plan.recall:
        return "我记得，" + plan.recall.replace("【想起】", "")[:55] + "。"
    return _fixed_reply(plan.text)


def _context_text(system: str, messages: Sequence[dict[str, str]]) -> str:
    parts = ["【system】", system, ""]
    for msg in messages:
        parts.append(f"【{msg['role']}】{msg['content']}")
    return "\n".join(parts)


def _token_count(system: str, messages: Sequence[dict[str, str]]) -> int:
    return estimate_tokens(system) + sum(estimate_tokens(m["content"]) + 4 for m in messages)


def _char_count(system: str, messages: Sequence[dict[str, str]]) -> int:
    return len(system) + sum(len(m["content"]) for m in messages)


class _Baseline:
    def __init__(self, system: str) -> None:
        self.system = system
        self.history: list[dict[str, str]] = []

    def context(self, text: str, window_turns: int | None = None) -> list[dict[str, str]]:
        history = self.history
        if window_turns is not None:
            history = history[-max(0, window_turns) * 2 :]
        return list(history) + [{"role": "user", "content": text}]

    def record(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})


def run_comparison(
    dialogue: Sequence[dict[str, Any]],
    *,
    config: Config | None = None,
    mode: str = "fixed",
    baseline_turns: int = 20,
    user_key: str = "compare",
) -> ComparisonResult:
    cfg = config or Config()
    cfg.db_path = ":memory:"
    cfg.metrics.jsonl_path = ""
    # Keep the deterministic reply identical across branches; compare context,
    # not the rhythm post-processor.
    cfg.aux.rhythm_postprocess = False
    cfg.aux.lexicon_postprocess = False
    cfg.shadow_mode = False

    engine = ZhenxinEngine(cfg)
    baseline = _Baseline(cfg.system_prompt)
    result = ComparisonResult(
        mode=mode,
        baseline_turns=baseline_turns,
        config={"system_prompt": cfg.system_prompt, "persona": cfg.persona, "mode": mode},
    )

    for item in sorted(dialogue, key=lambda r: int(r["round"])):
        round_no = int(item["round"])
        text = str(item["text"])
        plan = engine.begin_turn(user_key, text, round_no=round_no)
        plugin_messages = plan.messages
        plugin_context = _context_text(plan.stable_prefix, plugin_messages)
        # A solidified turn never reaches the model: its input cost is zero.
        if plan.solid_reply:
            plugin_tokens = 0
            plugin_chars = 0
        else:
            plugin_tokens = _token_count(plan.stable_prefix, plugin_messages)
            plugin_chars = _char_count(plan.stable_prefix, plugin_messages)

        if plan.solid_reply:
            plugin_reply = plan.solid_reply
        elif mode == "adaptive":
            plugin_reply = _adaptive_reply(plan)
        else:
            plugin_reply = _fixed_reply(text)
        completed = engine.complete_turn(plan, plugin_reply)

        full_messages = baseline.context(text, None)
        sliding_messages = baseline.context(text, baseline_turns)
        full_tokens = _token_count(baseline.system, full_messages)
        sliding_tokens = _token_count(baseline.system, sliding_messages)
        full_chars = _char_count(baseline.system, full_messages)
        sliding_chars = _char_count(baseline.system, sliding_messages)
        baseline_reply = _fixed_reply(text)
        baseline.record(text, baseline_reply)

        result.turns.append(
            TurnComparison(
                round_no=round_no,
                user=text,
                plugin_reply=completed.reply,
                baseline_reply=baseline_reply,
                plugin_tokens=plugin_tokens,
                full_tokens=full_tokens,
                sliding_tokens=sliding_tokens,
                plugin_chars=plugin_chars,
                full_chars=full_chars,
                sliding_chars=sliding_chars,
                mode=completed.trace.mode,
                l0_chars=completed.trace.l0_chars,
                wake=completed.trace.wake,
                solid=completed.trace.shortcircuit_hit,
                prefix_stable=bool(completed.trace.prefix_stable),
                plugin_context=plugin_context,
                full_context=_context_text(baseline.system, full_messages),
                sliding_context=_context_text(baseline.system, sliding_messages),
            )
        )

    totals = _totals(result.turns)
    result.totals = totals
    return result


def _totals(turns: Sequence[TurnComparison]) -> dict[str, Any]:
    n = max(1, len(turns))
    p = sum(t.plugin_tokens for t in turns)
    f = sum(t.full_tokens for t in turns)
    s = sum(t.sliding_tokens for t in turns)
    return {
        "rounds": len(turns),
        "plugin_total": p,
        "full_total": f,
        "sliding_total": s,
        "plugin_avg": round(p / n, 1),
        "full_avg": round(f / n, 1),
        "sliding_avg": round(s / n, 1),
        "plugin_peak": max((t.plugin_tokens for t in turns), default=0),
        "full_peak": max((t.full_tokens for t in turns), default=0),
        "sliding_peak": max((t.sliding_tokens for t in turns), default=0),
        "last_plugin": turns[-1].plugin_tokens if turns else 0,
        "last_full": turns[-1].full_tokens if turns else 0,
        "last_sliding": turns[-1].sliding_tokens if turns else 0,
        "saved_vs_full": round(1 - p / f, 4) if f else 0.0,
        "saved_vs_sliding": round(1 - p / s, 4) if s else 0.0,
        "solid_turns": sum(1 for t in turns if t.solid),
        "wake_turns": sum(1 for t in turns if t.wake),
        "l0_chars_max": max((t.l0_chars for t in turns), default=0),
        "prefix_stable_ratio": round(
            sum(1 for t in turns if t.prefix_stable) / max(1, len(turns)), 4
        ),
    }


def _svg_line_chart(result: ComparisonResult) -> str:
    turns = result.turns
    n = max(1, len(turns))
    width, height = 960, 340
    left, right, top, bottom = 64, 24, 30, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = (
        [t.plugin_tokens for t in turns]
        + [t.full_tokens for t in turns]
        + [t.sliding_tokens for t in turns]
    )
    max_y = max(values + [10]) * 1.12

    def x(i: int) -> float:
        return left + (plot_w * i / max(1, n - 1))

    def y(v: float) -> float:
        return top + plot_h - (plot_h * min(v, max_y) / max_y)

    bits = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="上下文 token 随轮次变化">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]
    for k in range(5):
        val = max_y * k / 4
        yy = y(val)
        bits.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        bits.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="12" fill="#6b7280">{int(val)}</text>')
    for i in range(n):
        if i % max(1, n // 8) == 0 or i == n - 1:
            bits.append(f'<text x="{x(i):.1f}" y="{height - 16}" text-anchor="middle" font-size="12" fill="#6b7280">{i + 1}</text>')
    series = [
        ("装插件", PLUGIN_COLOR, [t.plugin_tokens for t in turns]),
        ("不装 · 全量历史", FULL_COLOR, [t.full_tokens for t in turns]),
        ("不装 · 滑动窗口", SLIDING_COLOR, [t.sliding_tokens for t in turns]),
    ]
    for label, color, vals in series:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        bits.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>')
    # Legend
    lx = left
    for label, color, _ in series:
        bits.append(f'<circle cx="{lx + 6}" cy="16" r="5" fill="{color}"/>')
        bits.append(f'<text x="{lx + 16}" y="20" font-size="12" fill="#374151">{html.escape(label)}</text>')
        lx += 150
    bits.append("</svg>")
    return "".join(bits)


def _fmt(n: float | int) -> str:
    return f"{int(round(n)):,}"


def _render_chat(result: ComparisonResult) -> str:
    cards: list[str] = []
    for t in result.turns:
        token_label = "0（固化直答）" if t.solid else f"{_fmt(t.plugin_tokens)} tok"
        cards.append(
            f"""
<article class="turn">
  <div class="turn-head"><span class="round">第 {t.round_no} 轮</span><span class="user-text">{_e(t.user)}</span></div>
  <div class="bubbles">
    <section class="bubble plugin">
      <div class="bubble-title">装插件 <span class="badge teal">{t.mode}</span><span class="muted">{token_label}</span></div>
      <p>{_e(t.plugin_reply)}</p>
      {('<p class="muted">固化直答，本轮未调用模型，无上下文。</p>' if t.solid else f'<details><summary>看模型实际看到的上下文</summary><pre>{_e(_clip(t.plugin_context, 2400))}</pre></details>')}
    </section>
    <section class="bubble baseline">
      <div class="bubble-title">不装 <span class="badge red">full {_fmt(t.full_tokens)} tok</span><span class="muted">sliding {_fmt(t.sliding_tokens)} tok</span></div>
      <p>{_e(t.baseline_reply)}</p>
      <details><summary>看全量上下文</summary><pre>{_e(_clip(t.full_context, 2400))}</pre></details>
      <details><summary>看滑动窗口上下文</summary><pre>{_e(_clip(t.sliding_context, 1600))}</pre></details>
    </section>
  </div>
</article>"""
        )
    return "\n".join(cards)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "\n…（已截断）"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(result: ComparisonResult) -> str:
    t = result.totals
    saved = float(t.get("saved_vs_full", 0)) * 100
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>真心 · 装插件 vs 不装插件（上下文成本对比）</title>
<style>
  :root {{ color-scheme: light; --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f6f7fb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font: 15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-.02em; }}
  h2 {{ font-size:20px; margin:34px 0 12px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
  .card {{ background:white; border:1px solid var(--line); border-radius:14px; padding:16px 18px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:24px; font-weight:700; margin-top:2px; font-variant-numeric:tabular-nums; }}
  .card .h {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .teal {{ color:#0f766e; }} .red {{ color:#b91c1c; }} .amber {{ color:#b45309; }}
  .chart {{ background:white; border:1px solid var(--line); border-radius:14px; padding:10px 12px 0; overflow:hidden; }}
  .chart svg {{ width:100%; height:auto; display:block; }}
  .note {{ background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:12px; padding:12px 16px; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); border-radius:14px; overflow:hidden; font-size:13px; }}
  th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; }}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
  th {{ background:#f9fafb; color:#374151; font-weight:600; position:sticky; top:0; }}
  tr.solid td {{ background:#ecfdf5; }}
  .turn {{ background:white; border:1px solid var(--line); border-radius:14px; padding:14px 16px; margin:12px 0; }}
  .turn-head {{ display:flex; align-items:baseline; gap:10px; margin-bottom:10px; }}
  .round {{ flex:none; font-size:12px; color:var(--muted); background:#f3f4f6; padding:2px 8px; border-radius:999px; }}
  .user-text {{ font-weight:600; }}
  .bubbles {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .bubble {{ border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:#fcfcfd; min-width:0; }}
  .bubble.plugin {{ border-left:4px solid {PLUGIN_COLOR}; }}
  .bubble.baseline {{ border-left:4px solid {FULL_COLOR}; }}
  .bubble-title {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-bottom:6px; }}
  .bubble p {{ margin:4px 0 8px; }}
  .badge {{ font-size:11px; padding:1px 7px; border-radius:999px; background:#f3f4f6; color:#374151; }}
  .badge.teal {{ background:#ccfbf1; color:#115e59; }} .badge.red {{ background:#fee2e2; color:#991b1b; }}
  .muted {{ color:var(--muted); }}
  details summary {{ cursor:pointer; color:#2563eb; font-size:12px; }}
  pre {{ white-space:pre-wrap; word-break:break-word; font-size:11.5px; line-height:1.5; background:#0f172a; color:#e2e8f0; padding:10px; border-radius:8px; max-height:320px; overflow:auto; }}
  @media (max-width:760px) {{ .bubbles {{ grid-template-columns:1fr; }} .wrap {{ padding:20px 12px 48px; }} }}
  @media print {{ body {{ background:white; }} .turn,.card,.chart,table {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>装插件 vs 不装插件</h1>
  <p class="sub">同一段 {t.get('rounds', 0)} 轮对话 · 每轮模型可见输入对比 · 回复模式：{_e(result.mode)} · 滑窗基线：{result.baseline_turns} 轮</p>

  <div class="cards">
    <div class="card"><div class="k">装插件 · 总输入 token</div><div class="v teal">{_fmt(t.get('plugin_total', 0))}</div><div class="h">平均 {_fmt(t.get('plugin_avg', 0))}/轮</div></div>
    <div class="card"><div class="k">不装 · 全量历史</div><div class="v red">{_fmt(t.get('full_total', 0))}</div><div class="h">平均 {_fmt(t.get('full_avg', 0))}/轮</div></div>
    <div class="card"><div class="k">不装 · 滑动窗口</div><div class="v amber">{_fmt(t.get('sliding_total', 0))}</div><div class="h">平均 {_fmt(t.get('sliding_avg', 0))}/轮</div></div>
    <div class="card"><div class="k">相对全量节省</div><div class="v teal">{saved:.1f}%</div><div class="h">按估算输入 token</div></div>
    <div class="card"><div class="k">固化免调用</div><div class="v">{t.get('solid_turns', 0)} 轮</div><div class="h">这些轮输入为 0</div></div>
    <div class="card"><div class="k">L0 最大字符</div><div class="v">{t.get('l0_chars_max', 0)}</div><div class="h">≤150</div></div>
  </div>

  <h2>上下文 token 曲线</h2>
  <div class="chart">{_svg_line_chart(result)}</div>

  <div class="note" style="margin-top:14px">
    token 为字符估算：中文按 1 字≈1 token，英文/数字按 4 字符≈1 token。它不是账单，但三条曲线的相对关系是可靠的。
    真实模型下，装插件的固化轮次完全不发请求；其他轮次只发 STABLE_PREFIX + 最近窗口 + 临时尾部。
    滑窗基线只是“保留最近 N 轮”的 naive 方案：它可能比真心更便宜，但没有 L0 长期线索，也不会在固化轮次完全省掉请求。
  </div>

  <h2>逐轮明细</h2>
  <table>
    <thead><tr><th>轮</th><th>用户</th><th>装插件</th><th>全量</th><th>滑窗</th><th>模式</th><th>L0 字</th></tr></thead>
    <tbody>
    {''.join(f'<tr class="{"solid" if turn.solid else ""}"><td>{turn.round_no}</td><td>{_e(turn.user[:28])}</td><td>{_fmt(turn.plugin_tokens)}</td><td>{_fmt(turn.full_tokens)}</td><td>{_fmt(turn.sliding_tokens)}</td><td>{_e(turn.mode)}</td><td>{turn.l0_chars}</td></tr>' for turn in result.turns)}
    </tbody>
  </table>

  <h2>并排聊天</h2>
  <p class="sub">点开每轮可看模型实际收到的 system + messages。装插件栏在固化轮次显示“0（固化直答）”。</p>
  {_render_chat(result)}

  <p class="sub" style="margin-top:28px">生成命令：<code>python scripts/compare_baseline.py</code> · 数据：<code>data/sim_dialogue.jsonl</code></p>
</div>
</body>
</html>"""


def render_markdown(result: ComparisonResult) -> str:
    t = result.totals
    lines = [
        "# 装插件 vs 不装插件：上下文成本对比",
        "",
        f"- 对话轮次：{t.get('rounds', 0)}",
        f"- 回复模式：{result.mode}",
        "- token 为字符估算，不是账单；请看相对关系。",
        "",
        "| 指标 | 装插件 | 不装·全量 | 不装·滑窗 |",
        "|---|---:|---:|---:|",
        f"| 总输入 token | {_fmt(t.get('plugin_total', 0))} | {_fmt(t.get('full_total', 0))} | {_fmt(t.get('sliding_total', 0))} |",
        f"| 平均每轮 | {_fmt(t.get('plugin_avg', 0))} | {_fmt(t.get('full_avg', 0))} | {_fmt(t.get('sliding_avg', 0))} |",
        f"| 峰值单轮 | {_fmt(t.get('plugin_peak', 0))} | {_fmt(t.get('full_peak', 0))} | {_fmt(t.get('sliding_peak', 0))} |",
        f"| 末轮 | {_fmt(t.get('last_plugin', 0))} | {_fmt(t.get('last_full', 0))} | {_fmt(t.get('last_sliding', 0))} |",
        f"| 固化免调用 | {t.get('solid_turns', 0)} | - | - |",
        f"| L1 唤醒 | {t.get('wake_turns', 0)} | - | - |",
        f"| 前缀稳定率 | {float(t.get('prefix_stable_ratio', 0)):.1%} | - | - |",
        "",
        f"相对全量历史节省 **{float(t.get('saved_vs_full', 0)) * 100:.1f}%** 的估算输入 token；"
        f"相对滑动窗口节省 **{float(t.get('saved_vs_sliding', 0)) * 100:.1f}%**。",
        "",
        "完整逐轮上下文与并排聊天见 `reports/comparison.html`。",
        "",
    ]
    return "\n".join(lines)


def write_comparison(result: ComparisonResult, out_dir: str | Path = "reports") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "comparison.html"
    md_path = out / "comparison.md"
    json_path = out / "comparison.json"
    html_path.write_text(render_html(result), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    payload = asdict(result)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"html": html_path, "markdown": md_path, "json": json_path}
