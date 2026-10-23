#!/usr/bin/env python
"""Replay the 60-round dialogue in live and shadow mode and emit acceptance stats.

Usage:
    python scripts/run_shadow_sim.py
    python scripts/run_shadow_sim.py --data data/sim_dialogue.jsonl --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhenxin import Config, ZhenxinEngine  # noqa: E402


class ManualClock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def set_round(self, round_no: int, seconds_per_round: int = 3600) -> None:
        self.now = 1_700_000_000.0 + (round_no - 1) * seconds_per_round


def load_dialogue(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r["round"]))


def dummy_model(plan) -> str:
    text = plan.text
    if plan.recall:
        return "我记得，" + plan.recall.replace("【想起】", "")[:55] + "。"
    if "团子" in text or "猫" in text:
        return "团子还好吗？你先别太担心。"
    if "累" in text:
        return "那先歇一会儿，别硬撑。"
    if "加班" in text or "通宵" in text:
        return "今晚别熬太狠，能收就收。"
    if "电影" in text:
        return "看个轻松的，别选太费脑的。"
    if "晚安" in text:
        return "晚安，早点睡。"
    return "嗯，我在听。"


def run(mode: str, dialogue: list[dict[str, Any]]) -> tuple[ZhenxinEngine, dict[str, Any], list[dict[str, Any]]]:
    cfg = Config(db_path=":memory:")
    cfg.metrics.jsonl_path = ""  # keep the simulation self-contained
    cfg.shadow_mode = mode == "shadow"
    clock = ManualClock()
    engine = ZhenxinEngine(cfg, clock=clock)
    for item in dialogue:
        round_no = int(item["round"])
        clock.set_round(round_no)
        engine.handle_turn("sim-user", str(item["text"]), dummy_model, round_no=round_no, shadow=mode == "shadow")
    report = engine.metrics.report("sim-user")
    rows = [dict(r) for r in engine.db.fetchall("SELECT * FROM metrics WHERE user_key=? ORDER BY round_no", ("sim-user",))]
    curve = []
    wakes = 0
    for i, row in enumerate(rows, 1):
        wakes += int(row["wake"])
        curve.append({"round": i, "cumulative_wake_rate": wakes / i, "wake": bool(row["wake"])})
    report["cumulative_wake_curve"] = curve
    report["wake_rate_at_50"] = (
        sum(1 for r in rows[:50] if int(r["wake"])) / len(rows[:50]) if len(rows) >= 50 else None
    )
    report["last10_wake_rate"] = (
        sum(1 for r in rows[-10:] if int(r["wake"])) / len(rows[-10:]) if rows else 0.0
    )
    report["shadow"] = mode == "shadow"
    solid = engine.shortcircuit.stats("sim-user")
    degraded = engine.db.fetchall(
        "SELECT signature,confidence,neg_streak,degraded_until FROM solidified_paths "
        "WHERE user_key=? AND neg_streak>=?",
        ("sim-user", cfg.shortcircuit.neg_degrade_threshold),
    )
    report["degraded_paths"] = [dict(r) for r in degraded]
    report["solidified"] = solid
    report["db"] = engine.stats("sim-user")["db"]
    return engine, report, rows


def acceptance(live: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "50+ rounds": (shadow.get("turns_alive") or 0) >= 50,
        "wake_rate_at_50 <= 20%": (shadow.get("wake_rate_at_50") or 0) <= 0.20,
        "last10 wake rate <= 20%": (shadow.get("last10_wake_rate") or 0) <= 0.20,
        "prefix_stable_last20 >= 95%": (shadow.get("prefix_stable_last20") or 0) >= 0.95,
        "L2 LLM calls == 0": shadow.get("l2_llm_calls", 0) == 0,
        "L0 chars <= 150": (shadow.get("l0_chars_max") or 0) <= 150,
        "solidified hit > 0": (live.get("solid_hits") or 0) > 0,
        "negative degrade exists": bool((live.get("solidified") or {}).get("degrade_events")),
        "wake curve declines (first 10 > last 10)": _bucket(shadow, "1-10") > (shadow.get("last10_wake_rate") or 0),
    }
    return checks


def _bucket(report: dict[str, Any], key: str) -> float:
    return float((report.get("wake_buckets", {}).get(key) or {}).get("wake_rate", 0.0))


def to_markdown(live: dict[str, Any], shadow: dict[str, Any], checks: dict[str, Any]) -> str:
    lines = [
        "# 真心插件 · 影子模式验收报告",
        "",
        "数据：`data/sim_dialogue.jsonl`（60 轮），本地哈希向量后端，无 LLM API。",
        "",
        "## 关键指标",
        "",
        "| 指标 | live | shadow | 目标 |",
        "|---|---:|---:|---|",
        f"| 轮次 | {live.get('turns_alive')} | {shadow.get('turns_alive')} | ≥ 50 |",
        f"| L1 累计唤醒率（第 50 轮） | {live.get('wake_rate_at_50', 0):.1%} | {shadow.get('wake_rate_at_50', 0):.1%} | ≤ 20% |",
        f"| 末 10 轮唤醒率 | {live.get('last10_wake_rate', 0):.1%} | {shadow.get('last10_wake_rate', 0):.1%} | 下降 |",
        f"| 前缀稳定率（末 20 轮） | {live.get('prefix_stable_last20', 0):.1%} | {shadow.get('prefix_stable_last20', 0):.1%} | ≥ 95% |",
        f"| 前缀稳定率（整体，含冷启动） | {live.get('prefix_stable_ratio', 0):.1%} | {shadow.get('prefix_stable_ratio', 0):.1%} | 参考 |",
        f"| 最长连续稳定 | {live.get('prefix_longest_streak', 0)} | {shadow.get('prefix_longest_streak', 0)} | 参考 |",
        f"| L0 最大字符数 | {live.get('l0_chars_max')} | {shadow.get('l0_chars_max')} | ≤ 150 |",
        f"| L2 LLM 调用 | {live.get('l2_llm_calls')} | {shadow.get('l2_llm_calls')} | = 0 |",
        f"| 固化命中 | {live.get('solid_hits')} | {shadow.get('solid_hits')}（would-hit） | > 0 |",
        f"| 每轮注入 token 均值 | {live.get('injected_tokens_mean', 0):.0f} | {shadow.get('injected_tokens_mean', 0):.0f} | 近常数 |",
        "",
        "## 唤醒率曲线（cumulative）",
        "",
        "| 轮次 | 10 | 20 | 30 | 40 | 50 | 60 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    curve = {c["round"]: c["cumulative_wake_rate"] for c in shadow.get("cumulative_wake_curve", [])}
    vals = [f"{curve.get(i, 0):.1%}" for i in (10, 20, 30, 40, 50, 60)]
    lines.append("| shadow | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## 分桶唤醒率",
        "",
        "| 桶 | live | shadow |",
        "|---|---:|---:|",
    ]
    for key in sorted(shadow.get("wake_buckets", {})):
        lines.append(
            f"| {key} | {_bucket(live, key):.1%} | {_bucket(shadow, key):.1%} |"
        )
    lines += ["", "## 验收检查", "", "| 检查 | 结果 |", "|---|---|"]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'✅' if ok else '❌'} |")
    lines += [
        "",
        "> 说明：shadow 模式下短路器只记录 would-hit，不改变实际回复；live 模式用于验证固化命中与负反馈降级。"
        "唤醒率按 shadow 统计；前缀稳定率按实际请求的 STABLE_PREFIX 哈希计算。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "sim_dialogue.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()
    dialogue = load_dialogue(Path(args.data))
    _, live, _ = run("live", dialogue)
    _, shadow, _ = run("shadow", dialogue)
    checks = acceptance(live, shadow)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"live": live, "shadow": shadow, "checks": checks}
    (out / "shadow_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "shadow_report.md").write_text(to_markdown(live, shadow, checks), encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items()}, ensure_ascii=False, indent=2))
    print(f"wrote {out / 'shadow_report.json'}")
    print(f"wrote {out / 'shadow_report.md'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
