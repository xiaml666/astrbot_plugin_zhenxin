"""Blind pairwise quality judging with an independent model.

Each pair is judged twice with the two replies swapped, so position bias mostly
cancels: only verdicts that agree in both orders count as a win.

    python scripts/quality_judge.py --limit 2      # smoke test
    python scripts/quality_judge.py                # full run, ~10 min
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = Path(r"E:\ai相关\bot-local\AstrBot-4.29.1\data\cmd_config.json")
BASE = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

RUBRIC = """你在评审一个中文聊天机器人（扮演用户熟悉的朋友）的回复质量。
给你对话上下文、用户这句话、以及两条候选回复（顺序随机、来源隐藏）。

按下面 5 条各打 1-5 分（5 最好）：
1. fact：有没有接住上下文里的事实（项目、人名、偏好、时间），有没有说反或丢掉
2. ref：有没有接住指代与没说完的话（"那本""他们""这个"）
3. halluc：有没有编造用户没说过的事（5=完全没编造，1=明显编造）
4. human：像不像真人在聊天，有没有模板感/客服腔/机械排比
5. fit：长度与信息量是否合适（不是越短越好，也不是越长越好）

只输出 JSON：
{"A":{"fact":n,"ref":n,"halluc":n,"human":n,"fit":n},"B":{...},"winner":"A或B或tie","why":"一句话"}
"""


def api_key() -> str:
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    for src in cfg.get("provider_sources", []):
        if src.get("id") == "siliconflow":
            return (src.get("key") or [""])[0]
    raise SystemExit("no siliconflow key")


def ask(key: str, prompt: str, timeout: float = 90.0) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 400,
        }
    ).encode()
    req = urllib.request.Request(
        BASE,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def parse(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build(pair: dict, first: str, second: str) -> str:
    return (
        f"对话上下文：\n{pair.get('context', '') or '（无）'}\n\n"
        f"用户这句话：{pair['user']}\n\n"
        f"回复 A：{pair[first]}\n"
        f"回复 B：{pair[second]}\n\n"
        f"{RUBRIC}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data" / "quality_pairs_local.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "quality_judge.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pairs = [json.loads(x) for x in Path(args.pairs).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        pairs = pairs[: args.limit]
    key = api_key()

    rows = []
    for pair in pairs:
        record = {"id": pair["id"], "orders": []}
        for first, second, swapped in (
            ("baseline_reply", "plugin_reply", False),
            ("plugin_reply", "baseline_reply", True),
        ):
            started = time.time()
            try:
                parsed = parse(ask(key, build(pair, first, second)))
            except Exception as exc:  # noqa: BLE001
                parsed = None
                print(f"  {pair['id']} error {exc}", flush=True)
            record["orders"].append({"swapped": swapped, "scores": parsed})
            print(f"{pair['id']} swap={swapped} {time.time() - started:.1f}s", flush=True)
        rows.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in rows if all(o["scores"] for o in r["orders"]))
    print(f"parsed both orders for {ok}/{len(rows)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
