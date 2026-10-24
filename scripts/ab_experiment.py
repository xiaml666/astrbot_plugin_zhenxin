"""Run a full A/B experiment on the local AstrBot: two fresh WebChat sessions.

Creates a shadow session and a live session, points ``live_allowlist`` at the
live one, sends the same prompt script to both, and records the result.

    python scripts/ab_experiment.py --tag e1 --rounds 50 \
        --set aux.rhythm_postprocess=false --out reports/ab_e1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ab_run import BASE, mint_token, send  # noqa: E402


def _req(token: str, path: str, body: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def new_session(token: str) -> str:
    data = _req(token, "/api/v1/chat/sessions/new?platform_id=webchat")
    payload = data.get("data") or {}
    return payload.get("session_id") or payload.get("id") or ""


def set_config(token: str, values: dict) -> dict:
    data = _req(
        token,
        "/api/plug/astrbot_plugin_zhenxin/page/config",
        {"values": values},
        method="POST",
    )
    if not data.get("ok"):
        raise SystemExit(f"config rejected: {data}")
    return data["config"]


def run_arm(token: str, session: str, tag: str, prompts: list[dict]) -> list[dict]:
    rows = []
    for item in prompts:
        info = send(token, session, item["text"])
        info.update({"round": item["round"], "at": time.time()})
        rows.append(info)
        print(
            f"[{tag}] r{item['round']:>2} {info['seconds']:>5.2f}s {info['status']}",
            flush=True,
        )
        if not info["status"].startswith("ok"):
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--script", default=str(Path(__file__).resolve().parents[1] / "data" / "validation_50.jsonl"))
    ap.add_argument("--set", action="append", default=[], help="dotted=json value")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    overrides = {}
    for raw in args.set:
        key, _, value = raw.partition("=")
        try:
            overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key.strip()] = value.strip()

    prompts = [
        json.loads(line)
        for line in Path(args.script).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.rounds]

    token = mint_token()
    shadow = new_session(token)
    live = new_session(token)
    print(f"sessions shadow={shadow} live={live}", flush=True)

    values = {
        "shadow_mode": True,
        "live_allowlist": [live],
        "l3.host_context_mode": "replace",
        "l0.tail_injection": True,
        "l3.window_grow_turns": 4,
        "l3.recent_char_low_water": 0.7,
        "l3.recent_turns": 8,
        "shortcircuit.enabled": False,
    }
    values.update(overrides)
    cfg = set_config(token, values)
    print("config:", json.dumps({k: cfg for k in ()}), flush=True)
    print("rhythm_postprocess=", cfg["aux"]["rhythm_postprocess"], "tail=", cfg["l0"]["tail_injection"], flush=True)

    shadow_rows = run_arm(token, shadow, f"{args.tag}-shadow", prompts)
    live_rows = run_arm(token, live, f"{args.tag}-live", prompts)

    out = Path(args.out or f"reports/ab_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "shadow_session": shadow,
                "live_session": live,
                "overrides": overrides,
                "rounds": len(prompts),
                "shadow": shadow_rows,
                "live": live_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
