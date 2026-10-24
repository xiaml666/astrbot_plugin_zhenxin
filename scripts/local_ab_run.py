"""Drive one WebChat session through the local AstrBot dashboard API.

Sends each prompt in order and waits for that run's SSE stream to close before
moving on, so rounds never overlap.

    python scripts/local_ab_run.py --session <uuid> --tag shadow --out reports/ab_shadow.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:6185"
DEFAULT_ASTROOT = Path(__file__).resolve().parents[3] / "bot-local" / "AstrBot-4.29.1"


def mint_token(astrbot_root: Path = DEFAULT_ASTROOT) -> str:
    import base64
    import hashlib
    import hmac

    cfg = json.loads((astrbot_root / "data" / "cmd_config.json").read_text(encoding="utf-8-sig"))
    secret = cfg["dashboard"]["jwt_secret"]

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"username": "astrbot", "iat": int(time.time()), "exp": int(time.time()) + 7200}
    signing = (
        b64(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + b64(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest()
    return signing + "." + b64(sig)


def send(token: str, session_id: str, text: str, timeout: float = 180.0) -> dict:
    body = json.dumps({"message": text, "session_id": session_id}).encode()
    req = urllib.request.Request(
        BASE + "/api/chat/send",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    chunks = 0
    reply_chars = 0
    status = "ok"
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            chunks += 1
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error":
                status = "error:" + str(event.get("data"))[:80]
            data = event.get("data")
            if isinstance(data, dict) and isinstance(data.get("content"), str):
                reply_chars = max(reply_chars, len(data["content"]))
    return {
        "seconds": round(time.time() - started, 2),
        "events": chunks,
        "reply_chars": reply_chars,
        "status": status,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--script", default=str(ROOT / "data" / "validation_50.jsonl"))
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--astrbot-root", default=str(DEFAULT_ASTROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = [
        json.loads(line)
        for line in Path(args.script).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.rounds]
    token = mint_token(Path(args.astrbot_root))
    results = []
    for item in prompts:
        if int(item["round"]) < args.start:
            continue
        text = item["text"]
        if args.dry_run:
            print(f"[{args.tag}] r{item['round']}: {text}")
            continue
        info = send(token, args.session, text)
        info.update({"round": item["round"], "text": text, "at": time.time()})
        results.append(info)
        print(f"[{args.tag}] r{item['round']:>2} {info['seconds']:>6.2f}s "
              f"chars={info['reply_chars']:>4} {info['status']}", flush=True)
        if not info["status"].startswith("ok"):
            break
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
